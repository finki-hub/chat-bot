import type { CaptureResult, PostHogConfig } from 'posthog-js';
import type * as PostHogModule from 'posthog-js';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

type InitPostHog = (key: string, config: Partial<PostHogConfig>) => void;

const posthog = vi.hoisted(() => ({
  init: vi.fn<InitPostHog>(),
  register: vi.fn<(properties: Readonly<Record<string, string>>) => void>(),
  stopSessionRecording: vi.fn<() => void>(),
}));

vi.mock('posthog-js', () => ({ posthog }));
vi.mock('@/lib/user', () => ({ getAnonUserId: () => 'anonymous-user' }));

const importInstrumentation = async (): Promise<void> => {
  await import('@/instrumentation-client');
};

describe('client instrumentation', () => {
  beforeEach(() => {
    vi.resetModules();
    vi.stubEnv('NEXT_PUBLIC_POSTHOG_KEY', 'test-key');
    posthog.init.mockClear();
    posthog.register.mockClear();
    posthog.stopSessionRecording.mockClear();
    history.replaceState({}, '', '/');
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    history.replaceState({}, '', '/');
  });

  it('does not initialize PostHog on a shared conversation route', async () => {
    history.replaceState({}, '', '/share/secret-token');

    await importInstrumentation();

    expect(posthog.init).not.toHaveBeenCalled();
    expect(posthog.register).not.toHaveBeenCalled();
  });

  it('disables automatic content capture and strips SDK URL enrichment from explicit events', async () => {
    await importInstrumentation();
    const config = posthog.init.mock.calls[0]?.[1];

    /* eslint-disable camelcase -- PostHog SDK field names. */
    expect(config).toMatchObject({
      autocapture: false,
      capture_exceptions: false,
      capture_pageleave: false,
      capture_pageview: false,
      disable_session_recording: true,
    });

    const beforeSend = config?.before_send;
    if (typeof beforeSend !== 'function') {
      throw new TypeError('Missing before_send');
    }
    const sentinel =
      'https://host/private?key=sk-secret&email=person@example.test';
    for (const name of ['answer_copied', 'chat_regenerated', 'chat_stopped']) {
      const result = beforeSend({
        event: name,
        properties: {
          $current_url: sentinel,
          $pathname: '/private',
          $referrer: sentinel,
          $set_once: { $initial_current_url: sentinel },
          distinct_id: 'anonymous-user',
          inference_model: 'catalog-model',
          response_id: 'opaque-response',
          utm_source: sentinel,
        },
        uuid: 'event-id',
      });

      expect(result?.event).toBe(name);
      expect(result?.properties).toStrictEqual({
        distinct_id: 'anonymous-user',
        inference_model: 'catalog-model',
        response_id: 'opaque-response',
      });
      expect(JSON.stringify(result)).not.toContain(sentinel);
    }
    /* eslint-enable camelcase -- PostHog SDK field names. */
    history.replaceState({}, '', '/share/private');

    expect(
      beforeSend({ event: 'chat_stopped', properties: {}, uuid: 'event-id' }),
    ).toBeNull();
  });

  it('drops shared-route events and stops session recording', async () => {
    await importInstrumentation();
    const config = posthog.init.mock.calls[0]?.[1];
    const beforeSend = config?.before_send;
    if (typeof beforeSend !== 'function') {
      throw new TypeError('PostHog before_send hook was not configured');
    }
    /* eslint-disable camelcase -- PostHog event property names are snake_case. */
    const event: CaptureResult = {
      event: '$pageview',
      properties: {
        $current_url: 'http://localhost:3000/share/secret-token',
      },
      uuid: 'event-id',
    };
    /* eslint-enable camelcase -- end of PostHog snake_case properties. */

    const result = beforeSend(event);

    expect(result).toBeNull();
    expect(posthog.stopSessionRecording).toHaveBeenCalledOnce();
  });

  it('filters enrichment through the installed SDK capture pipeline', async () => {
    await importInstrumentation();
    const config = posthog.init.mock.calls[0]?.[1];
    vi.stubGlobal(
      'fetch',
      vi.fn<typeof fetch>().mockResolvedValue(Response.json({})),
    );
    const xhrSend = vi
      .spyOn(XMLHttpRequest.prototype, 'send')
      .mockImplementation(() => {});
    const { PostHog } =
      await vi.importActual<typeof PostHogModule>('posthog-js');
    const sdk = new PostHog();
    /* eslint-disable camelcase -- PostHog SDK field names. */
    sdk.init('test-key', {
      ...config,
      advanced_disable_flags: true,
      disable_external_dependency_loading: true,
      persistence: 'memory',
    });
    const sentinel = 'private@example.test';
    history.replaceState({}, '', `/private?email=${sentinel}`);
    sdk.register({ private_content: sentinel, service: 'chat-bot-web' });
    const event = sdk.capture(
      'chat_stopped',
      { inference_model: 'catalog-model', response_id: 'opaque-response' },
      { send_instantly: true },
    );

    expect(event?.properties).toMatchObject({
      inference_model: 'catalog-model',
      response_id: 'opaque-response',
      service: 'chat-bot-web',
    });
    expect(JSON.stringify(event)).not.toContain(sentinel);
    expect(event?.properties).not.toHaveProperty('$current_url');
    expect(event?.properties).not.toHaveProperty('$set_once');
    expect(sdk.config.autocapture).toBe(false);
    expect(sdk.config.disable_session_recording).toBe(true);
    expect(sdk.sessionRecordingStarted()).toBe(false);

    history.replaceState({}, '', '/share/private');

    expect(sdk.capture('chat_stopped')).toBeUndefined();

    sdk.opt_out_capturing();
    await sdk.shutdown();
    xhrSend.mockRestore();
    vi.unstubAllGlobals();
    /* eslint-enable camelcase -- PostHog SDK field names. */
  });
});
