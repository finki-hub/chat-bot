import type { CaptureResult, PostHogConfig } from 'posthog-js';

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
});
