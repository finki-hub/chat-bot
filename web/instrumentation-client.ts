import { posthog } from 'posthog-js';

import { getAnonUserId } from '@/lib/user';

const SHARED_CONVERSATION_PATH_PREFIX = '/share/';

// Explicit interaction metadata only, including SDK transport/session identifiers.
const EVENT_PROPERTIES = new Set([
  '$browser',
  '$browser_version',
  '$device_id',
  '$device_type',
  '$insert_id',
  '$is_identified',
  '$lib',
  '$lib_version',
  '$os',
  '$os_version',
  '$process_person_profile',
  '$screen_height',
  '$screen_width',
  '$session_id',
  '$time',
  '$viewport_height',
  '$viewport_width',
  '$window_id',
  'distinct_id',
  'inference_model',
  'message_index',
  'response_id',
  'service',
  'token',
]);

const isSharedConversationUrl = (value: string): boolean =>
  new URL(value, location.origin).pathname.startsWith(
    SHARED_CONVERSATION_PATH_PREFIX,
  );

const key = process.env['NEXT_PUBLIC_POSTHOG_KEY'];

if (
  key !== undefined &&
  key.length > 0 &&
  !location.pathname.startsWith(SHARED_CONVERSATION_PATH_PREFIX)
) {
  let distinctId: string | undefined;
  try {
    distinctId = getAnonUserId();
  } catch {
    // storage blocked
  }

  /* eslint-disable camelcase -- PostHog SDK option names are snake_case. */
  posthog.init(key, {
    api_host:
      process.env['NEXT_PUBLIC_POSTHOG_HOST'] ?? 'https://eu.i.posthog.com',
    autocapture: false,
    before_send: (event) => {
      if (event === null) {
        return null;
      }
      const currentUrl: unknown = event.properties['$current_url'];
      const pathname: unknown = event.properties['$pathname'];
      if (
        location.pathname.startsWith(SHARED_CONVERSATION_PATH_PREFIX) ||
        (typeof currentUrl === 'string' &&
          isSharedConversationUrl(currentUrl)) ||
        (typeof pathname === 'string' &&
          pathname.startsWith(SHARED_CONVERSATION_PATH_PREFIX))
      ) {
        posthog.stopSessionRecording();
        return null;
      }
      // Custom events also inherit full URLs, referrers, campaign values and
      // persisted initial-person properties from the SDK. Do not forward them.
      event.properties = Object.fromEntries(
        Object.entries(event.properties).filter(([name]) =>
          EVENT_PROPERTIES.has(name),
        ),
      );
      return event;
    },
    bootstrap:
      distinctId === undefined ? undefined : { distinctID: distinctId },
    capture_exceptions: false,
    capture_pageleave: false,
    capture_pageview: false,
    disable_session_recording: true,
    person_profiles: 'identified_only',
  });
  /* eslint-enable camelcase -- end of PostHog snake_case options. */
  posthog.register({ service: 'chat-bot-web' });
}
