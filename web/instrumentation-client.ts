import { posthog } from 'posthog-js';

import { getAnonUserId } from '@/lib/user';

const SHARED_CONVERSATION_PATH_PREFIX = '/share/';

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
    autocapture: true,
    before_send: (event) => {
      if (event === null) {
        return null;
      }
      const currentUrl: unknown = event.properties['$current_url'];
      const pathname: unknown = event.properties['$pathname'];
      if (
        (typeof currentUrl === 'string' &&
          isSharedConversationUrl(currentUrl)) ||
        (typeof pathname === 'string' &&
          pathname.startsWith(SHARED_CONVERSATION_PATH_PREFIX))
      ) {
        posthog.stopSessionRecording();
        return null;
      }
      return event;
    },
    bootstrap:
      distinctId === undefined ? undefined : { distinctID: distinctId },
    capture_exceptions: true,
    capture_pageview: 'history_change',
    person_profiles: 'identified_only',
    session_recording: {
      maskAllInputs: false,
    },
  });
  /* eslint-enable camelcase -- end of PostHog snake_case options. */
  posthog.register({ service: 'chat-bot-web' });
}
