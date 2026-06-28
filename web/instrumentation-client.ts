import { posthog } from 'posthog-js';

import { getAnonUserId } from '@/lib/user';

const key = process.env['NEXT_PUBLIC_POSTHOG_KEY'];

if (key !== undefined && key.length > 0) {
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
    bootstrap:
      distinctId === undefined ? undefined : { distinctID: distinctId },
    capture_exceptions: true,
    capture_pageview: 'history_change',
    person_profiles: 'identified_only',
    session_recording: {
      maskAllInputs: true,
      maskTextSelector: '.ph-no-capture',
    },
  });
  /* eslint-enable camelcase -- end of PostHog snake_case options. */
  posthog.register({ service: 'chat-bot-web' });
}
