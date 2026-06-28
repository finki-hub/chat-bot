import { posthog } from 'posthog-js';

import { getAnonUserId } from '@/lib/user';

// PostHog browser init. Next.js runs this client-side before hydration, so the
// first $pageview is captured early and no SSR guard is needed.
//
// Exception capture is intentionally ON — errors are surfaced to PostHog across
// all apps per project decision. The conversation transcript and composer carry
// `.ph-no-capture`, which masks them in session replay; maskAllInputs provides
// an additional replay-layer mask so prompt/answer text is never recorded.
const key = process.env['NEXT_PUBLIC_POSTHOG_KEY'];

if (key !== undefined && key.length > 0) {
  // Reuse the existing anonymous id so the browser person matches server-side feedback.
  // Falls back to undefined when storage is blocked (strict privacy mode, etc.) and
  // PostHog generates its own distinct id instead.
  let distinctId: string | undefined;
  try {
    distinctId = getAnonUserId();
  } catch {
    // falls back to PostHog generating its own id
  }

  /* eslint-disable camelcase -- PostHog SDK option names are snake_case. */
  posthog.init(key, {
    api_host:
      process.env['NEXT_PUBLIC_POSTHOG_HOST'] ?? 'https://eu.i.posthog.com',
    autocapture: true,
    bootstrap:
      distinctId === undefined ? undefined : { distinctID: distinctId },
    capture_exceptions: true,
    person_profiles: 'always',
    session_recording: {
      maskAllInputs: true,
      maskTextSelector: '.ph-no-capture',
    },
  });
  /* eslint-enable camelcase -- end of PostHog snake_case options. */
}
