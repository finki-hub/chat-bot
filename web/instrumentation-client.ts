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
  /* eslint-disable camelcase -- PostHog SDK option names are snake_case. */
  posthog.init(key, {
    api_host:
      process.env['NEXT_PUBLIC_POSTHOG_HOST'] ?? 'https://eu.i.posthog.com',
    autocapture: true,
    // Reuse the existing anonymous id (the feedback endpoint stores the same one),
    // so the browser person matches server-side feedback. Never call identify().
    bootstrap: { distinctID: getAnonUserId() },
    capture_exceptions: true,
    person_profiles: 'always',
    session_recording: {
      maskAllInputs: true,
      maskTextSelector: '.ph-no-capture',
    },
  });
  /* eslint-enable camelcase -- end of PostHog snake_case options. */
}
