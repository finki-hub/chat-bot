import 'server-only';

import type { ChatRequestBody } from '@/lib/api-types';

import { API_BASE_URL, CHAT_API_KEY } from '@/lib/env';

type UpstreamChatRequest = {
  readonly payload: ChatRequestBody & { readonly user_id: string };
  readonly posthogDistinctId: string | undefined;
  readonly posthogSessionId: string | undefined;
  readonly signal: AbortSignal;
  readonly streamId: string;
  readonly userId: string;
};

export const requestUpstreamChatStream = async ({
  payload,
  posthogDistinctId,
  posthogSessionId,
  signal,
  streamId,
  userId,
}: UpstreamChatRequest): Promise<Response> =>
  fetch(`${API_BASE_URL}/chat/`, {
    body: JSON.stringify(payload),
    headers: {
      'content-type': 'application/json',
      'x-api-key': CHAT_API_KEY,
      'X-Distinct-Id':
        posthogDistinctId !== undefined && posthogDistinctId.length > 0
          ? posthogDistinctId
          : userId,
      ...(posthogSessionId !== undefined &&
        posthogSessionId.length > 0 && {
          'X-PostHog-Session-Id': posthogSessionId,
        }),
      'X-Response-Id': streamId,
    },
    method: 'POST',
    signal,
  });
