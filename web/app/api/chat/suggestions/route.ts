import type {
  ChatSuggestionsClientPayload,
  ChatSuggestionsResponse,
} from '@/lib/api-types';

import {
  AuthenticationRequiredError,
  getAuthenticatedChatUserId,
} from '@/lib/authenticated-chat-user';
import {
  MAX_SUGGESTION_ANSWER_LENGTH,
  MAX_SUGGESTION_QUESTION_LENGTH,
  MAX_SUGGESTIONS,
} from '@/lib/chat-suggestions';
import { API_BASE_URL, CHAT_API_KEY } from '@/lib/env';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

type ParsePayloadResult =
  | { readonly kind: 'invalid' }
  | { readonly kind: 'ok'; readonly payload: ChatSuggestionsClientPayload }
  | { readonly kind: 'tooLarge' };

const nonEmpty = (value: unknown): value is string =>
  typeof value === 'string' && value.trim().length > 0;

const parsePayload = (value: unknown): ParsePayloadResult => {
  if (typeof value !== 'object' || value === null) {
    return { kind: 'invalid' };
  }

  const { answer, providerModel, question } = value as Record<string, unknown>;

  if (!nonEmpty(question) || !nonEmpty(answer)) {
    return { kind: 'invalid' };
  }

  if (
    question.length > MAX_SUGGESTION_QUESTION_LENGTH ||
    answer.length > MAX_SUGGESTION_ANSWER_LENGTH
  ) {
    return { kind: 'tooLarge' };
  }

  return {
    kind: 'ok',
    payload: {
      answer,
      question,
      ...(typeof providerModel === 'string' && { providerModel }),
    },
  };
};

const jsonError = (message: string, status: number): Response =>
  Response.json({ error: message }, { status });

const parseResponse = async (
  response: Response,
): Promise<ChatSuggestionsResponse | null> => {
  let body: unknown;

  try {
    body = await response.json();
  } catch {
    return null;
  }

  if (typeof body !== 'object' || body === null || !('suggestions' in body)) {
    return null;
  }

  const { suggestions } = body;

  return Array.isArray(suggestions)
    ? {
        suggestions: suggestions
          .filter((item): item is string => nonEmpty(item))
          .slice(0, MAX_SUGGESTIONS),
      }
    : null;
};

export const POST = async (req: Request): Promise<Response> => {
  let raw: unknown;

  try {
    raw = await req.json();
  } catch {
    return jsonError('Invalid JSON body', 400);
  }

  const result = parsePayload(raw);

  if (result.kind === 'tooLarge') {
    return jsonError('Suggestions payload is too large.', 413);
  }

  if (result.kind === 'invalid') {
    return jsonError(
      'Invalid suggestions payload: a question and an answer are required.',
      400,
    );
  }

  try {
    const userId = await getAuthenticatedChatUserId();
    const { payload } = result;
    const conversationId = req.headers.get('x-conversation-id');

    let upstream: Response;

    try {
      upstream = await fetch(`${API_BASE_URL}/chat/suggestions`, {
        body: JSON.stringify({
          answer: payload.answer,
          question: payload.question,
          ...(payload.providerModel !== undefined && {
            // eslint-disable-next-line camelcase -- snake_case mirrors the Python API wire contract
            provider_model: payload.providerModel,
          }),
          // eslint-disable-next-line camelcase -- snake_case mirrors the Python API wire contract
          user_id: userId,
        }),
        headers: {
          'content-type': 'application/json',
          'x-api-key': CHAT_API_KEY,
          ...(conversationId !== null && {
            'x-conversation-id': conversationId,
          }),
        },
        method: 'POST',
        signal: req.signal,
      });
    } catch {
      return jsonError('Failed to reach the suggestions service.', 502);
    }

    if (!upstream.ok) {
      return jsonError('The suggestions service rejected the request.', 502);
    }

    const suggestions = await parseResponse(upstream);

    return suggestions === null
      ? jsonError('The suggestions service returned an invalid response.', 502)
      : Response.json(suggestions, { status: 200 });
  } catch (error) {
    if (error instanceof AuthenticationRequiredError) {
      return jsonError('Authentication required', 401);
    }

    throw error;
  }
};
