import type {
  ChatTitleClientPayload,
  ChatTitleResponse,
  ConversationRole,
  ConversationTurn,
} from '@/lib/api-types';

import {
  AuthenticationRequiredError,
  getAuthenticatedChatUserId,
} from '@/lib/authenticated-chat-user';
import { API_BASE_URL, CHAT_API_KEY } from '@/lib/env';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const MAX_TITLE_CONTENT_LENGTH = 8_000;
const MAX_TITLE_MESSAGES = 4;

type ParsePayloadResult =
  | { readonly kind: 'invalid' }
  | { readonly kind: 'ok'; readonly payload: ChatTitleClientPayload }
  | { readonly kind: 'tooLarge' };

const isRole = (value: unknown): value is ConversationRole =>
  value === 'assistant' || value === 'user';

const parseTurn = (value: unknown): ConversationTurn | null => {
  if (typeof value !== 'object' || value === null) {
    return null;
  }

  const candidate = value as Record<string, unknown>;
  const { content, role } = candidate;

  if (typeof content !== 'string' || content.trim().length === 0) {
    return null;
  }

  if (!isRole(role)) {
    return null;
  }

  return { content, role };
};

const parsePayload = (value: unknown): ParsePayloadResult => {
  if (typeof value !== 'object' || value === null) {
    return { kind: 'invalid' };
  }

  const candidate = value as Record<string, unknown>;
  const rawMessages = candidate['messages'];

  if (!Array.isArray(rawMessages) || rawMessages.length === 0) {
    return { kind: 'invalid' };
  }

  if (rawMessages.length > MAX_TITLE_MESSAGES) {
    return { kind: 'tooLarge' };
  }

  const messages: ConversationTurn[] = [];
  for (const rawMessage of rawMessages) {
    if (typeof rawMessage === 'object' && rawMessage !== null) {
      const { content } = rawMessage as Record<string, unknown>;
      if (
        typeof content === 'string' &&
        content.length > MAX_TITLE_CONTENT_LENGTH
      ) {
        return { kind: 'tooLarge' };
      }
    }

    const turn = parseTurn(rawMessage);
    if (turn === null) {
      return { kind: 'invalid' };
    }
    messages.push(turn);
  }

  const queryTransformModel =
    typeof candidate['query_transform_model'] === 'string'
      ? candidate['query_transform_model']
      : candidate['queryTransformModel'];

  return {
    kind: 'ok',
    payload: {
      messages,
      ...(typeof queryTransformModel === 'string' && {
        queryTransformModel,
      }),
    },
  };
};

const jsonError = (message: string, status: number): Response =>
  Response.json(
    { error: message },
    {
      headers: { 'content-type': 'application/json' },
      status,
    },
  );

const unauthenticated = (): Response =>
  jsonError('Authentication required', 401);

const toSchema = (payload: ChatTitleClientPayload, userId: string) => ({
  messages: payload.messages,
  ...(payload.queryTransformModel !== undefined && {
    // eslint-disable-next-line camelcase -- snake_case mirrors the Python API wire contract
    query_transform_model: payload.queryTransformModel,
  }),
  // eslint-disable-next-line camelcase -- snake_case mirrors the Python API wire contract
  user_id: userId,
});

const parseResponse = async (
  response: Response,
): Promise<ChatTitleResponse | null> => {
  let body: Partial<ChatTitleResponse>;

  try {
    body = (await response.json()) as Partial<ChatTitleResponse>;
  } catch {
    return null;
  }

  if (typeof body.title !== 'string' || body.title.trim().length === 0) {
    return null;
  }
  return { title: body.title };
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
    return jsonError('Title payload is too large.', 413);
  }

  if (result.kind === 'invalid') {
    return jsonError(
      'Invalid title payload: at least one message is required.',
      400,
    );
  }

  try {
    const userId = await getAuthenticatedChatUserId();
    const { payload } = result;

    let upstream: Response;

    try {
      upstream = await fetch(`${API_BASE_URL}/chat/title`, {
        body: JSON.stringify(toSchema(payload, userId)),
        headers: {
          'content-type': 'application/json',
          'x-api-key': CHAT_API_KEY,
        },
        method: 'POST',
        signal: req.signal,
      });
    } catch {
      return jsonError('Failed to reach the title service.', 502);
    }

    if (!upstream.ok) {
      return jsonError('The title service rejected the request.', 502);
    }

    const title = await parseResponse(upstream);

    if (title === null) {
      return jsonError('The title service returned an invalid response.', 502);
    }

    return Response.json(title, {
      headers: { 'content-type': 'application/json' },
      status: 200,
    });
  } catch (error) {
    if (error instanceof AuthenticationRequiredError) {
      return unauthenticated();
    }

    throw error;
  }
};
