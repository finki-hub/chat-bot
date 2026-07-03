import type {
  ChatTitleClientPayload,
  ChatTitleResponse,
  ConversationRole,
  ConversationTurn,
} from '@/lib/api-types';

import { API_BASE_URL } from '@/lib/env';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

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

const parsePayload = (value: unknown): ChatTitleClientPayload | null => {
  if (typeof value !== 'object' || value === null) {
    return null;
  }

  const candidate = value as Record<string, unknown>;
  const rawMessages = candidate['messages'];

  if (!Array.isArray(rawMessages) || rawMessages.length === 0) {
    return null;
  }

  const messages: ConversationTurn[] = [];
  for (const rawMessage of rawMessages) {
    const turn = parseTurn(rawMessage);
    if (turn === null) {
      return null;
    }
    messages.push(turn);
  }

  const queryTransformModel =
    typeof candidate['query_transform_model'] === 'string'
      ? candidate['query_transform_model']
      : candidate['queryTransformModel'];

  return {
    messages,
    ...(typeof queryTransformModel === 'string' && {
      queryTransformModel,
    }),
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

const toSchema = (payload: ChatTitleClientPayload) => ({
  messages: payload.messages,
  ...(payload.queryTransformModel !== undefined && {
    // eslint-disable-next-line camelcase -- snake_case mirrors the Python API wire contract
    query_transform_model: payload.queryTransformModel,
  }),
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

  const payload = parsePayload(raw);

  if (payload === null) {
    return jsonError(
      'Invalid title payload: at least one message is required.',
      400,
    );
  }

  let upstream: Response;

  try {
    upstream = await fetch(`${API_BASE_URL}/chat/title`, {
      body: JSON.stringify(toSchema(payload)),
      headers: { 'content-type': 'application/json' },
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
};
