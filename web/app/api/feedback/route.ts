import type {
  FeedbackAck,
  FeedbackClientPayload,
  FeedbackRetractionAck,
  FeedbackRetractionClientPayload,
  FeedbackRetractionSchema,
  FeedbackSchema,
  FeedbackType,
} from '@/lib/api-types';

import {
  AuthenticationRequiredError,
  getAuthenticatedChatUserId,
} from '@/lib/authenticated-chat-user';
import { API_BASE_URL, CHAT_API_KEY } from '@/lib/env';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const isFeedbackType = (value: unknown): value is FeedbackType =>
  value === 'dislike' || value === 'like';

const parsePayload = (value: unknown): FeedbackClientPayload | null => {
  if (typeof value !== 'object' || value === null) {
    return null;
  }

  const candidate = value as Record<string, unknown>;
  const { feedbackType, responseId } = candidate;

  if (typeof responseId !== 'string' || responseId.length === 0) {
    return null;
  }

  if (!isFeedbackType(feedbackType)) {
    return null;
  }
  return {
    feedbackType,
    responseId,
  };
};

const toSchema = (
  payload: FeedbackClientPayload,
  userId: string,
): FeedbackSchema => ({
  client: 'web',
  /* eslint-disable camelcase -- snake_case mirrors the Python API wire contract */
  feedback_type: payload.feedbackType,
  response_id: payload.responseId,
  user_id: userId,
  /* eslint-enable camelcase -- snake_case mirrors the Python API wire contract */
});

const parseRetractionPayload = (
  value: unknown,
): FeedbackRetractionClientPayload | null => {
  if (typeof value !== 'object' || value === null) {
    return null;
  }

  const candidate = value as Record<string, unknown>;
  const { responseId } = candidate;
  if (typeof responseId !== 'string' || responseId.length === 0) {
    return null;
  }
  return { responseId };
};

const toRetractionSchema = (
  payload: FeedbackRetractionClientPayload,
  userId: string,
): FeedbackRetractionSchema => ({
  client: 'web',
  /* eslint-disable camelcase -- snake_case mirrors the Python API wire contract */
  response_id: payload.responseId,
  user_id: userId,
  /* eslint-enable camelcase -- snake_case mirrors the Python API wire contract */
});

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
      'Invalid feedback payload: responseId and feedbackType (like|dislike) are required.',
      400,
    );
  }

  let userId: string;

  try {
    userId = await getAuthenticatedChatUserId();
  } catch (error) {
    if (error instanceof AuthenticationRequiredError) {
      return unauthenticated();
    }

    throw error;
  }

  const schema = toSchema(payload, userId);

  let upstream: Response;

  try {
    upstream = await fetch(`${API_BASE_URL}/chat/feedback`, {
      body: JSON.stringify(schema),
      headers: {
        'content-type': 'application/json',
        'x-api-key': CHAT_API_KEY,
      },
      method: 'POST',
    });
  } catch {
    return jsonError('Failed to reach the feedback service.', 502);
  }

  if (!upstream.ok) {
    return jsonError('The feedback service rejected the request.', 502);
  }

  let ack: FeedbackAck;

  try {
    ack = (await upstream.json()) as FeedbackAck;
  } catch {
    return jsonError('The feedback service returned an invalid response.', 502);
  }

  return Response.json(ack, {
    headers: { 'content-type': 'application/json' },
    status: 200,
  });
};

export const DELETE = async (req: Request): Promise<Response> => {
  let raw: unknown;
  try {
    raw = await req.json();
  } catch {
    return jsonError('Invalid JSON body', 400);
  }

  const payload = parseRetractionPayload(raw);
  if (payload === null) {
    return jsonError('Invalid feedback payload: responseId is required.', 400);
  }

  let userId: string;
  try {
    userId = await getAuthenticatedChatUserId();
  } catch (error) {
    if (error instanceof AuthenticationRequiredError) {
      return unauthenticated();
    }
    throw error;
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${API_BASE_URL}/chat/feedback`, {
      body: JSON.stringify(toRetractionSchema(payload, userId)),
      headers: {
        'content-type': 'application/json',
        'x-api-key': CHAT_API_KEY,
      },
      method: 'DELETE',
    });
  } catch {
    return jsonError('Failed to reach the feedback service.', 502);
  }

  if (upstream.status === 404) {
    return jsonError('Feedback response not found.', 404);
  }
  if (!upstream.ok) {
    return jsonError('The feedback service rejected the request.', 502);
  }

  let ack: FeedbackRetractionAck;
  try {
    ack = (await upstream.json()) as FeedbackRetractionAck;
  } catch {
    return jsonError('The feedback service returned an invalid response.', 502);
  }

  return Response.json(ack, {
    headers: { 'content-type': 'application/json' },
    status: 200,
  });
};
