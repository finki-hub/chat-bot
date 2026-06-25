import type {
  FeedbackAck,
  FeedbackClientPayload,
  FeedbackSchema,
  FeedbackType,
} from '@/lib/api-types';

import { API_BASE_URL, CHAT_API_KEY } from '@/lib/env';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

type ValidPayload = FeedbackClientPayload & { userId: string };

const isFeedbackType = (value: unknown): value is FeedbackType =>
  value === 'dislike' || value === 'like';

const parsePayload = (value: unknown): null | ValidPayload => {
  if (typeof value !== 'object' || value === null) {
    return null;
  }

  const candidate = value as Record<string, unknown>;
  const { feedbackType, responseId, userId } = candidate;

  if (typeof responseId !== 'string' || responseId.length === 0) {
    return null;
  }

  if (typeof userId !== 'string' || userId.length === 0) {
    return null;
  }

  if (!isFeedbackType(feedbackType)) {
    return null;
  }

  return {
    feedbackType,
    responseId,
    userId,
    ...(typeof candidate['answerText'] === 'string' && {
      answerText: candidate['answerText'],
    }),
    ...(typeof candidate['inferenceModel'] === 'string' && {
      inferenceModel: candidate['inferenceModel'],
    }),
    ...(typeof candidate['questionText'] === 'string' && {
      questionText: candidate['questionText'],
    }),
  };
};

const toSchema = (payload: ValidPayload): FeedbackSchema => ({
  client: 'web',
  /* eslint-disable camelcase -- snake_case mirrors the Python API wire contract */
  feedback_type: payload.feedbackType,
  response_id: payload.responseId,
  user_id: payload.userId,
  ...(payload.answerText !== undefined && { answer_text: payload.answerText }),
  ...(payload.inferenceModel !== undefined && {
    inference_model: payload.inferenceModel,
  }),
  ...(payload.questionText !== undefined && {
    question_text: payload.questionText,
  }),
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
      'Invalid feedback payload: responseId, userId, and feedbackType (like|dislike) are required.',
      400,
    );
  }

  const schema = toSchema(payload);

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
