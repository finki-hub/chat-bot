import type {
  ChatCredentialProvider,
  ChatCredentialUpsert,
} from '@/lib/api-types';

import {
  AuthenticationRequiredError,
  getAuthenticatedChatUserId,
} from '@/lib/authenticated-chat-user';
import {
  ChatStateRequestError,
  createChatStateClient,
} from '@/lib/chat-state-client';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const PROVIDERS = ['anthropic', 'google', 'openai'] as const;
const PROVIDER_SET: ReadonlySet<unknown> = new Set(PROVIDERS);

const isProvider = (value: unknown): value is ChatCredentialProvider =>
  PROVIDER_SET.has(value);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const optionalText = (value: unknown): null | string | undefined => {
  if (value === null) {
    return null;
  }
  if (typeof value !== 'string') {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed.length === 0 ? null : trimmed;
};

const jsonError = (message: string, status: number): Response =>
  Response.json({ error: message }, { status });

const parseUpsert = async (
  request: Request,
): Promise<ChatCredentialUpsert | null> => {
  let raw: unknown;
  try {
    raw = await request.json();
  } catch (error) {
    if (error instanceof SyntaxError) {
      return null;
    }
    throw error;
  }
  if (!isRecord(raw) || !isProvider(raw['provider'])) {
    return null;
  }

  const apiKey = optionalText(raw['api_key'] ?? raw['apiKey']);
  if (apiKey === null || apiKey === undefined) {
    return null;
  }

  const baseUrl = optionalText(raw['base_url'] ?? raw['baseUrl']);
  return {
    apiKey,
    ...(baseUrl !== undefined && { baseUrl }),
    provider: raw['provider'],
  };
};

export const GET = async (): Promise<Response> => {
  try {
    const userId = await getAuthenticatedChatUserId();
    const credentials = await createChatStateClient().listCredentials({
      userId,
    });
    return Response.json(credentials);
  } catch (error) {
    if (error instanceof AuthenticationRequiredError) {
      return jsonError('Authentication required', 401);
    }
    if (error instanceof ChatStateRequestError) {
      return jsonError('Credential service request failed', error.status);
    }
    throw error;
  }
};

export const PUT = async (request: Request): Promise<Response> => {
  const payload = await parseUpsert(request);
  if (payload === null) {
    return jsonError('Invalid credential payload', 400);
  }

  try {
    const userId = await getAuthenticatedChatUserId();
    const credential = await createChatStateClient().upsertCredential({
      ...payload,
      userId,
    });
    return Response.json(credential);
  } catch (error) {
    if (error instanceof AuthenticationRequiredError) {
      return jsonError('Authentication required', 401);
    }
    if (error instanceof ChatStateRequestError) {
      return jsonError('Credential service request failed', error.status);
    }
    throw error;
  }
};
