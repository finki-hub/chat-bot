import type { ChatCredentialProvider } from '@/lib/api-types';

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

type RouteContext = {
  readonly params: Promise<{ readonly provider: string }>;
};

const PROVIDERS = ['anthropic', 'google', 'openai'] as const;
const PROVIDER_SET: ReadonlySet<string> = new Set(PROVIDERS);

const isProvider = (value: string): value is ChatCredentialProvider =>
  PROVIDER_SET.has(value);

const empty = (status: number): Response => new Response(null, { status });

export const DELETE = async (
  _request: Request,
  { params }: RouteContext,
): Promise<Response> => {
  const { provider } = await params;
  if (!isProvider(provider)) {
    return empty(404);
  }

  try {
    const userId = await getAuthenticatedChatUserId();
    await createChatStateClient().deleteCredential({ provider, userId });
    return empty(204);
  } catch (error) {
    if (error instanceof AuthenticationRequiredError) {
      return empty(401);
    }
    if (error instanceof ChatStateRequestError) {
      return empty(error.status);
    }
    throw error;
  }
};
