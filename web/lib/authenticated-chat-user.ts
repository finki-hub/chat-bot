import 'server-only';

import { auth, isAuthConfigured } from '@/auth';
import {
  type ChatUserProvider,
  createChatStateClient,
} from '@/lib/chat-state-client';

export class AuthenticationRequiredError extends Error {
  constructor() {
    super('Authentication required');
    this.name = 'AuthenticationRequiredError';
  }
}

const optionalText = (value: null | string | undefined): string | undefined =>
  value === null || value === undefined || value.length === 0
    ? undefined
    : value;

const isChatUserProvider = (
  value: string | undefined,
): value is ChatUserProvider =>
  value === 'google' || value === 'microsoft-entra-id';

export const getAuthenticatedChatUserId = async (): Promise<string> => {
  if (!isAuthConfigured()) {
    throw new AuthenticationRequiredError();
  }

  const session = await auth();
  const user = session?.user;
  const provider = user?.provider;
  const providerSubject = user?.providerSubject;

  if (
    !isChatUserProvider(provider) ||
    providerSubject === undefined ||
    providerSubject.length === 0
  ) {
    throw new AuthenticationRequiredError();
  }

  const chatUser = await createChatStateClient().upsertChatUser({
    avatarUrl: optionalText(user?.image),
    email: optionalText(user?.email),
    name: optionalText(user?.name),
    provider,
    providerSubject,
  });

  return chatUser.id;
};
