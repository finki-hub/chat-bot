import 'server-only';

import { auth } from '@/auth';
import { createChatStateClient } from '@/lib/chat-state-client';

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

export const getAuthenticatedChatUserId = async (): Promise<string> => {
  const session = await auth();
  const user = session?.user;
  const providerSubject = user?.googleSubject;

  if (providerSubject === undefined || providerSubject.length === 0) {
    throw new AuthenticationRequiredError();
  }

  const chatUser = await createChatStateClient().upsertGoogleUser({
    avatarUrl: optionalText(user?.image),
    email: optionalText(user?.email),
    name: optionalText(user?.name),
    providerSubject,
  });

  return chatUser.id;
};
