import { redirect } from 'next/navigation';

import { auth, isAuthConfigured, isPlaywrightAuthBypassEnabled } from '@/auth';
import { ChatScreen } from '@/components/chat/chat-screen';

export default async function HomePage() {
  if (isPlaywrightAuthBypassEnabled()) {
    return <ChatScreen />;
  }

  const session = isAuthConfigured() ? await auth() : null;

  if (session === null) {
    redirect('/signin?callbackUrl=/');
  }

  return <ChatScreen />;
}
