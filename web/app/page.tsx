import { redirect } from 'next/navigation';

import { auth, isAuthConfigured } from '@/auth';
import { ChatScreen } from '@/components/chat/chat-screen';

const HomePage = async () => {
  const session = isAuthConfigured() ? await auth() : null;

  if (session === null) {
    redirect('/signin?callbackUrl=/');
  }

  return <ChatScreen />;
};

export default HomePage;
