import type { Metadata } from 'next';

import { notFound } from 'next/navigation';

import { SharedChatScreen } from '@/components/chat/shared-chat-screen';
import {
  parseChatStateMessages,
  sanitizeSharedMessages,
} from '@/lib/chat-history';
import {
  createChatSharingClient,
  type SharedChatConversation,
} from '@/lib/chat-sharing-client';
import { ChatStateRequestError } from '@/lib/chat-state-client';
import { t } from '@/lib/i18n';

export const dynamic = 'force-dynamic';

export const metadata: Metadata = {
  referrer: 'no-referrer',
  robots: { follow: false, index: false },
  title: t('shared.pageTitle'),
};

type SharedChatPageProps = {
  readonly params: Promise<{ readonly token: string }>;
};

const loadSharedChat = async (
  token: string,
): Promise<SharedChatConversation> => {
  try {
    return await createChatSharingClient().loadSharedConversation({
      shareToken: token,
    });
  } catch (error) {
    if (
      error instanceof ChatStateRequestError &&
      (error.status === 404 || error.status === 422)
    ) {
      notFound();
    }
    throw error;
  }
};

const SharedChatPage = async ({ params }: SharedChatPageProps) => {
  const { token } = await params;
  const shared = await loadSharedChat(token);
  const messages = sanitizeSharedMessages(
    await parseChatStateMessages(shared.messages),
  );
  return (
    <SharedChatScreen
      messages={messages}
      title={shared.conversation.title ?? t('shared.untitled')}
    />
  );
};

export default SharedChatPage;
