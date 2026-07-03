import type { ChatTitleResponse, ModelId, MyUIMessage } from '@/lib/api-types';

import { toChatTitleRequestBody } from '@/lib/chat-title-translate';

type GenerateTitleInput = {
  readonly messages: readonly MyUIMessage[];
  readonly queryTransformModel: ModelId;
};

export const generateChatTitle = async ({
  messages,
  queryTransformModel,
}: GenerateTitleInput): Promise<null | string> => {
  const response = await fetch('/api/chat/title', {
    body: JSON.stringify(
      toChatTitleRequestBody({ messages, queryTransformModel }),
    ),
    headers: { 'content-type': 'application/json' },
    method: 'POST',
  });

  if (!response.ok) {
    return null;
  }

  const body = (await response.json()) as ChatTitleResponse;
  const title = body.title.trim();

  return title.length > 0 ? title : null;
};
