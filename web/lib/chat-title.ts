import type { ModelId, MyUIMessage } from '@/lib/api-types';

import { toChatTitleRequestBody } from '@/lib/chat-title-translate';

type GenerateTitleInput = {
  readonly messages: readonly MyUIMessage[];
  readonly providerModel?: ModelId;
  readonly queryTransformModel?: ModelId;
};

const parseTitle = (value: unknown): null | string => {
  if (typeof value !== 'object' || value === null) {
    return null;
  }

  if (!('title' in value)) {
    return null;
  }

  const { title } = value;

  if (typeof title !== 'string') {
    return null;
  }

  const trimmed = title.trim();

  return trimmed.length > 0 ? trimmed : null;
};

export const generateChatTitle = async ({
  messages,
  providerModel,
  queryTransformModel,
}: GenerateTitleInput): Promise<null | string> => {
  const response = await fetch('/api/chat/title', {
    body: JSON.stringify(
      toChatTitleRequestBody({ messages, providerModel, queryTransformModel }),
    ),
    headers: { 'content-type': 'application/json' },
    method: 'POST',
  });

  if (!response.ok) {
    return null;
  }

  let body: unknown;

  try {
    body = await response.json();
  } catch (error) {
    if (error instanceof SyntaxError) {
      return null;
    }
    throw error;
  }

  return parseTitle(body);
};
