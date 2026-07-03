import type {
  ChatTitleRequestBody,
  ConversationTurn,
  ModelId,
  MyUIMessage,
} from '@/lib/api-types';

import { joinText, lastText } from '@/lib/message-parts';

const TITLE_CONTEXT_TURNS = 4;

type ChatTitleTranslateInput = {
  readonly messages: readonly MyUIMessage[];
  readonly queryTransformModel?: ModelId;
};

const toTurn = (message: MyUIMessage): ConversationTurn => {
  const role = message.role === 'assistant' ? 'assistant' : 'user';
  const content =
    (role === 'assistant' ? lastText(message) : joinText(message)) ?? '';

  return { content, role };
};

export const toChatTitleRequestBody = (
  body: ChatTitleTranslateInput,
): ChatTitleRequestBody => ({
  messages: body.messages.slice(0, TITLE_CONTEXT_TURNS).map(toTurn),
  ...(body.queryTransformModel !== undefined && {
    // eslint-disable-next-line camelcase -- snake_case mirrors the Python API wire contract
    query_transform_model: body.queryTransformModel,
  }),
});
