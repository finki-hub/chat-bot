import { describe, expect, it } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import { toChatTitleRequestBody } from '@/lib/chat-title-translate';

const msg = (role: MyUIMessage['role'], ...texts: string[]): MyUIMessage => ({
  id: crypto.randomUUID(),
  parts: texts.map((text) => ({ text, type: 'text' })),
  role,
});

describe('toChatTitleRequestBody', () => {
  it('keeps a short oldest-first transcript and forwards the title model', () => {
    const messages = [
      msg('user', 'Прашање?'),
      msg('assistant', 'Одговор.'),
      msg('user', 'Дополнително?'),
      msg('assistant', 'Уште еден одговор.'),
      msg('user', 'Ова не влегува.'),
    ];

    expect(
      toChatTitleRequestBody({
        messages,
        providerModel: 'claude-sonnet-5',
        queryTransformModel: 'claude-sonnet-5',
      }),
    ).toStrictEqual({
      messages: [
        { content: 'Прашање?', role: 'user' },
        { content: 'Одговор.', role: 'assistant' },
        { content: 'Дополнително?', role: 'user' },
        { content: 'Уште еден одговор.', role: 'assistant' },
      ],
      // eslint-disable-next-line camelcase -- snake_case mirrors the Python API wire contract
      provider_model: 'claude-sonnet-5',
      // eslint-disable-next-line camelcase -- snake_case mirrors the Python API wire contract
      query_transform_model: 'claude-sonnet-5',
    });
  });

  it('uses the last assistant text part when a reset created multiple text parts', () => {
    const out = toChatTitleRequestBody({
      messages: [
        msg('user', 'Прашање?'),
        msg('assistant', 'премин', 'одговор'),
      ],
    });

    expect(out.messages.at(-1)).toStrictEqual({
      content: 'одговор',
      role: 'assistant',
    });
  });
});
