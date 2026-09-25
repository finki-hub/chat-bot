import {
  act,
  cleanup,
  render,
  renderHook,
  screen,
  waitFor,
} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ChatErrorCode } from '@/lib/api-types';

import { Thread } from '@/components/chat/thread';
import { translateToUiStream, type UiStreamPart } from '@/lib/chat-translate';
import { parseProtocolV2 } from '@/lib/sse';
import { useUiStore } from '@/lib/ui-store';
import { useConversations } from '@/lib/use-conversations';
import { ResizeObserverStub } from '@/test/helpers/dom-stubs';

const QUESTION = 'Кога е испитот?';
const PARTIAL = 'Делумен одговор';
const RETRY = 'Обиди се повторно';
const ADD_KEY = 'Додај API клуч';
const CONVERSATION_ID = 'error-characterization';
const PRIVATE_MARKERS = [
  'Provider 401/504',
  'sk-fake-test-secret',
  'https://provider.invalid/private',
] as const;
const RAW_DETAIL = PRIVATE_MARKERS.join(' ');
const { refreshConversations } = vi.hoisted(() => ({
  refreshConversations: vi.fn<() => Promise<void>>(),
}));

// Keep the real SDK, transport, runtime, retry handler and Thread. Only unrelated
// catalog/history/list hooks and the network/analytics boundaries are replaced.
vi.mock('@/lib/use-models', () => ({
  useModels: () => ({ models: [], refetch: vi.fn<() => Promise<void>>() }),
}));
vi.mock('@/lib/use-conversation-hydration', () => ({
  useConversationHydration: () => ({
    hydratingConversation: false,
    retryHydration: vi.fn<() => void>(),
  }),
}));
vi.mock('@/lib/use-conversation-list', () => ({
  useConversationList: () => ({
    conversations: [],
    error: false,
    loading: false,
    refreshConversations,
  }),
}));
vi.mock('posthog-js', () => ({
  posthog: {
    capture: vi.fn<(event: string) => void>(),
    // eslint-disable-next-line camelcase -- matches the analytics boundary.
    get_distinct_id: vi.fn<() => string | undefined>(),
    // eslint-disable-next-line camelcase -- matches the analytics boundary.
    get_session_id: vi.fn<() => string | undefined>(),
  },
}));

const frame = (event: string, data: unknown): string =>
  `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

const responseFor = async (code: ChatErrorCode, message: string) => {
  const wire = [
    frame('reset', {}),
    ...(code === 'interrupted' ? [frame('token', { text: PARTIAL })] : []),
    frame('error', { code, message }),
    frame('done', {}),
  ].join('');
  const source = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(wire));
      controller.close();
    },
  });
  const parts: UiStreamPart[] = [];
  await translateToUiStream(
    parseProtocolV2(source),
    {
      write: (part) => {
        parts.push(part);
      },
    },
    {},
  );
  // Serialize translator output at the mocked HTTP boundary; the real SDK
  // consumes these chunks and owns message assembly and completion callbacks.
  const body = parts
    .map((part) => `data: ${JSON.stringify(part)}\n\n`)
    .join('');
  return {
    parts,
    response: new Response(`${body}data: [DONE]\n\n`, {
      headers: {
        'content-type': 'text/event-stream',
        'x-vercel-ai-ui-message-stream': 'v1',
      },
    }),
  };
};

const startScenario = async (code: ChatErrorCode, message: string) => {
  const { parts, response } = await responseFor(code, message);
  const fetchMock = vi.fn<typeof fetch>().mockImplementation((input, init) => {
    if (input === '/api/chat' && init?.method === 'POST') {
      return Promise.resolve(response.clone());
    }
    if (
      input === `/api/chat/${CONVERSATION_ID}/stream` &&
      init?.method === 'GET'
    ) {
      return Promise.resolve(new Response(null, { status: 204 }));
    }
    throw new Error('Unexpected network operation in offline characterization');
  });
  vi.stubGlobal('fetch', fetchMock);
  const hook = renderHook(() => useConversations('model-a'));
  await act(async () => {
    await expect(hook.result.current.submitMessage(QUESTION)).resolves.toBe(
      true,
    );
  });
  await waitFor(() => {
    expect(hook.result.current.status).toBe('ready');
    expect(hook.result.current.activeError?.code).toBe(code);
  });
  const onManageCredentials = vi.fn<(target: HTMLElement) => void>();
  const view = render(
    <Thread
      {...hook.result.current}
      onManageCredentials={onManageCredentials}
      onRetry={hook.result.current.retry}
    />,
  );
  const posts = () =>
    fetchMock.mock.calls.filter(([, init]) => init?.method === 'POST');

  expect(posts()).toHaveLength(1);
  expect(screen.getByText(QUESTION)).toBeVisible();
  expect(screen.getByRole('alert')).toBeVisible();

  for (const marker of PRIVATE_MARKERS) {
    expect(view.container).not.toHaveTextContent(marker);
    expect(JSON.stringify(parts)).not.toContain(marker);
    expect(JSON.stringify(hook.result.current.messages)).not.toContain(marker);
    expect(JSON.stringify(hook.result.current.activeError)).not.toContain(
      marker,
    );
  }

  return { hook, onManageCredentials, parts, posts, view };
};

describe('upstream SSE error boundary characterization', () => {
  beforeEach(() => {
    refreshConversations.mockResolvedValue();
    useUiStore.setState({
      activeConversationId: CONVERSATION_ID,
      model: 'model-a',
    });
    vi.stubGlobal('ResizeObserver', ResizeObserverStub);
  });

  afterEach(() => {
    cleanup();
    useUiStore.setState({ activeConversationId: null });
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it.each(['Request failed', RAW_DETAIL])(
    'renders no-token agent_error as generic guidance plus explicit Retry (%s)',
    async (message) => {
      const { hook, parts, posts, view } = await startScenario(
        'agent_error',
        message,
      );

      expect(parts.map((part) => part.type)).toStrictEqual([
        'start',
        'data-reset',
        'data-error',
      ]);
      expect(screen.queryByTestId('answer-text')).not.toBeInTheDocument();
      expect(screen.queryByTestId('typing-indicator')).not.toBeInTheDocument();
      expect(screen.getByRole('alert')).toHaveTextContent(
        'Се случи неочекувана грешка. Обидете се повторно.',
      );
      expect(
        screen.queryByRole('button', { name: ADD_KEY }),
      ).not.toBeInTheDocument();
      expect(
        hook.result.current.messages.at(-1)?.metadata?.error,
      ).toStrictEqual({ code: 'agent_error', message: 'Request failed' });

      const originalUser = hook.result.current.messages.find(
        (item) => item.role === 'user',
      );

      expect(originalUser).toMatchObject({
        parts: [{ text: QUESTION, type: 'text' }],
        role: 'user',
      });

      const user = userEvent.setup();
      await user.click(screen.getByRole('button', { name: RETRY }));
      await waitFor(() => {
        expect(posts()).toHaveLength(2);
        expect(hook.result.current.status).toBe('ready');
      });

      expect(
        hook.result.current.messages.filter((item) => item.role === 'user'),
      ).toStrictEqual([originalUser]);

      const body = posts().at(-1)?.[1]?.body;
      if (typeof body !== 'string') {
        throw new TypeError('Expected a JSON retry request body');
      }
      const retryBody: unknown = JSON.parse(body);

      expect(retryBody).toMatchObject({
        messages: [originalUser],
        trigger: 'regenerate-message',
      });

      view.rerender(
        <Thread
          {...hook.result.current}
          onRetry={hook.result.current.retry}
        />,
      );

      expect(screen.getByText(QUESTION)).toBeVisible();
      expect(screen.getByRole('alert')).toBeVisible();
      expect(posts()).toHaveLength(2);

      for (const marker of PRIVATE_MARKERS) {
        expect(view.container).not.toHaveTextContent(marker);
        expect(body).not.toContain(marker);
      }
    },
  );

  it('routes credential_required to credential management rather than Retry', async () => {
    const { onManageCredentials, posts } = await startScenario(
      'credential_required',
      RAW_DETAIL,
    );

    expect(screen.getByRole('alert')).toHaveTextContent(
      'За избраниот модел е потребен API клуч.',
    );
    expect(
      screen.queryByRole('button', { name: RETRY }),
    ).not.toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: ADD_KEY }));

    expect(onManageCredentials).toHaveBeenCalledOnce();
    expect(posts()).toHaveLength(1);
  });

  it('preserves interrupted partial text alongside a soft notice without Retry', async () => {
    const { hook, posts } = await startScenario('interrupted', RAW_DETAIL);

    expect(screen.getByTestId('answer-text')).toHaveTextContent(PARTIAL);
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Одговорот е прекинат.',
    );
    expect(
      screen.queryByRole('button', { name: RETRY }),
    ).not.toBeInTheDocument();
    expect(hook.result.current.messages.at(-1)?.metadata?.error?.code).toBe(
      'interrupted',
    );
    expect(posts()).toHaveLength(1);
  });
});
