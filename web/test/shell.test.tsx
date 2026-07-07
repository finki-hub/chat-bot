import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import {
  act,
  fireEvent,
  render,
  render as rtlRender,
  screen,
  waitFor,
  within,
} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { SessionProvider } from 'next-auth/react';
import {
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from 'vitest';

import ChatPage from '@/app/page';
import { ConversationList } from '@/components/shell/conversation-list';
import { Sidebar } from '@/components/shell/sidebar';
import { type ConversationRow, db } from '@/lib/db';
import { lastText } from '@/lib/message-parts';
import { useUiStore } from '@/lib/ui-store';
import {
  createMemoryStorage,
  ResizeObserverStub,
} from '@/test/helpers/dom-stubs';

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', ResizeObserverStub);
});

const CLAUDE = 'claude-sonnet-4-6';
const GPT = 'gpt-5.4-mini';
const RESPONSE_ID = 'resp-123';
const FIRST_TITLE = 'Прв разговор';
const SECOND_TITLE = 'Втор разговор';
const REGENERATE_CONVERSATION_ID = 'c-regenerate';
const CHAT_HISTORY_URL_PATTERN = /\/api\/chat\/[^/]+\/history$/u;
const CHAT_STOP_URL_PATTERN = /\/api\/chat\/[^/]+\/stop$/u;
const CHAT_STREAM_URL_PATTERN = /\/api\/chat\/[^/]+\/stream$/u;

const rows: ConversationRow[] = [
  {
    createdAt: 1,
    id: 'c1',
    model: CLAUDE,
    title: FIRST_TITLE,
    updatedAt: 2,
  },
  {
    createdAt: 3,
    id: 'c2',
    model: GPT,
    title: SECOND_TITLE,
    updatedAt: 4,
  },
];

const noop = () => vi.fn<(...args: string[]) => void>();

const renderSidebar = (conversations: ConversationRow[] = rows) =>
  render(
    <Sidebar
      activeId={null}
      conversations={conversations}
      onClearAll={noop()}
      onClose={noop()}
      onDelete={noop()}
      onNewChat={noop()}
      onRename={noop()}
      onSelect={noop()}
      open
    />,
  );

const urlOf = (input: RequestInfo | URL): string => {
  if (typeof input === 'string') {
    return input;
  }
  if (input instanceof URL) {
    return input.href;
  }

  return input.url;
};

describe('useUiStore', () => {
  beforeEach(() => {
    useUiStore.setState({
      activeConversationId: null,
      model: CLAUDE,
      sidebarOpen: false,
    });
  });

  it('has sane defaults', () => {
    const s = useUiStore.getState();

    expect(s).toMatchObject({
      activeConversationId: null,
      model: CLAUDE,
      sidebarOpen: false,
    });
  });

  it('updates active conversation and model', () => {
    act(() => {
      useUiStore.getState().setActiveConversationId('c1');
      useUiStore.getState().setModel(GPT);
    });

    expect(useUiStore.getState().activeConversationId).toBe('c1');
    expect(useUiStore.getState().model).toBe(GPT);
  });

  it('toggles the sidebar', () => {
    act(() => {
      useUiStore.getState().toggleSidebar();
    });

    expect(useUiStore.getState()).toMatchObject({ sidebarOpen: true });

    act(() => {
      useUiStore.getState().setSidebarOpen(false);
    });

    expect(useUiStore.getState()).toMatchObject({ sidebarOpen: false });
  });

  it('persists the model and active conversation', () => {
    const original = localStorage;
    const storage = createMemoryStorage();
    vi.stubGlobal('localStorage', storage);

    try {
      act(() => {
        useUiStore.getState().setModel(GPT);
        useUiStore.getState().setActiveConversationId('c9');
      });

      const raw = storage.getItem('finkiHub.ui');

      expect(raw).toContain(`"model":"${GPT}"`);
      expect(raw).toContain('"activeConversationId":"c9"');
    } finally {
      vi.stubGlobal('localStorage', original);
    }
  });
});

describe('ConversationList', () => {
  it('lists conversations and marks the active one', () => {
    render(
      <ConversationList
        activeId="c2"
        conversations={rows}
        onDelete={noop()}
        onRename={noop()}
        onSelect={noop()}
      />,
    );

    expect(screen.getByText(FIRST_TITLE)).toBeInTheDocument();
    expect(screen.getByTestId('conversation-c2')).toHaveAttribute(
      'aria-current',
      'true',
    );
  });

  it('selects a conversation on click', () => {
    const onSelect = vi.fn<(id: string) => void>();
    render(
      <ConversationList
        activeId={null}
        conversations={rows}
        onDelete={noop()}
        onRename={noop()}
        onSelect={onSelect}
      />,
    );
    fireEvent.click(screen.getByText(FIRST_TITLE));

    expect(onSelect).toHaveBeenCalledWith('c1');
  });

  it('deletes a conversation after confirmation', () => {
    const onDelete = vi.fn<(id: string) => void>();
    render(
      <ConversationList
        activeId={null}
        conversations={rows}
        onDelete={onDelete}
        onRename={noop()}
        onSelect={noop()}
      />,
    );
    const item = screen.getByTestId('conversation-c1');
    fireEvent.click(within(item).getByRole('button', { name: 'Избриши' }));

    expect(onDelete).not.toHaveBeenCalled();

    fireEvent.click(screen.getByTestId('confirm-action'));

    expect(onDelete).toHaveBeenCalledWith('c1');
  });

  it('renames a conversation via the dialog', () => {
    const onRename = vi.fn<(id: string, title: string) => void>();
    render(
      <ConversationList
        activeId={null}
        conversations={rows}
        onDelete={noop()}
        onRename={onRename}
        onSelect={noop()}
      />,
    );
    const item = screen.getByTestId('conversation-c1');
    fireEvent.click(within(item).getByRole('button', { name: 'Преименувај' }));

    const input = screen.getByLabelText('Ново име на разговорот');
    fireEvent.change(input, { target: { value: 'Ново име' } });
    fireEvent.click(screen.getByTestId('confirm-rename'));

    expect(onRename).toHaveBeenCalledWith('c1', 'Ново име');
  });

  it('generates a conversation title from the row actions', () => {
    const onGenerateTitle = vi.fn<(id: string) => void>();

    render(
      <ConversationList
        activeId={null}
        conversations={rows}
        onDelete={noop()}
        onGenerateTitle={onGenerateTitle}
        onRename={noop()}
        onSelect={noop()}
      />,
    );

    const item = screen.getByTestId('conversation-c1');
    fireEvent.click(
      within(item).getByRole('button', { name: 'Генерирај име' }),
    );

    expect(onGenerateTitle).toHaveBeenCalledWith('c1');
  });
});

describe('Sidebar', () => {
  it('renders the new-chat button and the conversation list when open', () => {
    const onNewChat = vi.fn<() => void>();
    render(
      <Sidebar
        activeId="c1"
        conversations={rows}
        onClearAll={noop()}
        onClose={noop()}
        onDelete={noop()}
        onNewChat={onNewChat}
        onRename={noop()}
        onSelect={noop()}
        open
      />,
    );

    expect(screen.getByText(FIRST_TITLE)).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Нов разговор' }));

    expect(onNewChat).toHaveBeenCalledOnce();
  });

  it('marks the sidebar as collapsed when closed', () => {
    const { container } = render(
      <Sidebar
        activeId={null}
        conversations={rows}
        onClearAll={noop()}
        onClose={noop()}
        onDelete={noop()}
        onNewChat={noop()}
        onRename={noop()}
        onSelect={noop()}
        open={false}
      />,
    );

    const aside = container.querySelector('aside');

    expect(aside).toHaveAttribute('data-collapsed', 'true');
    expect(aside).toHaveAttribute('aria-hidden', 'true');
    expect(aside).toHaveAttribute('inert');
  });

  it('clears all conversations after confirmation', async () => {
    const onClearAll = vi.fn<() => void>();
    const user = userEvent.setup();
    render(
      <Sidebar
        activeId="c1"
        conversations={rows}
        onClearAll={onClearAll}
        onClose={noop()}
        onDelete={noop()}
        onNewChat={noop()}
        onRename={noop()}
        onSelect={noop()}
        open
      />,
    );

    await user.click(screen.getByTestId('delete-all'));

    expect(onClearAll).not.toHaveBeenCalled();

    await user.click(screen.getByTestId('confirm-action'));

    expect(onClearAll).toHaveBeenCalledOnce();
  });

  it('hides the delete-all button when there are no conversations', () => {
    render(
      <Sidebar
        activeId={null}
        conversations={[]}
        onClearAll={noop()}
        onClose={noop()}
        onDelete={noop()}
        onNewChat={noop()}
        onRename={noop()}
        onSelect={noop()}
        open
      />,
    );

    expect(screen.queryByTestId('delete-all')).not.toBeInTheDocument();
  });

  it('filters the conversation list to substring matches as the user types', () => {
    renderSidebar();

    fireEvent.change(screen.getByRole('searchbox'), {
      target: { value: 'втор' },
    });

    expect(screen.getByText(SECOND_TITLE)).toBeInTheDocument();
    expect(screen.queryByText(FIRST_TITLE)).not.toBeInTheDocument();
  });

  it('shows a no-matches message when nothing matches the query', () => {
    renderSidebar();

    fireEvent.change(screen.getByRole('searchbox'), {
      target: { value: 'зззз' },
    });

    expect(screen.getByTestId('no-results')).toBeInTheDocument();
    expect(screen.queryByText(FIRST_TITLE)).not.toBeInTheDocument();
  });

  it('restores the full list when the search is cleared', () => {
    renderSidebar();

    fireEvent.change(screen.getByRole('searchbox'), {
      target: { value: 'втор' },
    });

    expect(screen.queryByText(FIRST_TITLE)).not.toBeInTheDocument();

    fireEvent.click(
      screen.getByRole('button', { name: 'Исчисти пребарување' }),
    );

    expect(screen.getByText(FIRST_TITLE)).toBeInTheDocument();
    expect(screen.getByText(SECOND_TITLE)).toBeInTheDocument();
  });

  it('hides the search when there are no conversations', () => {
    renderSidebar([]);

    expect(screen.queryByRole('searchbox')).not.toBeInTheDocument();
  });
});

const sseChatResponse = ({
  answer = 'Здраво!',
  responseId = RESPONSE_ID,
}: {
  answer?: string;
  responseId?: string;
} = {}): Response => {
  const chunks = [
    {
      messageMetadata: {
        inferenceModel: CLAUDE,
        responseId,
      },
      type: 'start',
    },
    { id: 'txt-1', type: 'text-start' },
    { delta: answer, id: 'txt-1', type: 'text-delta' },
    { id: 'txt-1', type: 'text-end' },
    { type: 'finish' },
  ];
  const body = `${chunks
    .map((c) => `data: ${JSON.stringify(c)}\n\n`)
    .join('')}data: [DONE]\n\n`;

  return new Response(body, {
    headers: {
      'content-type': 'text/event-stream',
      'X-Response-Id': responseId,
      'x-vercel-ai-ui-message-stream': 'v1',
    },
    status: 200,
  });
};

const jsonOk = (body: unknown): Response =>
  Response.json(body, {
    headers: { 'content-type': 'application/json' },
    status: 200,
  });

const respondTo = (
  url: string,
  chat?: { answer?: string; responseId?: string },
): Promise<Response> => {
  if (url.endsWith('/api/models')) {
    return Promise.resolve(jsonOk([CLAUDE, GPT]));
  }
  if (url.endsWith('/api/health')) {
    return Promise.resolve(jsonOk({ ok: true }));
  }
  if (url.endsWith('/api/chat')) {
    return Promise.resolve(sseChatResponse(chat));
  }
  if (CHAT_STREAM_URL_PATTERN.test(url)) {
    return Promise.resolve(new Response(null, { status: 204 }));
  }
  if (CHAT_HISTORY_URL_PATTERN.test(url)) {
    return Promise.resolve(new Response(null, { status: 404 }));
  }
  if (CHAT_STOP_URL_PATTERN.test(url)) {
    return Promise.resolve(new Response(null, { status: 204 }));
  }
  if (url.endsWith('/api/chat/title')) {
    return Promise.resolve(jsonOk({ title: 'Испитен рок' }));
  }

  return Promise.resolve(jsonOk({}));
};

const renderChatPage = (): ReturnType<typeof rtlRender> => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retryDelay: 0 } },
  });

  return rtlRender(
    <SessionProvider>
      <QueryClientProvider client={queryClient}>
        <ChatPage />
      </QueryClientProvider>
    </SessionProvider>,
  );
};

const persistedAssistantFeedback = async (): Promise<string | undefined> => {
  const stored = await db.messages.toArray();

  return stored.find((m) => m.role === 'assistant')?.metadata?.feedback;
};

describe('ChatPage persistence', () => {
  beforeEach(async () => {
    await db.delete();
    await db.open();
    useUiStore.setState({
      activeConversationId: null,
      model: CLAUDE,
      sidebarOpen: true,
    });
    vi.stubGlobal('ResizeObserver', ResizeObserverStub);
    vi.stubGlobal('localStorage', createMemoryStorage());
    vi.stubGlobal(
      'fetch',
      vi.fn<(input: RequestInfo | URL) => Promise<Response>>((input) =>
        respondTo(urlOf(input)),
      ),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('shows the unavailable banner and disables the composer when the backend is down', async () => {
    vi.mocked(fetch).mockImplementation((input: RequestInfo | URL) => {
      const url = urlOf(input);
      if (url.endsWith('/api/health')) {
        return Promise.resolve(Response.json({ ok: false }, { status: 503 }));
      }

      return respondTo(url);
    });

    renderChatPage();

    await expect(
      screen.findByTestId('service-banner'),
    ).resolves.toBeInTheDocument();
    expect(screen.getByTestId('composer-input')).toBeDisabled();
    expect(screen.getByTestId('composer-submit')).toBeDisabled();
  });

  it('sends a message, renders the streamed answer, and persists to Dexie', async () => {
    const user = userEvent.setup();
    renderChatPage();

    await user.type(screen.getByRole('textbox'), 'Прашање?');
    await user.keyboard('{Enter}');

    await expect(screen.findByText('Прашање?')).resolves.toBeInTheDocument();
    await expect(screen.findByText('Здраво!')).resolves.toBeInTheDocument();

    await waitFor(async () => {
      const pending = await db.conversations.toArray();

      expect(pending).toHaveLength(1);
    });
    const convos = await db.conversations.toArray();
    const first = convos.at(0);

    expect(first).toBeDefined();

    if (!first) {
      throw new Error('expected a persisted conversation');
    }

    const msgs = await db.messages
      .where('conversationId')
      .equals(first.id)
      .toArray();
    const roles = msgs.map((m) => m.role);

    expect(roles).toContain('user');
    expect(roles).toContain('assistant');
    expect(msgs.find((m) => m.role === 'assistant')?.metadata?.responseId).toBe(
      RESPONSE_ID,
    );
  });

  it('generates and persists a title after the first message is sent', async () => {
    const user = userEvent.setup();

    renderChatPage();

    await user.type(screen.getByRole('textbox'), 'Кога е испитот?');
    await user.keyboard('{Enter}');

    await waitFor(
      async () => {
        const convos = await db.conversations.toArray();

        expect(convos.at(0)?.title).toBe('Испитен рок');
      },
      { timeout: 5_000 },
    );
  });

  it('persists a like and restores it after a remount (refresh)', async () => {
    const now = Date.now();
    await db.conversations.put({
      createdAt: now,
      id: 'c-like',
      model: CLAUDE,
      title: 'Оценет разговор',
      updatedAt: now,
    });
    await db.messages.bulkPut([
      {
        conversationId: 'c-like',
        createdAt: now,
        id: 'u-like',
        parts: [{ text: 'Прашање за оцена', type: 'text' }],
        role: 'user',
      },
      {
        conversationId: 'c-like',
        createdAt: now + 1,
        id: 'a-like',
        metadata: { responseId: 'resp-like' },
        parts: [{ text: 'Одговор за оценување', type: 'text' }],
        role: 'assistant',
      },
    ]);
    useUiStore.setState({
      activeConversationId: 'c-like',
      model: CLAUDE,
      sidebarOpen: true,
    });

    const user = userEvent.setup();
    const view = renderChatPage();

    await user.click(await screen.findByTestId('like-button'));

    await waitFor(async () => {
      await expect(persistedAssistantFeedback()).resolves.toBe('like');
    });

    // A refresh re-hydrates from Dexie; the vote must come back pressed.
    view.unmount();
    renderChatPage();

    await waitFor(() => {
      expect(screen.getByTestId('like-button')).toHaveAttribute(
        'aria-pressed',
        'true',
      );
    });
  });

  it('regenerates an assistant answer in place and persists only the replacement', async () => {
    const now = Date.now();
    await db.conversations.put({
      createdAt: now,
      id: REGENERATE_CONVERSATION_ID,
      model: CLAUDE,
      title: 'Прашање?',
      updatedAt: now,
    });
    await db.messages.bulkPut([
      {
        conversationId: REGENERATE_CONVERSATION_ID,
        createdAt: now,
        id: 'u-regenerate',
        parts: [{ text: 'Прашање?', type: 'text' }],
        role: 'user',
      },
      {
        conversationId: REGENERATE_CONVERSATION_ID,
        createdAt: now + 1,
        id: 'a-regenerate',
        metadata: { responseId: 'resp-old' },
        parts: [{ text: 'Стар одговор', type: 'text' }],
        role: 'assistant',
      },
    ]);
    useUiStore.setState({
      activeConversationId: REGENERATE_CONVERSATION_ID,
      model: CLAUDE,
      sidebarOpen: true,
    });

    vi.mocked(fetch).mockImplementation((input: RequestInfo | URL) =>
      respondTo(urlOf(input), {
        answer: 'Нов одговор',
        responseId: 'resp-new',
      }),
    );

    const user = userEvent.setup();
    renderChatPage();

    await expect(
      screen.findByText('Стар одговор'),
    ).resolves.toBeInTheDocument();

    await user.click(
      await screen.findByRole('button', { name: 'Регенерирај' }),
    );

    await expect(screen.findByText('Нов одговор')).resolves.toBeInTheDocument();

    const regenerated = await db.messages
      .where('conversationId')
      .equals(REGENERATE_CONVERSATION_ID)
      .sortBy('createdAt');
    const assistants = regenerated.filter((m) => m.role === 'assistant');

    expect(assistants).toHaveLength(1);
    expect(assistants[0]?.id).toBe('a-regenerate');
    expect(assistants[0] ? lastText(assistants[0]) : null).toBe('Нов одговор');
    expect(assistants[0]?.metadata?.responseId).toBe('resp-new');
  });

  it('prunes trailing messages when regenerating a middle answer and persists only the survivors', async () => {
    const now = Date.now();
    const convId = 'c-multi-regenerate';
    await db.conversations.put({
      createdAt: now,
      id: convId,
      model: CLAUDE,
      title: 'Повеќе пораки',
      updatedAt: now,
    });
    await db.messages.bulkPut([
      {
        conversationId: convId,
        createdAt: now,
        id: 'u1',
        parts: [{ text: 'Прво прашање', type: 'text' }],
        role: 'user',
      },
      {
        conversationId: convId,
        createdAt: now + 1,
        id: 'a1',
        metadata: { responseId: 'resp-a1' },
        parts: [{ text: 'Прв стар одговор', type: 'text' }],
        role: 'assistant',
      },
      {
        conversationId: convId,
        createdAt: now + 2,
        id: 'u2',
        parts: [{ text: 'Второ прашање', type: 'text' }],
        role: 'user',
      },
      {
        conversationId: convId,
        createdAt: now + 3,
        id: 'a2',
        metadata: { responseId: 'resp-a2' },
        parts: [{ text: 'Втор стар одговор', type: 'text' }],
        role: 'assistant',
      },
    ]);
    useUiStore.setState({
      activeConversationId: convId,
      model: CLAUDE,
      sidebarOpen: true,
    });

    vi.mocked(fetch).mockImplementation((input: RequestInfo | URL) =>
      respondTo(urlOf(input), {
        answer: 'Регенериран прв',
        responseId: 'resp-a1-new',
      }),
    );

    const user = userEvent.setup();
    renderChatPage();

    await expect(
      screen.findByText('Втор стар одговор'),
    ).resolves.toBeInTheDocument();

    const regenerateButtons = await screen.findAllByRole('button', {
      name: 'Регенерирај',
    });
    const firstRegenerate = regenerateButtons[0];
    if (!firstRegenerate) {
      throw new Error('expected a regenerate button for the first answer');
    }
    await user.click(firstRegenerate);

    await expect(
      screen.findByText('Регенериран прв'),
    ).resolves.toBeInTheDocument();

    await waitFor(async () => {
      const count = await db.messages
        .where('conversationId')
        .equals(convId)
        .count();

      expect(count).toBe(2);
    });

    const persisted = await db.messages
      .where('conversationId')
      .equals(convId)
      .sortBy('createdAt');
    const ids = persisted.map((row) => row.id);
    const a1 = persisted.find((m) => m.id === 'a1');

    expect(ids).toStrictEqual(['u1', 'a1']);
    expect(a1 ? lastText(a1) : null).toBe('Регенериран прв');
    expect(a1?.metadata?.responseId).toBe('resp-a1-new');
  });

  it('hydrates an existing conversation on mount', async () => {
    const now = Date.now();
    await db.conversations.put({
      createdAt: now,
      id: 'cX',
      model: CLAUDE,
      title: 'Стар разговор',
      updatedAt: now,
    });
    await db.messages.bulkPut([
      {
        conversationId: 'cX',
        createdAt: now,
        id: 'mU',
        parts: [{ text: 'Старо прашање', type: 'text' }],
        role: 'user',
      },
      {
        conversationId: 'cX',
        createdAt: now + 1,
        id: 'mA',
        metadata: { responseId: 'r-old' },
        parts: [{ text: 'Стар одговор', type: 'text' }],
        role: 'assistant',
      },
    ]);
    useUiStore.setState({
      activeConversationId: 'cX',
      model: CLAUDE,
      sidebarOpen: true,
    });

    renderChatPage();

    await expect(
      screen.findByText('Стар одговор'),
    ).resolves.toBeInTheDocument();
  });

  it('hydrates completed history from the server when IndexedDB was cleared', async () => {
    useUiStore.setState({
      activeConversationId: 'c-server',
      model: CLAUDE,
      sidebarOpen: true,
    });
    vi.mocked(fetch).mockImplementation((input: RequestInfo | URL) => {
      const url = urlOf(input);
      if (url.endsWith('/api/chat/c-server/history')) {
        return Promise.resolve(
          jsonOk({
            conversation: {
              id: 'c-server',
              model: CLAUDE,
              title: 'Серверски разговор',
            },
            messages: [
              {
                id: 'u-server',
                metadata: {},
                parts: [{ text: 'Серверско прашање', type: 'text' }],
                role: 'user',
              },
              {
                id: 'a-server',
                metadata: { responseId: 'resp-server' },
                parts: [{ text: 'Серверски одговор', type: 'text' }],
                role: 'assistant',
              },
            ],
          }),
        );
      }

      return respondTo(url);
    });

    renderChatPage();

    await expect(
      screen.findByText('Серверски одговор'),
    ).resolves.toBeInTheDocument();
    await expect(db.conversations.get('c-server')).resolves.toMatchObject({
      id: 'c-server',
      model: CLAUDE,
      title: 'Серверски разговор',
    });
    await expect(
      db.messages.where('conversationId').equals('c-server').count(),
    ).resolves.toBe(2);
  });
});
