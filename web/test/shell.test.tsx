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
    title: 'Втор разговор',
    updatedAt: 4,
  },
];

const noop = () => vi.fn<(...args: string[]) => void>();

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
      sidebarOpen: true,
    });
  });

  it('has sane defaults', () => {
    const s = useUiStore.getState();

    expect(s).toMatchObject({
      activeConversationId: null,
      model: CLAUDE,
      sidebarOpen: true,
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

    expect(useUiStore.getState()).toMatchObject({ sidebarOpen: false });

    act(() => {
      useUiStore.getState().setSidebarOpen(true);
    });

    expect(useUiStore.getState()).toMatchObject({ sidebarOpen: true });
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

  it('deletes a conversation', () => {
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

    expect(onDelete).toHaveBeenCalledWith('c1');
  });

  it('renames a conversation via prompt', () => {
    const onRename = vi.fn<(id: string, title: string) => void>();
    vi.spyOn(globalThis, 'prompt').mockReturnValue('Ново име');
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

    expect(onRename).toHaveBeenCalledWith('c1', 'Ново име');
  });
});

describe('Sidebar', () => {
  it('renders the new-chat button and the conversation list when open', () => {
    const onNewChat = vi.fn<() => void>();
    render(
      <Sidebar
        activeId="c1"
        conversations={rows}
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

  it('hides its content when collapsed', () => {
    render(
      <Sidebar
        activeId={null}
        conversations={rows}
        onDelete={noop()}
        onNewChat={noop()}
        onRename={noop()}
        onSelect={noop()}
        open={false}
      />,
    );

    expect(screen.queryByText(FIRST_TITLE)).not.toBeInTheDocument();
  });
});

// useChat/DefaultChatTransport only parses the AI SDK UI message stream, so the
// mock must emit that (not raw protocol-v2); responseId reaches the client via
// the start chunk's messageMetadata, not the header.
const sseChatResponse = (): Response => {
  const chunks = [
    {
      messageMetadata: {
        inferenceModel: CLAUDE,
        responseId: RESPONSE_ID,
      },
      type: 'start',
    },
    { id: 'txt-1', type: 'text-start' },
    { delta: 'Здраво!', id: 'txt-1', type: 'text-delta' },
    { id: 'txt-1', type: 'text-end' },
    { type: 'finish' },
  ];
  const body = `${chunks
    .map((c) => `data: ${JSON.stringify(c)}\n\n`)
    .join('')}data: [DONE]\n\n`;

  return new Response(body, {
    headers: {
      'content-type': 'text/event-stream',
      'X-Response-Id': RESPONSE_ID,
      'x-vercel-ai-ui-message-stream': 'v1',
    },
    status: 200,
  });
};

// These tests render ChatPage in isolation (no layout provider), so wrap each
// render in a fresh QueryClient to keep useModels supplied.
const renderChatPage = (): ReturnType<typeof rtlRender> => {
  const queryClient = new QueryClient();

  return rtlRender(
    <QueryClientProvider client={queryClient}>
      <ChatPage />
    </QueryClientProvider>,
  );
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
    // afterEach's unstubAllGlobals also clears the beforeAll ResizeObserver stub,
    // so re-install it for every render in this block (Conversation needs it).
    vi.stubGlobal('ResizeObserver', ResizeObserverStub);
    vi.stubGlobal('localStorage', createMemoryStorage());
    vi.stubGlobal(
      'fetch',
      vi.fn<(input: RequestInfo | URL) => Promise<Response>>((input) => {
        const url = urlOf(input);
        if (url.endsWith('/api/models')) {
          return Promise.resolve(
            Response.json([CLAUDE, GPT], {
              headers: { 'content-type': 'application/json' },
              status: 200,
            }),
          );
        }
        if (url.endsWith('/api/chat')) {
          return Promise.resolve(sseChatResponse());
        }

        return Promise.resolve(
          new Response('{}', {
            headers: { 'content-type': 'application/json' },
            status: 200,
          }),
        );
      }),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
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
});
