import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { FeedbackType, MyUIMessage } from '@/lib/api-types';

import { AnswerActions } from '@/components/chat/answer-actions';
import { createMemoryStorage } from '@/test/helpers/dom-stubs';

const assistant = (responseId?: string, text = 'Одговор'): MyUIMessage => ({
  id: 'a1',
  metadata: responseId ? { inferenceModel: 'claude-sonnet-5', responseId } : {},
  parts: [{ text, type: 'text' }],
  role: 'assistant',
});

const ack = Response.json(
  {
    // eslint-disable-next-line camelcase -- snake_case mirrors the Python API wire contract
    feedback_type: 'like',
    id: '00000000-0000-4000-8000-000000000000',
    // eslint-disable-next-line camelcase -- snake_case mirrors the Python API wire contract
    response_id: 'r1',
  },
  { headers: { 'content-type': 'application/json' }, status: 200 },
);

const PRESSED = 'aria-pressed';
const LIKE_LABEL = 'Ми се допаѓа';

const writeText = vi.fn<(text: string) => Promise<void>>().mockResolvedValue();

beforeEach(() => {
  vi.stubGlobal('localStorage', createMemoryStorage());
  writeText.mockClear();
  vi.stubGlobal('fetch', vi.fn<typeof fetch>().mockResolvedValue(ack.clone()));
  Object.assign(navigator, { clipboard: { writeText } });
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('AnswerActions', () => {
  it('renders nothing when there is no responseId', () => {
    const { container } = render(<AnswerActions message={assistant()} />);

    expect(container).toBeEmptyDOMElement();
  });

  it('copies the answer text to the clipboard', async () => {
    render(<AnswerActions message={assistant('r1', 'Текст за копирање')} />);
    fireEvent.click(screen.getByRole('button', { name: 'Копирај' }));

    await waitFor(() => {
      expect(writeText).toHaveBeenCalledWith('Текст за копирање');
    });
  });
});

describe('AnswerActions feedback', () => {
  it('posts a like to /api/feedback with only the response id and vote', async () => {
    render(<AnswerActions message={assistant('resp-9', 'Готов одговор')} />);
    fireEvent.click(screen.getByRole('button', { name: LIKE_LABEL }));

    const fetchMock = fetch as unknown as ReturnType<
      typeof vi.fn<typeof fetch>
    >;

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledOnce();
    });

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];

    expect(url).toBe('/api/feedback');
    expect(JSON.parse(init.body as string)).toStrictEqual({
      feedbackType: 'like',
      responseId: 'resp-9',
    });
  });

  it('posts a dislike and marks it as pressed', async () => {
    render(<AnswerActions message={assistant('resp-9')} />);
    const dislike = screen.getByRole('button', { name: 'Не ми се допаѓа' });
    fireEvent.click(dislike);

    await waitFor(() => {
      expect(dislike).toHaveAttribute(PRESSED, 'true');
    });

    const fetchMock = fetch as unknown as ReturnType<
      typeof vi.fn<typeof fetch>
    >;
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];

    expect(JSON.parse(init.body as string)).toStrictEqual({
      feedbackType: 'dislike',
      responseId: 'resp-9',
    });
  });

  it('marks the like button with selected action colors once pressed', async () => {
    render(<AnswerActions message={assistant('resp-9')} />);
    const like = screen.getByRole('button', { name: LIKE_LABEL });
    fireEvent.click(like);

    await waitFor(() => {
      expect(like).toHaveAttribute(PRESSED, 'true');
    });

    expect(like.className).toContain('bg-primary');
    expect(like.className).toContain('text-primary-foreground');
    expect(like.className).not.toContain('text-muted-foreground');
  });

  it('restores the persisted vote from message metadata on mount', () => {
    const message: MyUIMessage = {
      ...assistant('resp-9'),
      metadata: { feedback: 'like', responseId: 'resp-9' },
    };
    render(<AnswerActions message={message} />);

    expect(screen.getByRole('button', { name: LIKE_LABEL })).toHaveAttribute(
      PRESSED,
      'true',
    );
  });

  it('reports the vote to onVote once the request succeeds', async () => {
    const onVote = vi.fn<(vote: FeedbackType) => void>();
    render(
      <AnswerActions
        message={assistant('resp-9')}
        onVote={onVote}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: LIKE_LABEL }));

    await waitFor(() => {
      expect(onVote).toHaveBeenCalledWith('like');
    });
  });

  it('reverts the vote and skips onVote when the request fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn<typeof fetch>()
        .mockResolvedValue(new Response('{}', { status: 500 })),
    );
    const onVote = vi.fn<(vote: FeedbackType) => void>();
    render(
      <AnswerActions
        message={assistant('resp-9')}
        onVote={onVote}
      />,
    );
    const like = screen.getByRole('button', { name: LIKE_LABEL });
    fireEvent.click(like);

    await waitFor(() => {
      expect(like).toHaveAttribute(PRESSED, 'false');
    });

    expect(onVote).not.toHaveBeenCalled();
  });

  it('disables voting while the answer is still streaming', () => {
    const onVote = vi.fn<(vote: FeedbackType) => void>();
    render(
      <AnswerActions
        message={assistant('resp-9')}
        onVote={onVote}
        pending
      />,
    );
    const like = screen.getByRole('button', { name: LIKE_LABEL });
    fireEvent.click(like);

    expect(like).toBeDisabled();
    expect(fetch).not.toHaveBeenCalled();
    expect(onVote).not.toHaveBeenCalled();
  });

  it('invokes onRegenerate when Регенерирај is clicked', () => {
    const onRegenerate = vi.fn<() => void>();
    render(
      <AnswerActions
        message={assistant('resp-9')}
        onRegenerate={onRegenerate}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Регенерирај' }));

    expect(onRegenerate).toHaveBeenCalledOnce();
  });
});
