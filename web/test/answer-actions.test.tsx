import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { MyUIMessage } from '@/lib/api-types';

import { AnswerActions } from '@/components/chat/answer-actions';
import { createMemoryStorage } from '@/test/helpers/dom-stubs';

const assistant = (responseId?: string, text = 'Одговор'): MyUIMessage => ({
  id: 'a1',
  metadata: responseId
    ? { inferenceModel: 'claude-sonnet-4-6', responseId }
    : {},
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
  it('posts a like to /api/feedback with the anon user id and metadata', async () => {
    localStorage.setItem('finkiHub.anonUserId', 'user-xyz');
    render(
      <AnswerActions
        message={assistant('resp-9', 'Готов одговор')}
        questionText="Прашање?"
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Допаѓа' }));

    const fetchMock = fetch as unknown as ReturnType<
      typeof vi.fn<typeof fetch>
    >;

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledOnce();
    });

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];

    expect(url).toBe('/api/feedback');
    expect(JSON.parse(init.body as string)).toMatchObject({
      answerText: 'Готов одговор',
      feedbackType: 'like',
      inferenceModel: 'claude-sonnet-4-6',
      questionText: 'Прашање?',
      responseId: 'resp-9',
      userId: 'user-xyz',
    });
  });

  it('optimistically marks the chosen vote as pressed', async () => {
    render(<AnswerActions message={assistant('resp-9')} />);
    const dislike = screen.getByRole('button', { name: 'Не допаѓа' });
    fireEvent.click(dislike);

    await waitFor(() => {
      expect(dislike).toHaveAttribute('aria-pressed', 'true');
    });
  });

  it('paints the like button green once pressed', async () => {
    render(<AnswerActions message={assistant('resp-9')} />);
    const like = screen.getByRole('button', { name: 'Допаѓа' });
    fireEvent.click(like);

    await waitFor(() => {
      expect(like).toHaveAttribute('aria-pressed', 'true');
    });

    expect(like.className).toContain('text-green-600');
    expect(like.className).not.toContain('text-muted-foreground');
  });

  it('reverts the optimistic vote when the request fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn<typeof fetch>()
        .mockResolvedValue(new Response('{}', { status: 500 })),
    );
    render(<AnswerActions message={assistant('resp-9')} />);
    const like = screen.getByRole('button', { name: 'Допаѓа' });
    fireEvent.click(like);

    await waitFor(() => {
      expect(like).toHaveAttribute('aria-pressed', 'false');
    });
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
