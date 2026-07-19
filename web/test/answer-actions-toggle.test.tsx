import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { FeedbackType, MyUIMessage } from '@/lib/api-types';

import { AnswerActions } from '@/components/chat/answer-actions';

const LIKE_LABEL = 'Ми се допаѓа';
const DISLIKE_LABEL = 'Не ми се допаѓа';
const FEEDBACK_CASES: ReadonlyArray<readonly [FeedbackType, string]> = [
  ['like', LIKE_LABEL],
  ['dislike', DISLIKE_LABEL],
];

const assistant = (feedback: FeedbackType): MyUIMessage => ({
  id: 'a1',
  metadata: { feedback, responseId: 'resp-9' },
  parts: [{ text: 'Одговор', type: 'text' }],
  role: 'assistant',
});

beforeEach(() => {
  vi.stubGlobal(
    'fetch',
    vi.fn<typeof fetch>().mockResolvedValue(
      Response.json(
        {
          // eslint-disable-next-line camelcase -- snake_case mirrors the Python API wire contract
          feedback_type: null,
          // eslint-disable-next-line camelcase -- snake_case mirrors the Python API wire contract
          response_id: 'resp-9',
        },
        { status: 200 },
      ),
    ),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('AnswerActions feedback toggles', () => {
  it.each(FEEDBACK_CASES)(
    'retracts an already selected %s',
    async (feedback, label) => {
      const onVote = vi.fn<(vote: FeedbackType | null) => void>();
      render(
        <AnswerActions
          message={assistant(feedback)}
          onVote={onVote}
        />,
      );
      const button = screen.getByRole('button', { name: label });

      fireEvent.click(button);

      await waitFor(() => {
        expect(button).toHaveAttribute('aria-pressed', 'false');
      });
      const fetchMock = fetch as unknown as ReturnType<
        typeof vi.fn<typeof fetch>
      >;
      const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];

      expect(url).toBe('/api/feedback');
      expect(init.method).toBe('DELETE');
      expect(JSON.parse(init.body as string)).toStrictEqual({
        responseId: 'resp-9',
      });
      expect(onVote).toHaveBeenCalledWith(null);
    },
  );

  it('still switches from like to dislike', async () => {
    const onVote = vi.fn<(vote: FeedbackType | null) => void>();
    render(
      <AnswerActions
        message={assistant('like')}
        onVote={onVote}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: DISLIKE_LABEL }));

    await waitFor(() => {
      expect(onVote).toHaveBeenCalledWith('dislike');
    });
    const fetchMock = fetch as unknown as ReturnType<
      typeof vi.fn<typeof fetch>
    >;
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];

    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toStrictEqual({
      feedbackType: 'dislike',
      responseId: 'resp-9',
    });
  });

  it('restores the selected feedback when retraction fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn<typeof fetch>()
        .mockResolvedValue(new Response('{}', { status: 500 })),
    );
    const onVote = vi.fn<(vote: FeedbackType | null) => void>();
    render(
      <AnswerActions
        message={assistant('like')}
        onVote={onVote}
      />,
    );
    const like = screen.getByRole('button', { name: LIKE_LABEL });

    fireEvent.click(like);

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledOnce();
    });

    expect(like).toHaveAttribute('aria-pressed', 'true');
    expect(onVote).not.toHaveBeenCalled();
  });

  it('serializes feedback mutations while a request is pending', async () => {
    const request = Promise.withResolvers<Response>();
    const fetchMock = vi.fn<typeof fetch>().mockReturnValue(request.promise);
    vi.stubGlobal('fetch', fetchMock);
    const onVote = vi.fn<(vote: FeedbackType | null) => void>();
    render(
      <AnswerActions
        message={assistant('like')}
        onVote={onVote}
      />,
    );
    const like = screen.getByRole('button', { name: LIKE_LABEL });
    const dislike = screen.getByRole('button', { name: DISLIKE_LABEL });

    fireEvent.click(like);
    fireEvent.click(dislike);

    expect(fetchMock).toHaveBeenCalledOnce();
    expect(like).toBeDisabled();
    expect(dislike).toBeDisabled();

    request.resolve(Response.json({}));
    await waitFor(() => {
      expect(onVote).toHaveBeenCalledWith(null);
    });

    expect(like).toBeEnabled();
    expect(dislike).toBeEnabled();
  });
});
