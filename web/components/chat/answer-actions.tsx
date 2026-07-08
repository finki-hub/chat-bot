import { Check, Copy, RotateCcw, ThumbsDown, ThumbsUp } from 'lucide-react';
import { posthog } from 'posthog-js';
import { useState } from 'react';

import type { FeedbackType, MyUIMessage } from '@/lib/api-types';

import { t } from '@/lib/i18n';
import { lastText } from '@/lib/message-parts';
import { cn } from '@/lib/utils';

export type AnswerActionsProps = {
  message: MyUIMessage;
  onRegenerate?: () => void;
  onVote?: (vote: FeedbackType) => void;
  pending?: boolean;
  regenerateDisabled?: boolean;
};

const BTN =
  'inline-flex size-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none disabled:pointer-events-none disabled:opacity-40';

export const AnswerActions = ({
  message,
  onRegenerate,
  onVote,
  pending = false,
  regenerateDisabled = false,
}: AnswerActionsProps) => {
  const responseId = message.metadata?.responseId;
  const [copied, setCopied] = useState(false);
  const [vote, setVote] = useState<FeedbackType | null>(
    message.metadata?.feedback ?? null,
  );

  if (!responseId) {
    return null;
  }

  const text = lastText(message) ?? '';

  const copy = async (): Promise<void> => {
    await navigator.clipboard.writeText(text);
    /* eslint-disable camelcase -- PostHog event properties are snake_case. */
    posthog.capture('answer_copied', {
      inference_model: message.metadata?.inferenceModel,
      response_id: responseId,
    });
    /* eslint-enable camelcase -- end of PostHog snake_case properties. */
    setCopied(true);
    setTimeout(() => {
      setCopied(false);
    }, 1_500);
  };

  const sendFeedback = async (feedbackType: FeedbackType): Promise<void> => {
    if (pending) {
      return;
    }
    const previous = vote;
    setVote(feedbackType);
    try {
      const res = await fetch('/api/feedback', {
        body: JSON.stringify({
          feedbackType,
          responseId,
        }),
        headers: { 'content-type': 'application/json' },
        method: 'POST',
      });
      if (!res.ok) {
        setVote(previous);
        return;
      }
      onVote?.(feedbackType);
    } catch {
      setVote(previous);
    }
  };

  return (
    <div
      className="flex items-center gap-1"
      data-testid="answer-actions"
    >
      <button
        aria-label={t('actions.copy')}
        className={BTN}
        onClick={() => {
          void copy();
        }}
        type="button"
      >
        {copied ? (
          <Check
            aria-hidden="true"
            className="size-4"
          />
        ) : (
          <Copy
            aria-hidden="true"
            className="size-4"
          />
        )}
      </button>
      {onRegenerate ? (
        <button
          aria-label={t('actions.regenerate')}
          className={BTN}
          disabled={regenerateDisabled}
          onClick={onRegenerate}
          type="button"
        >
          <RotateCcw
            aria-hidden="true"
            className="size-4"
          />
        </button>
      ) : null}
      <button
        aria-label={t('actions.like')}
        aria-pressed={vote === 'like'}
        className={cn(
          BTN,
          vote === 'like' &&
            'bg-primary text-primary-foreground hover:bg-primary/90 hover:text-primary-foreground',
        )}
        data-testid="like-button"
        disabled={pending}
        onClick={() => {
          void sendFeedback('like');
        }}
        type="button"
      >
        <ThumbsUp
          aria-hidden="true"
          className={cn('size-4', vote === 'like' && 'fill-current')}
        />
      </button>
      <button
        aria-label={t('actions.dislike')}
        aria-pressed={vote === 'dislike'}
        className={cn(
          BTN,
          vote === 'dislike' &&
            'bg-destructive text-destructive-foreground hover:bg-destructive/90 hover:text-destructive-foreground',
        )}
        data-testid="dislike-button"
        disabled={pending}
        onClick={() => {
          void sendFeedback('dislike');
        }}
        type="button"
      >
        <ThumbsDown
          aria-hidden="true"
          className={cn('size-4', vote === 'dislike' && 'fill-current')}
        />
      </button>
    </div>
  );
};
