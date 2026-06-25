import { Check, Copy, RotateCcw, ThumbsDown, ThumbsUp } from 'lucide-react';
import { useState } from 'react';

import type { FeedbackType, MyUIMessage } from '@/lib/api-types';

import { t } from '@/lib/i18n';
import { getAnonUserId } from '@/lib/user';

export type AnswerActionsProps = {
  message: MyUIMessage;
  onRegenerate?: () => void;
  questionText?: string;
};

const answerText = (message: MyUIMessage): string => {
  const texts = message.parts.filter(
    (p): p is { text: string; type: 'text' } => p.type === 'text',
  );
  return texts.at(-1)?.text ?? '';
};

const BTN =
  'inline-flex items-center justify-center rounded-md p-1.5 text-muted-foreground hover:bg-muted';

export const AnswerActions = ({
  message,
  onRegenerate,
  questionText,
}: AnswerActionsProps) => {
  const responseId = message.metadata?.responseId;
  const [copied, setCopied] = useState(false);
  const [vote, setVote] = useState<FeedbackType | null>(null);

  if (!responseId) {
    return null;
  }

  const text = answerText(message);

  const copy = async (): Promise<void> => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => {
      setCopied(false);
    }, 1_500);
  };

  const sendFeedback = async (feedbackType: FeedbackType): Promise<void> => {
    const previous = vote;
    setVote(feedbackType);
    try {
      const res = await fetch('/api/feedback', {
        body: JSON.stringify({
          answerText: text,
          feedbackType,
          inferenceModel: message.metadata?.inferenceModel,
          questionText,
          responseId,
          userId: getAnonUserId(),
        }),
        headers: { 'content-type': 'application/json' },
        method: 'POST',
      });
      if (!res.ok) {
        setVote(previous);
      }
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
        className={`${BTN} ${vote === 'like' ? 'text-green-600' : ''}`}
        data-testid="like-button"
        onClick={() => {
          void sendFeedback('like');
        }}
        type="button"
      >
        <ThumbsUp
          aria-hidden="true"
          className="size-4"
        />
      </button>
      <button
        aria-label={t('actions.dislike')}
        aria-pressed={vote === 'dislike'}
        className={`${BTN} ${vote === 'dislike' ? 'text-red-600' : ''}`}
        onClick={() => {
          void sendFeedback('dislike');
        }}
        type="button"
      >
        <ThumbsDown
          aria-hidden="true"
          className="size-4"
        />
      </button>
    </div>
  );
};
