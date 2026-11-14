import { Check, Copy, RotateCcw, ThumbsDown, ThumbsUp } from 'lucide-react';
import { posthog } from 'posthog-js';
import { useRef, useState } from 'react';

import type {
  FeedbackSelection,
  FeedbackType,
  MyUIMessage,
} from '@/lib/api-types';

import {
  type FeedbackDetails,
  FeedbackReasonDialog,
} from '@/components/chat/feedback-reason-dialog';
import { ControlTooltip } from '@/components/ui/icon-controls';
import { useCitations } from '@/lib/citation-context';
import { renumberText } from '@/lib/citations';
import { t } from '@/lib/i18n';
import { answerText } from '@/lib/message-parts';
import { withSourceList } from '@/lib/source-groups';
import { cn } from '@/lib/utils';

export type AnswerActionsProps = {
  message: MyUIMessage;
  onRegenerate?: () => void;
  onVote?: (vote: FeedbackSelection) => void;
  pending?: boolean;
  regenerateDisabled?: boolean;
};

const BTN =
  'inline-flex size-11 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none disabled:pointer-events-none disabled:opacity-40 sm:pointer-fine:size-8';

// Re-sends the dislike with its details; the upsert keeps one row per answer.
const sendDislikeDetails = async (
  responseId: string,
  details: FeedbackDetails,
): Promise<boolean> => {
  try {
    const res = await fetch('/api/feedback', {
      body: JSON.stringify({ feedbackType: 'dislike', responseId, ...details }),
      headers: { 'content-type': 'application/json' },
      method: 'POST',
    });
    return res.ok;
  } catch {
    return false;
  }
};

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
  const [submittingFeedback, setSubmittingFeedback] = useState(false);
  const [reasonOpen, setReasonOpen] = useState(false);
  const feedbackPendingRef = useRef(false);
  // Inside an answer, copies cite the numbers its chips show.
  const citations = useCitations();

  if (!responseId) {
    return null;
  }

  const text = withSourceList(
    citations === null
      ? (answerText(message) ?? '')
      : renumberText(answerText(message) ?? '', citations.numbering),
    citations?.sources ?? message.metadata?.sources ?? [],
    t('sources.title'),
  );
  const votingDisabled = pending || submittingFeedback;

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
    const pendingRequest = feedbackPendingRef;
    if (pending || pendingRequest.current) {
      return;
    }
    pendingRequest.current = true;
    setSubmittingFeedback(true);
    const previous = vote;
    const nextVote: FeedbackSelection =
      vote === feedbackType ? null : feedbackType;
    setVote(nextVote);
    try {
      const res = await fetch('/api/feedback', {
        body: JSON.stringify(
          nextVote === null
            ? { responseId }
            : { feedbackType: nextVote, responseId },
        ),
        headers: { 'content-type': 'application/json' },
        method: nextVote === null ? 'DELETE' : 'POST',
      });
      if (!res.ok) {
        setVote(previous);
        return;
      }
      onVote?.(nextVote);
      if (nextVote === 'dislike') {
        setReasonOpen(true);
      }
    } catch {
      setVote(previous);
    } finally {
      pendingRequest.current = false;
      setSubmittingFeedback(false);
    }
  };

  return (
    <div
      className="flex items-center gap-1"
      data-testid="answer-actions"
    >
      <FeedbackReasonDialog
        onOpenChange={setReasonOpen}
        onSubmit={async (details) => sendDislikeDetails(responseId, details)}
        open={reasonOpen}
      />
      <ControlTooltip label={t('actions.copy')}>
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
      </ControlTooltip>
      {onRegenerate ? (
        <ControlTooltip
          disabled={regenerateDisabled}
          label={t('actions.regenerate')}
        >
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
        </ControlTooltip>
      ) : null}
      <ControlTooltip
        disabled={pending}
        label={t('actions.like')}
      >
        <button
          aria-busy={submittingFeedback || undefined}
          aria-label={t('actions.like')}
          aria-pressed={vote === 'like'}
          className={cn(
            BTN,
            vote === 'like' &&
              'bg-primary text-primary-foreground hover:bg-primary/90 hover:text-primary-foreground',
          )}
          data-testid="like-button"
          disabled={votingDisabled}
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
      </ControlTooltip>
      <ControlTooltip
        disabled={pending}
        label={t('actions.dislike')}
      >
        <button
          aria-busy={submittingFeedback || undefined}
          aria-label={t('actions.dislike')}
          aria-pressed={vote === 'dislike'}
          className={cn(
            BTN,
            vote === 'dislike' &&
              'bg-destructive text-destructive-foreground hover:bg-destructive/90 hover:text-destructive-foreground',
          )}
          data-testid="dislike-button"
          disabled={votingDisabled}
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
      </ControlTooltip>
    </div>
  );
};
