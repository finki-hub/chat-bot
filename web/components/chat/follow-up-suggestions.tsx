import { CornerDownRight } from 'lucide-react';
import { posthog } from 'posthog-js';

import type { MyUIMessage } from '@/lib/api-types';
import type { FollowUps } from '@/lib/use-follow-up-suggestions';

import { t } from '@/lib/i18n';

const CHIP =
  'inline-flex min-h-11 items-center gap-2 rounded-xl border border-border bg-card/60 px-3 py-2 text-left text-pretty text-sm text-foreground/90 transition-colors hover:border-primary/40 hover:bg-card focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring sm:pointer-fine:min-h-9';

const SuggestionList = ({
  onPick,
  responseId,
  suggestions,
}: {
  onPick: (text: string) => unknown;
  responseId?: string;
  suggestions: readonly string[];
}) => (
  <nav
    aria-label={t('thread.followUps')}
    className="mt-3 flex flex-col items-start gap-2 motion-safe:animate-in motion-safe:fade-in-0 motion-safe:duration-300"
  >
    {suggestions.map((suggestion, index) => (
      <button
        className={CHIP}
        key={suggestion}
        onClick={() => {
          /* eslint-disable camelcase -- PostHog event properties are snake_case. */
          posthog.capture('follow_up_clicked', {
            position: index,
            response_id: responseId,
            suggestion_count: suggestions.length,
          });
          /* eslint-enable camelcase -- end of PostHog snake_case properties. */
          onPick(suggestion);
        }}
        type="button"
      >
        <CornerDownRight
          aria-hidden="true"
          className="size-4 shrink-0 text-muted-foreground"
        />
        <span>{suggestion}</span>
      </button>
    ))}
  </nav>
);

// Shown only under the settled last answer the suggestions were made for.
export const FollowUpSuggestions = ({
  followUps,
  message,
  onPick,
  settledLastMessageId,
}: {
  followUps: FollowUps | null | undefined;
  message: MyUIMessage;
  onPick: ((text: string) => unknown) | undefined;
  settledLastMessageId: string | undefined;
}) =>
  onPick === undefined ||
  message.id !== settledLastMessageId ||
  followUps?.messageId !== message.id ? null : (
    <SuggestionList
      onPick={onPick}
      responseId={message.metadata?.responseId}
      suggestions={followUps.suggestions}
    />
  );
