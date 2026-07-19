'use client';

import {
  Check,
  CircleAlert,
  Copy,
  LoaderCircle,
  Share2,
  Unlink,
} from 'lucide-react';

import {
  type ConversationShareState,
  useConversationSharing,
} from '@/components/chat/use-conversation-sharing';
import { IconButton } from '@/components/ui/icon-controls';
import { t } from '@/lib/i18n';

type ActiveShareState = Extract<
  ConversationShareState,
  { readonly status: 'revoking' | 'shared' }
>;

const shareLabelFor = (state: ConversationShareState): string => {
  switch (state.status) {
    case 'checking':
      return t('header.shareChecking');
    case 'failed':
      return t('header.shareFailed');
    case 'idle':
      return t('header.share');
    case 'pending':
      return t('header.sharePending');
    case 'revoking':
    case 'shared':
      return t('header.share');
    default: {
      const exhaustiveState: never = state;
      return exhaustiveState;
    }
  }
};

const shareIconFor = (state: ConversationShareState) => {
  switch (state.status) {
    case 'checking':
    case 'pending':
      return (
        <LoaderCircle
          aria-hidden="true"
          className="h-5 w-5 animate-spin"
        />
      );
    case 'failed':
      return (
        <CircleAlert
          aria-hidden="true"
          className="h-5 w-5"
        />
      );
    case 'idle':
    case 'revoking':
    case 'shared':
      return (
        <Share2
          aria-hidden="true"
          className="h-5 w-5"
        />
      );
    default: {
      const exhaustiveState: never = state;
      return exhaustiveState;
    }
  }
};

const stopLabelFor = (state: ActiveShareState): string => {
  if (state.status === 'revoking') return t('header.shareRevoking');
  return state.revokeFailed
    ? t('header.shareRevokeFailed')
    : t('header.shareStop');
};

const StopIcon = ({ state }: { readonly state: ActiveShareState }) => {
  if (state.status === 'revoking') {
    return (
      <LoaderCircle
        aria-hidden="true"
        className="h-5 w-5 animate-spin"
      />
    );
  }
  if (state.revokeFailed) {
    return (
      <CircleAlert
        aria-hidden="true"
        className="h-5 w-5"
      />
    );
  }
  return (
    <Unlink
      aria-hidden="true"
      className="h-5 w-5"
    />
  );
};

const ActiveShareControls = ({
  copyShareUrl,
  revoke,
  state,
}: {
  readonly copyShareUrl: () => Promise<void>;
  readonly revoke: () => Promise<void>;
  readonly state: ActiveShareState;
}) => {
  const revoking = state.status === 'revoking';
  const stopLabel = stopLabelFor(state);
  const showCopy = state.status === 'shared' && state.shareUrl !== null;
  const copyLabel =
    state.status === 'shared' && state.copied
      ? t('header.shareCopied')
      : t('header.shareCopy');

  return (
    <div
      aria-live="polite"
      className="flex items-center gap-2"
    >
      {showCopy ? (
        <IconButton
          aria-label={copyLabel}
          onClick={() => {
            void copyShareUrl();
          }}
          title={copyLabel}
        >
          {state.copied ? (
            <Check
              aria-hidden="true"
              className="h-5 w-5"
            />
          ) : (
            <Copy
              aria-hidden="true"
              className="h-5 w-5"
            />
          )}
        </IconButton>
      ) : null}
      <IconButton
        aria-busy={revoking}
        aria-label={stopLabel}
        disabled={revoking}
        onClick={() => {
          void revoke();
        }}
        title={stopLabel}
      >
        <StopIcon state={state} />
      </IconButton>
    </div>
  );
};

export const ShareConversationButton = ({
  conversationId,
}: {
  readonly conversationId: null | string;
}) => {
  const { copyShareUrl, revoke, share, state } =
    useConversationSharing(conversationId);

  if (state.status === 'shared' || state.status === 'revoking') {
    return (
      <ActiveShareControls
        copyShareUrl={copyShareUrl}
        revoke={revoke}
        state={state}
      />
    );
  }

  const pending = state.status === 'checking' || state.status === 'pending';
  const label = shareLabelFor(state);
  return (
    <IconButton
      aria-busy={pending}
      aria-label={label}
      disabled={conversationId === null || pending}
      onClick={() => {
        void share();
      }}
      title={label}
    >
      {shareIconFor(state)}
    </IconButton>
  );
};
