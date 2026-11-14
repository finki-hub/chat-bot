'use client';

import { Check, Copy, LoaderCircle, Share2 } from 'lucide-react';
import { useState } from 'react';

import {
  type ConversationShareState,
  type ConversationSharing,
  useConversationSharing,
} from '@/components/chat/use-conversation-sharing';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { IconButton } from '@/components/ui/icon-controls';
import { Input } from '@/components/ui/input';
import { t } from '@/lib/i18n';

type ActiveShareState = Extract<
  ConversationShareState,
  { readonly status: 'revoking' | 'shared' }
>;

const isActive = (state: ConversationShareState): state is ActiveShareState =>
  state.status === 'shared' || state.status === 'revoking';

const stopLabelFor = (state: ActiveShareState): string => {
  if (state.status === 'revoking') return t('header.shareRevoking');
  return state.revokeFailed
    ? t('header.shareRevokeFailed')
    : t('header.shareStop');
};

const ActiveShare = ({
  copyShareUrl,
  revoke,
  state,
}: {
  readonly copyShareUrl: () => Promise<void>;
  readonly revoke: () => Promise<void>;
  readonly state: ActiveShareState;
}) => {
  const copied = state.status === 'shared' && state.copied;
  const copyLabel = copied ? t('header.shareCopied') : t('header.shareCopy');

  return (
    <>
      {state.shareUrl === null ? null : (
        <Input
          aria-label={t('share.linkLabel')}
          className="font-mono text-xs"
          data-testid="share-url"
          onFocus={(event) => {
            event.currentTarget.select();
          }}
          readOnly
          value={state.shareUrl}
        />
      )}
      <DialogFooter>
        <Button
          aria-busy={state.status === 'revoking'}
          disabled={state.status === 'revoking'}
          onClick={() => {
            void revoke();
          }}
          variant="outline"
        >
          {stopLabelFor(state)}
        </Button>
        {state.shareUrl === null ? null : (
          <Button
            aria-live="polite"
            disabled={state.status !== 'shared'}
            onClick={() => {
              void copyShareUrl();
            }}
          >
            {copied ? (
              <Check aria-hidden="true" />
            ) : (
              <Copy aria-hidden="true" />
            )}
            {copyLabel}
          </Button>
        )}
      </DialogFooter>
    </>
  );
};

// Sharing makes the conversation readable by anyone with the link, so it only
// happens after the dialog says so and the user confirms.
export const ShareConversationButton = ({
  conversationId,
  sharing,
}: {
  readonly conversationId: null | string;
  // Passed when several buttons (mobile header, desktop bar) share one state.
  readonly sharing?: ConversationSharing;
}) => {
  const own = useConversationSharing(
    sharing === undefined ? conversationId : null,
  );
  const { copyShareUrl, revoke, share, state } = sharing ?? own;
  const [open, setOpen] = useState(false);
  const active = isActive(state);
  const checking = state.status === 'checking';
  const pending = state.status === 'pending';
  const label = active ? t('header.shareShared') : t('header.share');

  return (
    <Dialog
      onOpenChange={setOpen}
      open={open}
    >
      <IconButton
        aria-busy={checking}
        aria-haspopup="dialog"
        aria-label={label}
        className="relative"
        disabled={conversationId === null || checking}
        onClick={() => {
          setOpen(true);
        }}
        title={label}
      >
        {checking ? (
          <LoaderCircle
            aria-hidden="true"
            className="h-5 w-5 animate-spin"
          />
        ) : (
          <Share2
            aria-hidden="true"
            className="h-5 w-5"
          />
        )}
        {/* A shared conversation is public, so the header says so at a glance. */}
        {active ? (
          <span
            aria-hidden="true"
            className="absolute right-1.5 top-1.5 size-2 rounded-full bg-primary ring-2 ring-background"
            data-testid="share-active"
          />
        ) : null}
      </IconButton>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>
            {active ? t('share.titleShared') : t('header.share')}
          </DialogTitle>
          <DialogDescription>{t('share.description')}</DialogDescription>
        </DialogHeader>
        {active ? (
          <ActiveShare
            copyShareUrl={copyShareUrl}
            revoke={revoke}
            state={state}
          />
        ) : (
          <>
            {state.status === 'failed' ? (
              <p
                className="text-sm text-destructive"
                role="alert"
              >
                {t('header.shareFailed')}
              </p>
            ) : null}
            <DialogFooter>
              <DialogClose asChild>
                <Button variant="outline">{t('common.cancel')}</Button>
              </DialogClose>
              <Button
                aria-busy={pending}
                disabled={pending}
                onClick={() => {
                  void share();
                }}
              >
                {pending ? (
                  <LoaderCircle
                    aria-hidden="true"
                    className="animate-spin"
                  />
                ) : null}
                {pending ? t('header.sharePending') : t('share.create')}
              </Button>
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
};
