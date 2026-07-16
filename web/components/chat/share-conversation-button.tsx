'use client';

import { Check, CircleAlert, LoaderCircle, Share2 } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

import { IconButton } from '@/components/ui/icon-controls';
import { t } from '@/lib/i18n';

type ShareStatus = 'copied' | 'failed' | 'idle' | 'pending';

const labelFor = (status: ShareStatus): string => {
  switch (status) {
    case 'copied':
      return t('header.shareCopied');
    case 'failed':
      return t('header.shareFailed');
    case 'idle':
      return t('header.share');
    case 'pending':
      return t('header.sharePending');
    default: {
      const exhaustiveStatus: never = status;
      return exhaustiveStatus;
    }
  }
};

const iconFor = (status: ShareStatus) => {
  switch (status) {
    case 'copied':
      return (
        <Check
          aria-hidden="true"
          className="h-5 w-5"
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
      return (
        <Share2
          aria-hidden="true"
          className="h-5 w-5"
        />
      );
    case 'pending':
      return (
        <LoaderCircle
          aria-hidden="true"
          className="h-5 w-5 animate-spin"
        />
      );
    default: {
      const exhaustiveStatus: never = status;
      return exhaustiveStatus;
    }
  }
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

export const ShareConversationButton = ({
  conversationId,
}: {
  readonly conversationId: null | string;
}) => {
  const [status, setStatus] = useState<ShareStatus>('idle');
  const requestControllerRef = useRef<AbortController | null>(null);
  const pending = status === 'pending';
  const label = labelFor(status);

  useEffect(() => {
    const controller = requestControllerRef.current;
    requestControllerRef.current = null;
    controller?.abort();
    setStatus('idle');
    return () => {
      const activeController = requestControllerRef.current;
      requestControllerRef.current = null;
      activeController?.abort();
    };
  }, [conversationId]);

  useEffect(() => {
    const timer =
      status === 'copied'
        ? setTimeout(() => {
            setStatus((currentStatus) =>
              currentStatus === 'copied' ? 'idle' : currentStatus,
            );
          }, 1_500)
        : null;
    return () => {
      if (timer !== null) {
        clearTimeout(timer);
      }
    };
  }, [status]);

  const share = async (): Promise<void> => {
    if (conversationId === null || pending) {
      return;
    }
    requestControllerRef.current?.abort();
    const controller = new AbortController();
    requestControllerRef.current = controller;
    setStatus('pending');
    try {
      const response = await fetch(
        `/api/chat/${encodeURIComponent(conversationId)}/share`,
        { method: 'POST', signal: controller.signal },
      );
      if (requestControllerRef.current !== controller) {
        return;
      }
      if (!response.ok) {
        setStatus('failed');
        return;
      }
      const body: unknown = await response.json();
      if (!isRecord(body) || typeof body['shareToken'] !== 'string') {
        setStatus('failed');
        return;
      }
      const url = new URL(
        `/share/${encodeURIComponent(body['shareToken'])}`,
        location.origin,
      ).href;
      await navigator.clipboard.writeText(url);
      if (requestControllerRef.current !== controller) {
        return;
      }
      setStatus('copied');
    } catch (error) {
      if (requestControllerRef.current !== controller) {
        return;
      }
      if (
        error instanceof DOMException ||
        error instanceof SyntaxError ||
        error instanceof TypeError
      ) {
        setStatus('failed');
        return;
      }
      throw error;
    } finally {
      if (requestControllerRef.current === controller) {
        requestControllerRef.current = null;
      }
    }
  };

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
      {iconFor(status)}
    </IconButton>
  );
};
