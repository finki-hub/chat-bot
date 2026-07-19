'use client';

import { useEffect, useRef, useState } from 'react';

export type ConversationShareState =
  | {
      readonly copied: boolean;
      readonly revokeFailed: boolean;
      readonly shareUrl: null | string;
      readonly status: 'shared';
    }
  | { readonly shareUrl: null | string; readonly status: 'revoking' }
  | { readonly status: 'checking' | 'failed' | 'idle' | 'pending' };

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isAbortError = (error: unknown): boolean =>
  error instanceof DOMException && error.name === 'AbortError';

const shareUrlFromResponseBody = (body: unknown): null | string => {
  if (!isRecord(body) || typeof body['shareToken'] !== 'string') return null;
  return new URL(
    `/share/${encodeURIComponent(body['shareToken'])}`,
    location.origin,
  ).href;
};

export const useConversationSharing = (conversationId: null | string) => {
  const [state, setState] = useState<ConversationShareState>({
    status: 'idle',
  });
  const requestControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    requestControllerRef.current?.abort();
    requestControllerRef.current = null;
    let controller: AbortController | null = null;
    if (conversationId === null) {
      setState({ status: 'idle' });
    } else {
      controller = new AbortController();
      requestControllerRef.current = controller;
      setState({ status: 'checking' });
      const loadShareStatus = async (): Promise<void> => {
        try {
          const response = await fetch(
            `/api/chat/${encodeURIComponent(conversationId)}/share`,
            { method: 'GET', signal: controller?.signal },
          );
          if (requestControllerRef.current !== controller) return;
          if (response.status === 200) {
            const shareUrl = shareUrlFromResponseBody(await response.json());
            if (requestControllerRef.current === controller) {
              setState(
                shareUrl === null
                  ? { status: 'failed' }
                  : {
                      copied: false,
                      revokeFailed: false,
                      shareUrl,
                      status: 'shared',
                    },
              );
            }
            return;
          }
          setState({ status: response.status === 204 ? 'idle' : 'failed' });
        } catch (error) {
          if (
            requestControllerRef.current === controller &&
            !isAbortError(error)
          ) {
            setState({ status: 'failed' });
          }
        } finally {
          if (requestControllerRef.current === controller) {
            requestControllerRef.current = null;
          }
        }
      };
      void loadShareStatus();
    }

    return () => {
      if (controller === null) return;
      if (requestControllerRef.current === controller) {
        requestControllerRef.current = null;
      }
      controller.abort();
    };
  }, [conversationId]);

  const copied = state.status === 'shared' && state.copied;
  useEffect(() => {
    const timer = copied
      ? setTimeout(() => {
          setState((current) =>
            current.status === 'shared'
              ? { ...current, copied: false }
              : current,
          );
        }, 1_500)
      : null;
    return () => {
      if (timer !== null) clearTimeout(timer);
    };
  }, [copied]);

  const share = async (): Promise<void> => {
    if (
      conversationId === null ||
      (state.status !== 'idle' && state.status !== 'failed')
    ) {
      return;
    }
    requestControllerRef.current?.abort();
    const controller = new AbortController();
    requestControllerRef.current = controller;
    setState({ status: 'pending' });
    try {
      const response = await fetch(
        `/api/chat/${encodeURIComponent(conversationId)}/share`,
        { method: 'POST', signal: controller.signal },
      );
      if (requestControllerRef.current !== controller) return;
      if (!response.ok) {
        setState({ status: 'failed' });
        return;
      }
      const body: unknown = await response.json();
      const shareUrl = shareUrlFromResponseBody(body);
      if (shareUrl === null) {
        setState({ status: 'failed' });
        return;
      }
      let copiedShare = false;
      try {
        await navigator.clipboard.writeText(shareUrl);
        copiedShare = true;
      } catch {
        copiedShare = false;
      }
      if (requestControllerRef.current !== controller) return;
      setState({
        copied: copiedShare,
        revokeFailed: false,
        shareUrl,
        status: 'shared',
      });
    } catch (error) {
      if (requestControllerRef.current === controller && !isAbortError(error)) {
        setState({ status: 'failed' });
      }
    } finally {
      if (requestControllerRef.current === controller) {
        requestControllerRef.current = null;
      }
    }
  };

  const copyShareUrl = async (): Promise<void> => {
    if (state.status !== 'shared' || state.shareUrl === null) return;
    const { shareUrl } = state;
    try {
      await navigator.clipboard.writeText(shareUrl);
      setState((current) =>
        current.status === 'shared' && current.shareUrl === shareUrl
          ? { ...current, copied: true }
          : current,
      );
    } catch {
      setState((current) =>
        current.status === 'shared' && current.shareUrl === shareUrl
          ? { ...current, copied: false }
          : current,
      );
    }
  };

  const revoke = async (): Promise<void> => {
    if (conversationId === null || state.status !== 'shared') return;
    const { shareUrl } = state;
    requestControllerRef.current?.abort();
    const controller = new AbortController();
    requestControllerRef.current = controller;
    setState({ shareUrl, status: 'revoking' });
    try {
      const response = await fetch(
        `/api/chat/${encodeURIComponent(conversationId)}/share`,
        { method: 'DELETE', signal: controller.signal },
      );
      if (requestControllerRef.current !== controller) return;
      setState(
        response.ok
          ? { status: 'idle' }
          : {
              copied: false,
              revokeFailed: true,
              shareUrl,
              status: 'shared',
            },
      );
    } catch (error) {
      if (requestControllerRef.current === controller && !isAbortError(error)) {
        setState({
          copied: false,
          revokeFailed: true,
          shareUrl,
          status: 'shared',
        });
      }
    } finally {
      if (requestControllerRef.current === controller) {
        requestControllerRef.current = null;
      }
    }
  };

  return { copyShareUrl, revoke, share, state };
};
