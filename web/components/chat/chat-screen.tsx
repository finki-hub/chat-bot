'use client';

import { useEffect, useLayoutEffect, useMemo, useState } from 'react';

import { Composer } from '@/components/chat/composer';
import { ServiceBanner } from '@/components/chat/service-banner';
import { Thread } from '@/components/chat/thread';
import { CredentialSettingsDialog } from '@/components/shell/credential-settings-dialog';
import { Header } from '@/components/shell/header';
import { Sidebar } from '@/components/shell/sidebar';
import { recoverSelectedModel } from '@/lib/model-catalog';
import { isReasoningCapableModel } from '@/lib/reasoning';
import { DEFAULT_MODEL, useUiStore } from '@/lib/ui-store';
import { useConversations } from '@/lib/use-conversations';
import { useCredentials } from '@/lib/use-credentials';
import { useHealth } from '@/lib/use-health';
import { useModels } from '@/lib/use-models';

const DESKTOP_SIDEBAR_QUERY = '(min-width: 768px)';
const useIsomorphicLayoutEffect =
  typeof window === 'undefined' ? useEffect : useLayoutEffect;

const noop = () => {};

const subscribeToMediaQuery = (
  media: MediaQueryList,
  onChange: (event: MediaQueryListEvent) => void,
) => {
  media.addEventListener('change', onChange);

  return () => {
    media.removeEventListener('change', onChange);
  };
};

export const ChatScreen = () => {
  const model = useUiStore((s) => s.model);
  const setModel = useUiStore((s) => s.setModel);
  const reasoning = useUiStore((s) => s.reasoning);
  const setReasoning = useUiStore((s) => s.setReasoning);
  const sidebarOpen = useUiStore((s) => s.sidebarOpen);
  const reasoningActive = reasoning && isReasoningCapableModel(model);
  const toggleSidebar = useUiStore((s) => s.toggleSidebar);
  const setSidebarOpen = useUiStore((s) => s.setSidebarOpen);
  const [sidebarSynced, setSidebarSynced] = useState(false);
  const [desktopSidebar, setDesktopSidebar] = useState(false);
  const [credentialSettingsOpen, setCredentialSettingsOpen] = useState(false);

  const { unavailable } = useHealth();

  useIsomorphicLayoutEffect(() => {
    if (typeof matchMedia !== 'function') {
      return noop;
    }

    const media = matchMedia(DESKTOP_SIDEBAR_QUERY);
    const syncSidebarWithViewport = (event: MediaQueryListEvent) => {
      setDesktopSidebar(event.matches);
      setSidebarOpen(event.matches);
    };

    setDesktopSidebar(media.matches);
    setSidebarOpen(media.matches);
    setSidebarSynced(true);

    return subscribeToMediaQuery(media, syncSidebarWithViewport);
  }, [setSidebarOpen]);

  const {
    isError: modelsError,
    isLoading: modelsLoading,
    models: modelList,
  } = useModels();
  const {
    credentials,
    isError: credentialsError,
    isLoading: credentialsLoading,
  } = useCredentials();
  const availableProviders = useMemo(
    () =>
      new Set<string>(
        credentials
          .filter((credential) => credential.has_api_key)
          .map((credential) => credential.provider),
      ),
    [credentials],
  );
  const availableModels = useMemo(
    () => modelList.filter((entry) => availableProviders.has(entry.provider)),
    [availableProviders, modelList],
  );

  useEffect(() => {
    if (
      modelsLoading ||
      credentialsLoading ||
      modelsError ||
      availableModels.length === 0
    ) {
      return;
    }
    const recovered = recoverSelectedModel(
      availableModels,
      model,
      DEFAULT_MODEL,
    );
    if (recovered !== model) {
      setModel(recovered);
    }
  }, [
    availableModels,
    credentialsLoading,
    modelsLoading,
    modelsError,
    model,
    setModel,
  ]);
  const {
    activeError,
    activeId,
    activeStatus,
    conversationListError,
    conversationListLoading,
    conversations,
    generatingTitleId,
    messages,
    onClearAll,
    onDelete,
    onGenerateTitle,
    onNewChat,
    onRename,
    onRetryConversationList,
    onSelect,
    onStop,
    renderActions,
    retry,
    status,
    submitMessage,
  } = useConversations(model, unavailable, reasoningActive);

  return (
    <div className="flex h-dvh w-full flex-col">
      <Header
        onOpenCredentials={() => {
          setCredentialSettingsOpen(true);
        }}
        onToggleSidebar={toggleSidebar}
      />
      <CredentialSettingsDialog
        onOpenChange={setCredentialSettingsOpen}
        open={credentialSettingsOpen}
      />
      {unavailable ? <ServiceBanner /> : null}
      <div className="flex min-h-0 flex-1">
        <Sidebar
          activeId={activeId}
          conversations={conversations}
          generatingTitleId={generatingTitleId}
          listError={conversationListError}
          listLoading={conversationListLoading}
          mobile={sidebarSynced && !desktopSidebar}
          onClearAll={onClearAll}
          onClose={() => {
            setSidebarOpen(false);
          }}
          onDelete={onDelete}
          onGenerateTitle={onGenerateTitle}
          onNewChat={onNewChat}
          onRename={onRename}
          onRetryList={onRetryConversationList}
          onSelect={onSelect}
          open={sidebarOpen}
          synced={sidebarSynced}
        />
        <main className="flex min-w-0 flex-1 flex-col">
          <Thread
            activeError={activeError}
            activeStatus={activeStatus}
            messages={messages}
            onPickSuggestion={unavailable ? undefined : submitMessage}
            onRetry={unavailable ? undefined : retry}
            renderActions={renderActions}
            status={status}
          />
          <Composer
            availableProviders={availableProviders}
            credentialsError={credentialsError}
            credentialsLoading={credentialsLoading}
            disabled={unavailable}
            model={model}
            models={modelList}
            modelsError={modelsError}
            modelsLoading={modelsLoading}
            onModelChange={setModel}
            onReasoningChange={setReasoning}
            onStop={onStop}
            onSubmit={submitMessage}
            reasoning={reasoningActive}
            status={status}
          />
        </main>
      </div>
    </div>
  );
};
