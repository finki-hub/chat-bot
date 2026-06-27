'use client';

import { Composer } from '@/components/chat/composer';
import { ServiceBanner } from '@/components/chat/service-banner';
import { Thread } from '@/components/chat/thread';
import { Header } from '@/components/shell/header';
import { Sidebar } from '@/components/shell/sidebar';
import { isReasoningCapableModel } from '@/lib/reasoning';
import { useUiStore } from '@/lib/ui-store';
import { useConversations } from '@/lib/use-conversations';
import { useHealth } from '@/lib/use-health';
import { useModels } from '@/lib/use-models';

const ChatScreen = () => {
  const model = useUiStore((s) => s.model);
  const setModel = useUiStore((s) => s.setModel);
  const reasoning = useUiStore((s) => s.reasoning);
  const setReasoning = useUiStore((s) => s.setReasoning);
  const sidebarOpen = useUiStore((s) => s.sidebarOpen);
  // Honor the reasoning toggle only for capable models. The persisted flag is kept
  // (so it re-applies when switching back to a capable model), but it is never sent to
  // the backend nor shown as "on" for a model that would ignore/reject it.
  const reasoningActive = reasoning && isReasoningCapableModel(model);
  const toggleSidebar = useUiStore((s) => s.toggleSidebar);
  const setSidebarOpen = useUiStore((s) => s.setSidebarOpen);

  const { unavailable } = useHealth();
  const {
    data: modelList,
    isError: modelsError,
    isLoading: modelsLoading,
  } = useModels();
  const {
    activeError,
    activeId,
    activeStatus,
    conversations,
    messages,
    onClearAll,
    onDelete,
    onNewChat,
    onRename,
    onSelect,
    onStop,
    renderActions,
    retry,
    status,
    streamStartedAt,
    submitMessage,
  } = useConversations(model, unavailable, reasoningActive);

  return (
    <div className="flex h-dvh w-full flex-col">
      <Header onToggleSidebar={toggleSidebar} />
      {unavailable ? <ServiceBanner /> : null}
      <div className="flex min-h-0 flex-1">
        <Sidebar
          activeId={activeId}
          conversations={conversations}
          onClearAll={onClearAll}
          onClose={() => {
            setSidebarOpen(false);
          }}
          onDelete={onDelete}
          onNewChat={onNewChat}
          onRename={onRename}
          onSelect={onSelect}
          open={sidebarOpen}
        />
        <main className="flex min-w-0 flex-1 flex-col">
          <Thread
            activeError={activeError}
            activeStatus={activeStatus}
            messages={messages}
            onPickSuggestion={submitMessage}
            onRetry={unavailable ? undefined : retry}
            renderActions={renderActions}
            status={status}
            streamStartedAt={streamStartedAt}
          />
          <Composer
            disabled={unavailable}
            model={model}
            models={modelList ?? []}
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

export default ChatScreen;
