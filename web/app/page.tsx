'use client';

import { Composer } from '@/components/chat/composer';
import { Thread } from '@/components/chat/thread';
import { Header } from '@/components/shell/header';
import { Sidebar } from '@/components/shell/sidebar';
import { useUiStore } from '@/lib/ui-store';
import { useConversations } from '@/lib/use-conversations';
import { useModels } from '@/lib/use-models';

const ChatScreen = () => {
  const model = useUiStore((s) => s.model);
  const setModel = useUiStore((s) => s.setModel);
  const sidebarOpen = useUiStore((s) => s.sidebarOpen);
  const toggleSidebar = useUiStore((s) => s.toggleSidebar);
  const setSidebarOpen = useUiStore((s) => s.setSidebarOpen);

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
  } = useConversations(model);

  return (
    <div className="flex h-dvh w-full flex-col">
      <Header onToggleSidebar={toggleSidebar} />
      <div className="flex min-h-0 flex-1">
        <Sidebar
          activeId={activeId}
          conversations={conversations}
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
            onRetry={retry}
            renderActions={renderActions}
            status={status}
            streamStartedAt={streamStartedAt}
          />
          <Composer
            model={model}
            models={modelList ?? []}
            modelsError={modelsError}
            modelsLoading={modelsLoading}
            onModelChange={setModel}
            onStop={onStop}
            onSubmit={submitMessage}
            status={status}
          />
        </main>
      </div>
    </div>
  );
};

export default ChatScreen;
