'use client';

import { create } from 'zustand';

export type UiState = {
  activeConversationId: null | string;
  model: string;
  setActiveConversationId: (id: null | string) => void;
  setModel: (model: string) => void;
  setSidebarOpen: (open: boolean) => void;
  sidebarOpen: boolean;
  toggleSidebar: () => void;
};

export const useUiStore = create<UiState>((set) => ({
  activeConversationId: null,
  model: 'claude-sonnet-4-6',
  setActiveConversationId: (id) => {
    set({ activeConversationId: id });
  },
  setModel: (model) => {
    set({ model });
  },
  setSidebarOpen: (open) => {
    set({ sidebarOpen: open });
  },
  sidebarOpen: true,
  toggleSidebar: () => {
    set((s) => ({ sidebarOpen: !s.sidebarOpen }));
  },
}));
