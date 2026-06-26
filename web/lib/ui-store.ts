'use client';

import { create } from 'zustand';
import {
  createJSONStorage,
  persist,
  type StateStorage,
} from 'zustand/middleware';

export const DEFAULT_MODEL = 'claude-sonnet-4-6';

const STORAGE_KEY = 'finkiHub.ui';

export type UiState = {
  activeConversationId: null | string;
  model: string;
  setActiveConversationId: (id: null | string) => void;
  setModel: (model: string) => void;
  setSidebarOpen: (open: boolean) => void;
  sidebarOpen: boolean;
  toggleSidebar: () => void;
};

// Read localStorage through a helper whose return type is widened to include
// `undefined` so SSR (no localStorage) and tests that stub it after import both
// behave correctly, and so the optional chaining below stays meaningful.
const safeLocalStorage = (): Storage | undefined => localStorage;

const lazyLocalStorage: StateStorage = {
  getItem: (name) => safeLocalStorage()?.getItem(name) ?? null,
  removeItem: (name) => {
    safeLocalStorage()?.removeItem(name);
  },
  setItem: (name, value) => {
    safeLocalStorage()?.setItem(name, value);
  },
};

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      activeConversationId: null,
      model: DEFAULT_MODEL,
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
    }),
    {
      name: STORAGE_KEY,
      // Only persist user-meaningful selections; sidebar open state stays
      // per-session so mobile/desktop layouts decide it on load.
      partialize: (state) => ({
        activeConversationId: state.activeConversationId,
        model: state.model,
      }),
      // Hydrate manually after mount (see Providers) to avoid SSR/client
      // markup mismatches from persisted values.
      skipHydration: true,
      storage: createJSONStorage(() => lazyLocalStorage),
    },
  ),
);
