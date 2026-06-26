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
  reasoning: boolean;
  setActiveConversationId: (id: null | string) => void;
  setModel: (model: string) => void;
  setReasoning: (reasoning: boolean) => void;
  setSidebarOpen: (open: boolean) => void;
  sidebarOpen: boolean;
  toggleSidebar: () => void;
};

// Read localStorage through a `typeof` guard so SSR (no `localStorage` global)
// and tests that stub it after import both behave correctly; the widened
// `| undefined` return keeps the optional chaining below meaningful.
const safeLocalStorage = (): Storage | undefined =>
  typeof localStorage === 'undefined' ? undefined : localStorage;

const lazyLocalStorage: StateStorage = {
  getItem: (name) => {
    try {
      return safeLocalStorage()?.getItem(name) ?? null;
    } catch {
      return null;
    }
  },
  removeItem: (name) => {
    try {
      safeLocalStorage()?.removeItem(name);
    } catch {
      // Persistence is best-effort; ignore storage failures.
    }
  },
  setItem: (name, value) => {
    try {
      safeLocalStorage()?.setItem(name, value);
    } catch {
      // Persistence is best-effort; ignore storage failures (e.g. quota).
    }
  },
};

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      activeConversationId: null,
      model: DEFAULT_MODEL,
      reasoning: false,
      setActiveConversationId: (id) => {
        set({ activeConversationId: id });
      },
      setModel: (model) => {
        set({ model });
      },
      setReasoning: (reasoning) => {
        set({ reasoning });
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
        reasoning: state.reasoning,
      }),
      // Hydrate manually after mount (see Providers) to avoid SSR/client
      // markup mismatches from persisted values.
      skipHydration: true,
      storage: createJSONStorage(() => lazyLocalStorage),
    },
  ),
);
