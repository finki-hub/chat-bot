'use client';

import { create } from 'zustand';
import {
  createJSONStorage,
  persist,
  type StateStorage,
} from 'zustand/middleware';

export const DEFAULT_MODEL = 'claude-sonnet-5-5';

const STORAGE_KEY = 'finkiHub.ui';
const STORAGE_VERSION = 1;

type PersistedUiState = Pick<
  UiState,
  'activeConversationId' | 'model' | 'reasoning'
>;

const LEGACY_MODEL_MIGRATIONS = new Map([
  ['claude-haiku-4-5', 'claude-haiku-5-5'],
  ['claude-opus-4-8', 'claude-opus-5-5'],
  ['claude-sonnet-5', 'claude-sonnet-5-5'],
  ['gemini-3.1-flash-lite', 'gemini-3.5-flash-lite'],
  ['gemini-3.5-flash', 'gemini-3.8-flash'],
  ['gpt-5.4', 'gpt-5.6-terra'],
  ['gpt-5.5', 'gpt-6.1-sol'],
  ['gpt-5.6-luna', 'gpt-6-luna'],
  ['gpt-5.6-sol', 'gpt-6.1-sol'],
  [
    'openrouter:deepseek/deepseek-v4-flash',
    'openrouter:deepseek/deepseek-v4.1-flash',
  ],
  [
    'openrouter:deepseek/deepseek-v4-flash-0731',
    'openrouter:deepseek/deepseek-v4.1-flash',
  ],
  ['openrouter:qwen/qwen3.8-max', 'openrouter:qwen/qwen3.8-max-0902'],
  ['openrouter:x-ai/grok-4.6', 'openrouter:x-ai/grok-4.7'],
]);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const persistedUiState = (value: unknown): Partial<PersistedUiState> => {
  if (!isRecord(value)) return {};
  const activeConversationId = value['activeConversationId'];
  const model = value['model'];
  const reasoning = value['reasoning'];
  const state: Partial<PersistedUiState> = {};
  if (
    activeConversationId === null ||
    typeof activeConversationId === 'string'
  ) {
    state.activeConversationId = activeConversationId;
  }
  if (typeof model === 'string') state.model = model;
  if (typeof reasoning === 'boolean') state.reasoning = reasoning;
  return state;
};

const migratePersistedUiState = (
  persisted: unknown,
  version: number,
): Partial<PersistedUiState> => {
  const state = persistedUiState(persisted);
  if (version !== 0 || typeof state.model !== 'string') return state;
  return {
    ...state,
    model: LEGACY_MODEL_MIGRATIONS.get(state.model) ?? state.model,
  };
};

export type UiState = {
  activeConversationId: null | string;
  model: string;
  reasoning: boolean;
  resetModel: () => void;
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
      resetModel: () => {
        set({ model: DEFAULT_MODEL });
      },
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
      sidebarOpen: false,
      toggleSidebar: () => {
        set((s) => ({ sidebarOpen: !s.sidebarOpen }));
      },
    }),
    {
      merge: (persisted, current) => {
        const persistedState = persistedUiState(persisted);
        return {
          ...current,
          ...persistedState,
        };
      },
      migrate: migratePersistedUiState,
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
      version: STORAGE_VERSION,
    },
  ),
);
