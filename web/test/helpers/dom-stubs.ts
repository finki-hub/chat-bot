// jsdom lacks ResizeObserver, which the vendored Conversation (use-stick-to-bottom)
// constructs with `new`. Provide a constructable no-op stub.
export class ResizeObserverStub {
  callbacks: ResizeObserverCallback[] = [];

  constructor(callback: ResizeObserverCallback) {
    this.callbacks.push(callback);
  }

  disconnect(): void {
    this.callbacks = [];
  }

  observe(): void {
    this.callbacks.at(0);
  }

  unobserve(): void {
    this.callbacks.at(0);
  }
}

// jsdom 26 only exposes localStorage for a concrete origin; under the default
// vitest jsdom URL the origin is opaque, so we install a conforming in-memory
// store the transport's getAnonUserId can use.
export const createMemoryStorage = (): Storage => {
  const store = new Map<string, string>();

  return {
    clear: () => {
      store.clear();
    },
    getItem: (key) => store.get(key) ?? null,
    key: (index) => store.keys().toArray()[index] ?? null,
    get length() {
      return store.size;
    },
    removeItem: (key) => {
      store.delete(key);
    },
    setItem: (key, value) => {
      store.set(key, value);
    },
  };
};
