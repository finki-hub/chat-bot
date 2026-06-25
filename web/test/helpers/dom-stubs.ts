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
