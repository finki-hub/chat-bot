'use client';

type MutationListener = () => void;

const pendingSessionKeys = new Set<string>();
const listeners = new Set<MutationListener>();

const notifyListeners = () => {
  for (const listener of listeners) {
    listener();
  }
};

export const beginCredentialMutation = (sessionKey: string): boolean => {
  if (pendingSessionKeys.has(sessionKey)) {
    return false;
  }
  pendingSessionKeys.add(sessionKey);
  notifyListeners();
  return true;
};

export const finishCredentialMutation = (sessionKey: string): void => {
  if (pendingSessionKeys.delete(sessionKey)) {
    notifyListeners();
  }
};

export const hasPendingCredentialMutation = (
  sessionKey: null | string,
): boolean => sessionKey !== null && pendingSessionKeys.has(sessionKey);

export const subscribeCredentialMutations = (
  listener: MutationListener,
): (() => void) => {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
};
