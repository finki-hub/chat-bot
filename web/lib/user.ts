// Stable anonymous per-browser id, persisted in localStorage. Sent to
// /api/feedback as user_id (the backend requires a non-empty user_id, spec §7).
export const ANON_USER_ID_KEY = 'finkiHub.anonUserId';

export const getAnonUserId = (): string => {
  const existing = localStorage.getItem(ANON_USER_ID_KEY);

  if (existing && existing.length > 0) {
    return existing;
  }

  const id = crypto.randomUUID();

  localStorage.setItem(ANON_USER_ID_KEY, id);

  return id;
};
