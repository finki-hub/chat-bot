// Sent to /api/feedback as user_id, which the backend requires to be non-empty.
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
