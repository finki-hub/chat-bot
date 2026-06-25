// NEVER import this from a Client Component; it must only run in Route Handlers /
// server code.
import 'server-only';

const required = (name: string): string => {
  const value = process.env[name];

  if (!value || value.length === 0) {
    throw new Error(
      `Missing required server env var ${name}. Set it in the BFF environment ` +
        `(e.g. .env.local or the compose service env); it must never be exposed to the browser.`,
    );
  }

  return value;
};

// No /API prefix; /chat/ has a trailing slash.
export const API_BASE_URL = required('API_BASE_URL');

export const CHAT_API_KEY = required('CHAT_API_KEY');

export const env = { API_BASE_URL, CHAT_API_KEY } as const;
