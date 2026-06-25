// Server-only access to BFF env vars. NEVER import this from a Client Component
// (no 'use client'); it must only run in Route Handlers / server code.
// There are no NEXT_PUBLIC_* vars in v1 — the browser only calls same-origin /api/*.
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

/** Base URL of the Python chat API (protocol-v2), e.g. http://api:8880. No /API prefix; /chat/ has a trailing slash. */
export const API_BASE_URL = required('API_BASE_URL');

/** Master x-api-key for POST /chat/feedback. Server-only — injected by the BFF, never sent to the browser. */
export const CHAT_API_KEY = required('CHAT_API_KEY');

export const env = { API_BASE_URL, CHAT_API_KEY } as const;
