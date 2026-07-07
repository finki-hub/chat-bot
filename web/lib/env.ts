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

export const API_BASE_URL = required('API_BASE_URL');

export const CHAT_API_KEY = required('CHAT_API_KEY');

export const RESUMABLE_STREAM_REDIS_URL = required(
  'RESUMABLE_STREAM_REDIS_URL',
);

export const env = {
  API_BASE_URL,
  CHAT_API_KEY,
  RESUMABLE_STREAM_REDIS_URL,
} as const;
