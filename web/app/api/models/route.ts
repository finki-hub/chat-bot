// BFF: proxy GET {API_BASE_URL}/chat/models -> a flat string[] of model ids.
// Server-only (Route Handler): API_BASE_URL never reaches the browser.
// The upstream list changes rarely, so we cache briefly. Any upstream failure
// degrades to [] (the model picker falls back to its defaults) and is marked
// with X-Models-Source: error so callers/tests can detect the fallback path.
import type { ModelId } from '@/lib/api-types';

import { API_BASE_URL } from '@/lib/env';

// Always run on the server; never statically prerender (needs server env).
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const CACHE_CONTROL = 'public, max-age=300, stale-while-revalidate=600';

const isStringArray = (value: unknown): value is ModelId[] =>
  Array.isArray(value) && value.every((item) => typeof item === 'string');

const fallback = (): Response =>
  Response.json([] as ModelId[], {
    headers: {
      'content-type': 'application/json',
      'x-models-source': 'error',
    },
    status: 200,
  });

export const GET = async (): Promise<Response> => {
  let upstream: Response;

  try {
    upstream = await fetch(`${API_BASE_URL}/chat/models`, {
      headers: { accept: 'application/json' },
      method: 'GET',
    });
  } catch {
    return fallback();
  }

  if (!upstream.ok) {
    return fallback();
  }

  let body: unknown;

  try {
    body = await upstream.json();
  } catch {
    return fallback();
  }

  if (!isStringArray(body)) {
    return fallback();
  }

  return Response.json(body, {
    headers: {
      'cache-control': CACHE_CONTROL,
      'content-type': 'application/json',
    },
    status: 200,
  });
};
