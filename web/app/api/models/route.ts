import type { ModelCatalog } from '@/lib/api-types';

import { API_BASE_URL, CHAT_API_KEY } from '@/lib/env';
import { parseModelCatalog } from '@/lib/model-catalog';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const CACHE_CONTROL = 'public, max-age=300, stale-while-revalidate=600';

const ERROR_CATALOG: ModelCatalog = { models: [], source: 'error', version: 1 };

const respond = (catalog: ModelCatalog, cache: boolean): Response =>
  Response.json(catalog, {
    headers: {
      ...(cache && { 'cache-control': CACHE_CONTROL }),
      'content-type': 'application/json',
      'x-models-source': catalog.source,
    },
    status: 200,
  });

const fallback = (): Response => respond(ERROR_CATALOG, false);

export const GET = async (): Promise<Response> => {
  let upstream: Response;

  try {
    upstream = await fetch(`${API_BASE_URL}/chat/models`, {
      headers: { accept: 'application/json', 'x-api-key': CHAT_API_KEY },
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

  const catalog = parseModelCatalog(body);
  if (catalog.source === 'error') {
    return fallback();
  }

  return respond(catalog, true);
};
