import type { ModelCatalog } from '@/lib/api-types';

import { getAuthenticatedChatUserId } from '@/lib/authenticated-chat-user';
import { API_BASE_URL, CHAT_API_KEY } from '@/lib/env';
import { parseModelCatalog } from '@/lib/model-catalog';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const CACHE_CONTROL = 'private, no-store';

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
    const userId = await getAuthenticatedChatUserId();
    const url = `${API_BASE_URL}/chat/models?user_id=${encodeURIComponent(userId)}`;
    upstream = await fetch(url, {
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

  const parsedCatalog = parseModelCatalog(body);
  if (parsedCatalog.source === 'error') {
    return fallback();
  }

  return respond(parsedCatalog, true);
};
