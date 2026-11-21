import { API_BASE_URL, CHAT_API_KEY } from '@/lib/env';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const MAX_BODY_BYTES = 20_000;
const RATE_LIMIT_WINDOW_MS = 60_000;
const RATE_LIMIT_MAX = 6;
const RATE_LIMIT_MAX_KEYS = 5_000;
const RATE_LIMIT_PRUNE_THRESHOLD = 1_000;

type RateLimitEntry = {
  count: number;
  resetAt: number;
};

const rateLimits = new Map<string, RateLimitEntry>();

const normalizeIp = (value: null | string): null | string => {
  const trimmed = value?.trim();
  return trimmed && trimmed.length > 0 ? trimmed : null;
};

// The reverse proxy is expected to set X-Real-IP or overwrite X-Forwarded-For.
// Prefer the proxy-set header and use the last forwarded hop, so a
// client-supplied prefix cannot rotate the rate-limit key.
const clientKey = (request: Request): string => {
  const realIp = normalizeIp(request.headers.get('x-real-ip'));
  if (realIp) return realIp;

  const forwarded = normalizeIp(request.headers.get('x-forwarded-for'));
  if (forwarded) {
    const last = normalizeIp(forwarded.split(',').at(-1) ?? null);
    if (last) return last;
  }

  return 'unknown';
};

const pruneExpired = (now: number): void => {
  for (const [key, entry] of rateLimits) {
    if (entry.resetAt <= now) {
      rateLimits.delete(key);
    }
  }
};

const isRateLimited = (request: Request): boolean => {
  const key = clientKey(request);
  const now = Date.now();

  if (rateLimits.size > RATE_LIMIT_PRUNE_THRESHOLD) {
    pruneExpired(now);
  }

  const current = rateLimits.get(key);

  if (current === undefined) {
    if (rateLimits.size >= RATE_LIMIT_MAX_KEYS) {
      return true;
    }
    rateLimits.set(key, { count: 1, resetAt: now + RATE_LIMIT_WINDOW_MS });
    return false;
  }

  if (current.resetAt <= now) {
    rateLimits.set(key, { count: 1, resetAt: now + RATE_LIMIT_WINDOW_MS });
    return false;
  }

  if (current.count >= RATE_LIMIT_MAX) {
    return true;
  }

  current.count += 1;
  return false;
};

const parsePayload = async (request: Request): Promise<null | object> => {
  const contentLength = request.headers.get('content-length');
  if (contentLength !== null && Number(contentLength) > MAX_BODY_BYTES) {
    return null;
  }

  const raw = await request.text();
  if (raw.length > MAX_BODY_BYTES) {
    return null;
  }

  try {
    const payload: unknown = JSON.parse(raw);
    return typeof payload === 'object' &&
      payload !== null &&
      !Array.isArray(payload)
      ? payload
      : null;
  } catch {
    return null;
  }
};

export const POST = async (request: Request): Promise<Response> => {
  if (isRateLimited(request)) {
    return Response.json(
      { error: 'Too many requests. Try again in a minute.' },
      { headers: { 'retry-after': '60' }, status: 429 },
    );
  }

  const payload = await parsePayload(request);
  if (payload === null) {
    return Response.json(
      { error: 'Invalid committee request' },
      { status: 400 },
    );
  }

  try {
    const includePriorKey = 'include_prior';
    const forwarded = { ...payload, [includePriorKey]: false };
    const response = await fetch(`${API_BASE_URL}/recommendations/`, {
      body: JSON.stringify(forwarded),
      cache: 'no-store',
      headers: {
        'content-type': 'application/json',
        'x-api-key': CHAT_API_KEY,
      },
      method: 'POST',
      signal: AbortSignal.timeout(45_000),
    });

    if (response.status === 401) {
      return Response.json(
        { error: 'Committee service is unavailable' },
        { status: 502 },
      );
    }

    const body = await response.text();
    return new Response(body, {
      headers: {
        'content-type':
          response.headers.get('content-type') ?? 'application/json',
      },
      status: response.status,
    });
  } catch {
    return Response.json(
      { error: 'Committee service is unavailable' },
      { status: 502 },
    );
  }
};
