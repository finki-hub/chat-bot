import { API_BASE_URL } from '@/lib/env';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const HEALTH_TIMEOUT_MS = 5_000;

const result = (ok: boolean): Response =>
  Response.json(
    { ok },
    {
      headers: { 'cache-control': 'no-store' },
      status: 200,
    },
  );

export const GET = async (): Promise<Response> => {
  try {
    const upstream = await fetch(`${API_BASE_URL}/health/health`, {
      headers: { accept: 'application/json' },
      method: 'GET',
      signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS),
    });

    return result(upstream.ok);
  } catch {
    return result(false);
  }
};
