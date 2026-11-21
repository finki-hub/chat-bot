import { API_BASE_URL, CHAT_API_KEY } from '@/lib/env';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const upstreamError = (status: number): Response =>
  Response.json(
    { error: 'Professor directory is unavailable' },
    { status: status === 401 ? 502 : status },
  );

export const GET = async (): Promise<Response> => {
  try {
    const response = await fetch(`${API_BASE_URL}/professors/`, {
      cache: 'no-store',
      headers: { 'x-api-key': CHAT_API_KEY },
    });

    if (!response.ok) {
      return upstreamError(response.status);
    }

    return Response.json(await response.json(), {
      headers: {
        'cache-control': 'public, max-age=300, stale-while-revalidate=3600',
      },
    });
  } catch {
    return upstreamError(502);
  }
};
