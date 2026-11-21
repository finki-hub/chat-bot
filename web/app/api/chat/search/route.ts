import {
  AuthenticationRequiredError,
  getAuthenticatedChatUserId,
} from '@/lib/authenticated-chat-user';
import {
  ChatStateRequestError,
  createChatStateClient,
} from '@/lib/chat-state-client';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const MAX_QUERY_LENGTH = 200;
const DEFAULT_LIMIT = 20;
const MAX_LIMIT = 50;

const parseInteger = (
  value: null | string,
  fallback: number,
  minimum: number,
  maximum: number,
): number => {
  if (value === null || value.trim() === '') {
    return fallback;
  }
  const parsed = Number(value);
  return Number.isSafeInteger(parsed)
    ? Math.min(Math.max(parsed, minimum), maximum)
    : fallback;
};

export const GET = async (request: Request): Promise<Response> => {
  const url = new URL(request.url);
  const query = (url.searchParams.get('q') ?? '').trim();
  if (query.length < 2 || query.length > MAX_QUERY_LENGTH) {
    return Response.json({ error: 'Invalid search query' }, { status: 400 });
  }

  const limit = parseInteger(
    url.searchParams.get('limit'),
    DEFAULT_LIMIT,
    1,
    MAX_LIMIT,
  );
  const offset = parseInteger(
    url.searchParams.get('offset'),
    0,
    0,
    Number.MAX_SAFE_INTEGER,
  );

  try {
    const userId = await getAuthenticatedChatUserId();
    const result = await createChatStateClient().searchConversations({
      limit,
      offset,
      query,
      userId,
    });

    return Response.json({
      query: result.query,
      results: result.results.map((row) => ({
        conversationId: row.conversation_id,
        conversationTitle: row.conversation_title,
        conversationUpdatedAt: row.conversation_updated_at,
        message:
          row.message === null
            ? null
            : {
                createdAt: row.message.created_at,
                id: row.message.id,
                role: row.message.role,
              },
        snippet: row.snippet,
      })),
    });
  } catch (error) {
    if (error instanceof AuthenticationRequiredError) {
      return Response.json(
        { error: 'Authentication required' },
        { status: 401 },
      );
    }
    if (error instanceof ChatStateRequestError) {
      return Response.json(
        { error: 'Search failed' },
        { status: error.status },
      );
    }
    if (error instanceof TypeError) {
      return Response.json({ error: 'Search is unavailable' }, { status: 502 });
    }

    throw error;
  }
};
