import type { ModelId } from '@/lib/api-types';

export const MAX_SUGGESTION_QUESTION_LENGTH = 2_000;
export const MAX_SUGGESTION_ANSWER_LENGTH = 8_000;
export const MAX_SUGGESTIONS = 3;

type FetchSuggestionsInput = {
  readonly answer: string;
  readonly conversationId: string;
  readonly providerModel?: ModelId;
  readonly question: string;
};

const parseSuggestions = (value: unknown): readonly string[] => {
  if (
    typeof value !== 'object' ||
    value === null ||
    !('suggestions' in value)
  ) {
    return [];
  }

  const { suggestions } = value;

  return Array.isArray(suggestions)
    ? suggestions
        .filter(
          (item): item is string =>
            typeof item === 'string' && item.trim() !== '',
        )
        .slice(0, MAX_SUGGESTIONS)
    : [];
};

// Suggestions are optional extras: any failure shows none.
export const fetchChatSuggestions = async ({
  answer,
  conversationId,
  providerModel,
  question,
}: FetchSuggestionsInput): Promise<readonly string[]> => {
  if (question.length > MAX_SUGGESTION_QUESTION_LENGTH) {
    return [];
  }

  try {
    const response = await fetch('/api/chat/suggestions', {
      body: JSON.stringify({
        // A long answer is cut: its start carries the topic.
        answer: answer.slice(0, MAX_SUGGESTION_ANSWER_LENGTH),
        question,
        ...(providerModel !== undefined && { providerModel }),
      }),
      headers: {
        'content-type': 'application/json',
        'x-conversation-id': conversationId,
      },
      method: 'POST',
    });

    return response.ok ? parseSuggestions(await response.json()) : [];
  } catch {
    return [];
  }
};
