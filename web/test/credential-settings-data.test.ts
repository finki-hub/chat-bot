import { describe, expect, it } from 'vitest';

import { parseCredentials } from '@/components/shell/credential-settings-data';

const BASE_URL_FIELD = 'base_url';
const HAS_API_KEY_FIELD = 'has_api_key';
const USER_ID_FIELD = 'user_id';

describe('parseCredentials', () => {
  it('drops malformed credential records', () => {
    // Given: the credential service returns one complete record and one partial record.
    const credentials = [
      {
        [BASE_URL_FIELD]: null,
        [HAS_API_KEY_FIELD]: true,
        provider: 'openai',
        [USER_ID_FIELD]: '00000000-0000-4000-8000-000000000001',
      },
      {
        [BASE_URL_FIELD]: null,
        provider: 'google',
      },
    ];

    // When: the response is parsed at the UI boundary.
    const parsed = parseCredentials(credentials);

    // Then: only records matching the public credential contract are kept.
    expect(parsed).toStrictEqual([credentials[0]]);
  });
});
