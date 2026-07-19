import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { ErrorNotice } from '@/lib/api-types';

import { MessageError } from '@/components/chat/message';

const errorPart: ErrorNotice = {
  code: 'free_quota_exhausted',
  message: 'backend quota detail must not render',
  // eslint-disable-next-line camelcase -- mirrors the backend wire contract.
  resets_at: '2030-01-01T00:00:00Z',
};

describe('MessageError', () => {
  it('uses the generic manage-credentials label for sponsored quota errors', () => {
    const onManageCredentials = vi.fn<() => void>();

    render(
      <MessageError
        errorPart={errorPart}
        onManageCredentials={onManageCredentials}
      />,
    );

    expect(
      screen.getByRole('button', { name: 'Додај API клуч' }),
    ).toBeVisible();
  });
});
