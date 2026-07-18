/* eslint-disable camelcase -- wire-contract fixture fields */
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import type { ModelDescriptor } from '@/lib/api-types';

import {
  ComposerActions,
  type ComposerActionsProps,
} from '@/components/chat/composer-actions';

const LUNA_ID = 'gpt-5.6-luna';
const LUNA_NAME = 'GPT-5.6 Luna';
const LUNA_OPTION_NAME = /GPT-5\.6 Luna.*Бесплатно/u;

const createProps = (
  availability: NonNullable<ModelDescriptor['availability']>,
): ComposerActionsProps => ({
  availableProviders: new Set(),
  groups: [
    {
      models: [
        {
          availability,
          id: LUNA_ID,
          name: LUNA_NAME,
          provider: 'openai',
          sponsored_quota: {
            limit: 10,
            remaining: 8,
            resets_at: '2030-01-01T00:00:00Z',
          },
        },
      ],
      provider: 'openai',
    },
  ],
  isBusy: false,
  model: LUNA_ID,
  modelPlaceholder: 'Модел',
  modelSelectDisabled: false,
  onButtonClick: vi.fn<() => void>(),
  onModelChange: vi.fn<(model: string) => void>(),
  onReasoningChange: vi.fn<(reasoning: boolean) => void>(),
  reasoning: false,
  showModelPlaceholder: false,
  status: 'ready',
  submitDisabled: false,
});

describe('ComposerActions Luna availability', () => {
  it.each([
    ['sponsored', true, 'Бесплатно'],
    ['both', true, 'Бесплатно'],
    ['byok', false, 'Потребен е API клуч'],
    ['unavailable', false, 'Не е достапен'],
  ] as const)(
    'renders the %s state with the correct accessible label',
    async (availability, enabled, label) => {
      render(<ComposerActions {...createProps(availability)} />);
      const user = userEvent.setup();

      await user.click(screen.getByTestId('composer-model'));

      const options = await screen.findAllByRole('option');
      const option = options.find((candidate) =>
        candidate.textContent.includes(LUNA_NAME),
      );

      expect(option).not.toBeNull();

      if (option === undefined) {
        throw new Error('Luna option not found');
      }

      expect(option.getAttribute('aria-disabled')).toBe(
        enabled ? null : 'true',
      );
      expect(option).toHaveTextContent(label);

      expect(option.textContent.includes('8/10')).toBe(enabled);
    },
  );

  it('keeps the both quota unchanged after selecting Luna', async () => {
    const props = { ...createProps('both'), model: 'another-model' };
    render(<ComposerActions {...props} />);
    const user = userEvent.setup();

    await user.click(screen.getByTestId('composer-model'));
    const option = await screen.findByRole('option', {
      name: LUNA_OPTION_NAME,
    });
    await user.click(option);

    expect(props.onModelChange).toHaveBeenCalledWith(LUNA_ID);
    expect(screen.queryByText('8/10')).not.toBeInTheDocument();
  });
});
