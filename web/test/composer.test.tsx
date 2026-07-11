import type { ComponentProps } from 'react';

import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import type { ModelDescriptor } from '@/lib/api-types';

import { Composer } from '@/components/chat/composer';

const CLAUDE_ID = 'claude-sonnet-5';
const CLAUDE_NAME = 'Claude Sonnet 5';
const GPT_ID = 'gpt-5.4-mini';
const GPT_NAME = 'GPT-5.4 Mini';
const GPT_PREMIUM_ID = 'gpt-5.4';
const GPT_PREMIUM_NAME = 'GPT-5.4';
const GPT_CHEAP_ID = 'gpt-5.4-nano';
const GPT_CHEAP_NAME = 'GPT-5.4 Nano';

const MODELS: ModelDescriptor[] = [
  {
    id: GPT_PREMIUM_ID,
    name: GPT_PREMIUM_NAME,
    provider: 'openai',
    tier: 'premium',
  },
  { id: GPT_ID, name: GPT_NAME, provider: 'openai', tier: 'default' },
  { id: CLAUDE_ID, name: CLAUDE_NAME, provider: 'anthropic', tier: 'default' },
  { id: GPT_CHEAP_ID, name: GPT_CHEAP_NAME, provider: 'openai', tier: 'cheap' },
];

const setup = (overrides: Partial<ComponentProps<typeof Composer>> = {}) => {
  const onSubmit = vi.fn<(text: string) => void>();
  const onStop = vi.fn<() => void>();
  const onModelChange = vi.fn<(model: string) => void>();
  const onReasoningChange = vi.fn<(reasoning: boolean) => void>();
  render(
    <Composer
      model={CLAUDE_ID}
      models={MODELS}
      onModelChange={onModelChange}
      onReasoningChange={onReasoningChange}
      onStop={onStop}
      onSubmit={onSubmit}
      reasoning={false}
      status="ready"
      {...overrides}
    />,
  );
  return { onModelChange, onReasoningChange, onStop, onSubmit };
};

describe('Composer', () => {
  it('submits trimmed text on Enter (no Shift)', () => {
    const { onSubmit } = setup();
    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: '  Здраво  ' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });

    expect(onSubmit).toHaveBeenCalledWith('Здраво');
  });

  it('does NOT submit on Shift+Enter (newline)', () => {
    const { onSubmit } = setup();
    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: 'ред' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: true });

    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('does not submit empty/whitespace-only input', () => {
    const { onSubmit } = setup();
    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: ' '.repeat(3) } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });

    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('calls onStop when the submit button is clicked while streaming', () => {
    const { onStop } = setup({ status: 'streaming' });
    fireEvent.click(screen.getByTestId('composer-submit'));

    expect(onStop).toHaveBeenCalledOnce();
  });

  it('keeps the Stop button clickable while streaming even when disabled', () => {
    const { onStop } = setup({ disabled: true, status: 'streaming' });
    const button = screen.getByTestId('composer-submit');

    expect(button).toBeEnabled();

    fireEvent.click(button);

    expect(onStop).toHaveBeenCalledOnce();
  });

  it('disables the input and submit button when disabled and idle', () => {
    setup({ disabled: true, status: 'ready' });

    expect(screen.getByTestId('composer-input')).toBeDisabled();
    expect(screen.getByTestId('composer-submit')).toBeDisabled();
  });

  it('renders every model as an option, labelled by its display name', async () => {
    setup();
    const user = userEvent.setup();
    await user.click(screen.getByTestId('composer-model'));

    await expect(
      screen.findByRole('option', { name: CLAUDE_NAME }),
    ).resolves.toBeInTheDocument();
    expect(screen.getByRole('option', { name: GPT_NAME })).toBeInTheDocument();
    expect(
      screen.getByRole('option', { name: GPT_PREMIUM_NAME }),
    ).toBeInTheDocument();
  });

  it('renders accessible tier groups before their provider subgroups', async () => {
    setup();
    const user = userEvent.setup();
    await user.click(screen.getByTestId('composer-model'));

    const groupLabels = await screen.findAllByTestId('model-tier-label');

    expect(groupLabels.map((label) => label.textContent)).toStrictEqual([
      'Премиум',
      'Стандарден',
      'Економичен',
    ]);

    const providerLabels = screen.getAllByTestId('model-provider-label');

    expect(providerLabels.map((label) => label.textContent)).toStrictEqual([
      'OpenAI',
      'OpenAI',
      'Anthropic',
      'OpenAI',
    ]);
  });

  it('focuses the input when typing starts elsewhere on the page', () => {
    setup({ status: 'error' });
    const textarea = screen.getByTestId('composer-input');

    expect(textarea).not.toHaveFocus();

    fireEvent.keyDown(document.body, { key: 'к' });

    expect(textarea).toHaveFocus();
  });

  it('ignores shortcuts and non-character keys for type-to-focus', () => {
    setup({ status: 'error' });
    const textarea = screen.getByTestId('composer-input');

    fireEvent.keyDown(document.body, { ctrlKey: true, key: 'a' });
    fireEvent.keyDown(document.body, { key: 'ArrowDown' });

    expect(textarea).not.toHaveFocus();
  });

  it('reports model changes', async () => {
    const { onModelChange } = setup();
    const user = userEvent.setup();
    await user.click(screen.getByTestId('composer-model'));
    await user.click(await screen.findByRole('option', { name: GPT_NAME }));

    expect(onModelChange).toHaveBeenCalledWith(GPT_ID);
  });

  it('toggles reasoning for a reasoning-capable model', () => {
    const { onReasoningChange } = setup({ model: CLAUDE_ID, reasoning: false });
    const pill = screen.getByTestId('composer-reasoning');

    expect(pill).toHaveAttribute('aria-pressed', 'false');

    fireEvent.click(pill);

    expect(onReasoningChange).toHaveBeenCalledWith(true);
  });

  it('disables the reasoning toggle for non-capable models', () => {
    setup({ model: 'qwen3:30b-a3b-instruct-2507-q4_K_M', reasoning: false });

    expect(screen.getByTestId('composer-reasoning')).toBeDisabled();
  });
});
