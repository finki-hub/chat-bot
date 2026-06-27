import type { ComponentProps } from 'react';

import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { Composer } from '@/components/chat/composer';
import { groupModelsByProvider } from '@/lib/use-models';

const CLAUDE = 'claude-sonnet-4-6';
const GPT = 'gpt-5.4-mini';

describe('groupModelsByProvider', () => {
  it('groups by the part before "/" when present', () => {
    const groups = groupModelsByProvider(['BAAI/bge-m3', 'BAAI/bge-large']);

    expect(groups).toStrictEqual([
      { models: ['BAAI/bge-large', 'BAAI/bge-m3'], provider: 'BAAI' },
    ]);
  });

  it('infers the provider from the name prefix when there is no slash', () => {
    const groups = groupModelsByProvider([CLAUDE, GPT, 'claude-opus-4-8']);

    expect(groups).toStrictEqual([
      { models: ['claude-opus-4-8', CLAUDE], provider: 'claude' },
      { models: [GPT], provider: 'gpt' },
    ]);
  });

  it('sorts providers and models and handles unknown ids', () => {
    const groups = groupModelsByProvider(['zeta', 'BAAI/bge-m3']);

    expect(groups).toStrictEqual([
      { models: ['BAAI/bge-m3'], provider: 'BAAI' },
      { models: ['zeta'], provider: 'zeta' },
    ]);
  });
});

const setup = (overrides: Partial<ComponentProps<typeof Composer>> = {}) => {
  const onSubmit = vi.fn<(text: string) => void>();
  const onStop = vi.fn<() => void>();
  const onModelChange = vi.fn<(model: string) => void>();
  const onReasoningChange = vi.fn<(reasoning: boolean) => void>();
  render(
    <Composer
      model={CLAUDE}
      models={[CLAUDE, GPT]}
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

  it('renders every model id as an option', async () => {
    setup();
    const user = userEvent.setup();
    await user.click(screen.getByTestId('composer-model'));

    await expect(
      screen.findByRole('option', { name: CLAUDE }),
    ).resolves.toBeInTheDocument();
    expect(screen.getByRole('option', { name: GPT })).toBeInTheDocument();
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
    await user.click(await screen.findByRole('option', { name: GPT }));

    expect(onModelChange).toHaveBeenCalledWith(GPT);
  });

  it('toggles reasoning for a reasoning-capable model', () => {
    const { onReasoningChange } = setup({ model: CLAUDE, reasoning: false });
    const pill = screen.getByTestId('composer-reasoning');

    expect(pill).toHaveAttribute('aria-pressed', 'false');

    fireEvent.click(pill);

    expect(onReasoningChange).toHaveBeenCalledWith(true);
  });

  it('disables the reasoning toggle for non-capable models', () => {
    setup({ model: 'llama3.3:70b', reasoning: false });

    expect(screen.getByTestId('composer-reasoning')).toBeDisabled();
  });
});
