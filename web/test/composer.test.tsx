import type { ComponentProps } from 'react';

import { fireEvent, render, screen } from '@testing-library/react';
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
  render(
    <Composer
      model={CLAUDE}
      models={[CLAUDE, GPT]}
      onModelChange={onModelChange}
      onStop={onStop}
      onSubmit={onSubmit}
      status="ready"
      {...overrides}
    />,
  );
  return { onModelChange, onStop, onSubmit };
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

  it('renders every model id as an option', () => {
    setup();

    expect(screen.getByRole('option', { name: CLAUDE })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: GPT })).toBeInTheDocument();
  });

  it('reports model changes', () => {
    const { onModelChange } = setup();
    fireEvent.change(screen.getByTestId('composer-model'), {
      target: { value: GPT },
    });

    expect(onModelChange).toHaveBeenCalledWith(GPT);
  });
});
