import type { ComponentProps } from 'react';

import { fireEvent, render, screen, waitFor } from '@testing-library/react';
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
const OLLAMA_ID = 'llama3.2:latest';
const OLLAMA_NAME = 'llama3.2:latest';
const OLLAMA_OPTION_NAME = /llama3\.2:latest/u;
const OLLAMA_UNKNOWN_ID = 'qwen3:14b';
const OLLAMA_UNKNOWN_NAME = 'qwen3:14b';
const MODEL_SELECTOR_TEST_ID = 'composer-model';

const MODELS: ModelDescriptor[] = [
  {
    id: GPT_PREMIUM_ID,
    name: GPT_PREMIUM_NAME,
    provider: 'openai',
  },
  { id: GPT_ID, name: GPT_NAME, provider: 'openai' },
  { id: CLAUDE_ID, name: CLAUDE_NAME, provider: 'anthropic' },
  { id: OLLAMA_ID, loaded: true, name: OLLAMA_NAME, provider: 'ollama' },
  {
    id: OLLAMA_UNKNOWN_ID,
    loaded: null,
    name: OLLAMA_UNKNOWN_NAME,
    provider: 'ollama',
  },
  { id: GPT_CHEAP_ID, name: GPT_CHEAP_NAME, provider: 'openai' },
];
const ALL_PROVIDERS = new Set(['anthropic', 'ollama', 'openai']);

const setup = (overrides: Partial<ComponentProps<typeof Composer>> = {}) => {
  const onSubmit = vi
    .fn<(text: string) => Promise<boolean>>()
    .mockResolvedValue(true);
  const onStop = vi.fn<() => void>();
  const onModelChange = vi.fn<(model: string) => void>();
  const onReasoningChange = vi.fn<(reasoning: boolean) => void>();
  render(
    <Composer
      availableProviders={ALL_PROVIDERS}
      credentialsLoading={false}
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
  it('submits trimmed text on Enter (no Shift)', async () => {
    const { onSubmit } = setup();
    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: '  Здраво  ' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });

    await waitFor(() => {
      expect(onSubmit).toHaveBeenCalledWith('Здраво');
    });
  });

  it('keeps the draft when submission fails before the message is accepted', async () => {
    const user = userEvent.setup();
    setup({
      onSubmit: vi
        .fn<(text: string) => Promise<boolean>>()
        .mockResolvedValue(false),
    });
    const textarea = screen.getByRole('textbox');

    await user.type(textarea, 'Не го губи прашањето');
    await user.keyboard('{Enter}');

    expect(textarea).toHaveValue('Не го губи прашањето');
  });

  it('clears the draft after submission is accepted', async () => {
    const user = userEvent.setup();
    setup();
    const textarea = screen.getByRole('textbox');

    await user.type(textarea, 'Зачувај го прашањето');
    await user.keyboard('{Enter}');

    await waitFor(() => {
      expect(textarea).toHaveValue('');
    });
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
    await user.click(screen.getByTestId(MODEL_SELECTOR_TEST_ID));

    await expect(
      screen.findByRole('option', { name: CLAUDE_NAME }),
    ).resolves.toBeInTheDocument();
    expect(screen.getByRole('option', { name: GPT_NAME })).toBeInTheDocument();
    expect(
      screen.getByRole('option', { name: GPT_PREMIUM_NAME }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('option', { name: OLLAMA_OPTION_NAME }),
    ).toHaveTextContent('Вчитан');
    expect(
      screen.getByRole('option', { name: OLLAMA_UNKNOWN_NAME }),
    ).not.toHaveTextContent('Не е вчитан');
  });

  it('groups models only by provider in catalog order', async () => {
    setup();
    const user = userEvent.setup();
    await user.click(screen.getByTestId(MODEL_SELECTOR_TEST_ID));

    const providerLabels = await screen.findAllByTestId('model-provider-label');

    expect(providerLabels.map((label) => label.textContent)).toStrictEqual([
      'OpenAI',
      'Anthropic',
      'Ollama',
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
    await user.click(screen.getByTestId(MODEL_SELECTOR_TEST_ID));
    await user.click(await screen.findByRole('option', { name: GPT_NAME }));

    expect(onModelChange).toHaveBeenCalledWith(GPT_ID);
  });

  it('shows models without provider credentials as disabled', async () => {
    const { onModelChange } = setup({
      availableProviders: new Set(['anthropic']),
    });
    const user = userEvent.setup();

    await user.click(screen.getByTestId(MODEL_SELECTOR_TEST_ID));

    const unavailableLabel = await screen.findByText(GPT_NAME);
    const unavailable = unavailableLabel.closest('[role="option"]');

    expect(unavailable).not.toBeNull();

    if (unavailable === null) {
      throw new Error('Unavailable model option not found');
    }

    expect(unavailable).toHaveAttribute('aria-disabled', 'true');
    expect(unavailable).toHaveTextContent('Потребен е API клуч');

    await user.click(unavailable);

    expect(onModelChange).not.toHaveBeenCalled();
  });

  it('keeps configured provider models selectable', async () => {
    const { onModelChange } = setup({
      availableProviders: new Set(['openai']),
      model: GPT_PREMIUM_ID,
    });
    const user = userEvent.setup();

    await user.click(screen.getByTestId(MODEL_SELECTOR_TEST_ID));
    await user.click(await screen.findByRole('option', { name: GPT_NAME }));

    expect(onModelChange).toHaveBeenCalledWith(GPT_ID);
  });

  it('blocks submission when the selected model provider is unavailable', () => {
    const { onSubmit } = setup({ availableProviders: new Set(['openai']) });
    const textarea = screen.getByRole('textbox');

    fireEvent.change(textarea, { target: { value: 'Прашање' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });

    expect(onSubmit).not.toHaveBeenCalled();
    expect(screen.getByTestId('composer-submit')).toBeDisabled();
  });

  it('surfaces credential loading errors and disables model selection', () => {
    setup({ credentialsError: true });

    expect(screen.getByTestId(MODEL_SELECTOR_TEST_ID)).toBeDisabled();
    expect(screen.getByTestId(MODEL_SELECTOR_TEST_ID)).toHaveTextContent(
      'API клучевите се недостапни',
    );
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
