import type { Page } from '@playwright/test';

// Mirrors the typed catalog the API now serves at GET /chat/models. Spans every
// provider and tier, and includes the web default (claude-sonnet-5) plus every id
// the specs reference, so stale-model recovery never fires mid-run.
export const MODEL_CATALOG = {
  models: [
    { id: 'gpt-5.4', name: 'GPT-5.4', provider: 'openai', tier: 'premium' },
    {
      id: 'gpt-5.4-mini',
      name: 'GPT-5.4 Mini',
      provider: 'openai',
      tier: 'default',
    },
    {
      id: 'gpt-5.4-nano',
      name: 'GPT-5.4 Nano',
      provider: 'openai',
      tier: 'cheap',
    },
    {
      id: 'gemini-2.5-pro',
      name: 'Gemini 2.5 Pro',
      provider: 'google',
      tier: 'premium',
    },
    {
      id: 'gemini-2.5-flash',
      name: 'Gemini 2.5 Flash',
      provider: 'google',
      tier: 'default',
    },
    {
      id: 'claude-opus-4-8',
      name: 'Claude Opus 4.8',
      provider: 'anthropic',
      tier: 'premium',
    },
    {
      id: 'claude-sonnet-5',
      name: 'Claude Sonnet 5',
      provider: 'anthropic',
      tier: 'default',
    },
    {
      id: 'claude-haiku-4-5',
      name: 'Claude Haiku 4.5',
      provider: 'anthropic',
      tier: 'cheap',
    },
    {
      id: 'llama3.3:70b',
      name: 'Llama 3.3 70B',
      provider: 'ollama',
      tier: 'default',
    },
    {
      id: 'deepseek-r1:70b',
      name: 'DeepSeek R1 70B',
      provider: 'ollama',
      tier: 'default',
    },
    {
      id: 'hf.co/LVSTCK/domestic-yak-8B-instruct-GGUF:Q8_0',
      name: 'Domestic Yak 8B Instruct',
      provider: 'ollama',
      tier: 'cheap',
    },
    {
      id: 'hf.co/mradermacher/VezilkaLLM-GGUF:Q8_0',
      name: 'VezilkaLLM',
      provider: 'ollama',
      tier: 'cheap',
    },
  ],
  source: 'live',
  version: 1,
};

export const mockModels = async (page: Page): Promise<void> => {
  await page.route('**/api/models', async (route) => {
    await route.fulfill({
      body: JSON.stringify(MODEL_CATALOG),
      contentType: 'application/json',
      status: 200,
    });
  });
};
