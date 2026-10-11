import { devices, type PlaywrightTestConfig } from '@playwright/test';

const isCI = Boolean(process.env['CI']);
const port = process.env['PLAYWRIGHT_PORT'] ?? '3000';
const baseURL =
  process.env['PLAYWRIGHT_BASE_URL'] ?? `http://127.0.0.1:${port}`;
const DESKTOP_CHAT_REGEX = /mobile-chat\.spec\.ts/u;
const MOBILE_CHAT_REGEX = /mobile-chat\.spec\.ts/u;

const config = {
  forbidOnly: isCI,
  fullyParallel: true,
  projects: [
    {
      name: 'chromium',
      testIgnore: DESKTOP_CHAT_REGEX,
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'mobile-chromium',
      testMatch: MOBILE_CHAT_REGEX,
      use: { ...devices['Pixel 5'] },
    },
  ],
  retries: isCI ? 2 : 0,
  testDir: './e2e',
  use: {
    baseURL,
    trace: 'on-first-retry',
  },
  webServer: {
    command: `npm run dev -- --hostname 127.0.0.1 --port ${port}`,
    env: {
      ...process.env,
      PLAYWRIGHT_AUTH_BYPASS: '1',
    },
    reuseExistingServer:
      !isCI && process.env['PLAYWRIGHT_REUSE_SERVER'] === '1',
    timeout: 120_000,
    url: baseURL,
  },
} satisfies PlaywrightTestConfig;

export default config;
