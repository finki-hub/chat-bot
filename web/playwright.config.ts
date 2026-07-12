import { defineConfig, devices } from '@playwright/test';

const isCI = Boolean(process.env['CI']);
const port = process.env['PLAYWRIGHT_PORT'] ?? '3000';
const baseURL =
  process.env['PLAYWRIGHT_BASE_URL'] ?? `http://127.0.0.1:${port}`;

export default defineConfig({
  forbidOnly: isCI,
  fullyParallel: true,
  projects: [
    {
      name: 'chromium',
      testIgnore: /mobile-chat\.spec\.ts/u,
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'mobile-chromium',
      testMatch: /mobile-chat\.spec\.ts/u,
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
});
