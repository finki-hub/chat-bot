import { defineConfig, devices } from '@playwright/test';

const isCI = Boolean(process.env['CI']);

export default defineConfig({
  forbidOnly: isCI,
  fullyParallel: true,
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  retries: isCI ? 2 : 0,
  testDir: './e2e',
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
  },
  webServer: {
    command: 'npm run dev',
    reuseExistingServer: !isCI,
    timeout: 120_000,
    url: 'http://localhost:3000',
  },
});
