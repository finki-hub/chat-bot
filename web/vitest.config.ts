import react from '@vitejs/plugin-react';
import { join } from 'node:path';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': import.meta.dirname,
      'server-only': join(import.meta.dirname, 'test/stubs/server-only.ts'),
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    include: ['test/**/*.test.{ts,tsx}'],
    maxWorkers: 4,
    setupFiles: ['./vitest.setup.ts'],
  },
});
