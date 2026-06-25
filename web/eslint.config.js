import {
  base,
  browser,
  jsxA11y,
  node,
  perfectionist,
  prettier,
  react,
  typescript,
  vitest,
} from 'eslint-config-imperium';

// Playwright specs use their own `test`/`expect` (not the vitest API), so the
// globally-applied vitest/* rules misfire on valid e2e code; turn them off for e2e.
const vitestRuleKeys = (Array.isArray(vitest) ? vitest : [vitest])
  .flatMap((block) => Object.keys(block?.rules ?? {}))
  .filter((key) => key.startsWith('vitest/'));
const vitestRulesOff = Object.fromEntries(
  vitestRuleKeys.map((key) => [key, 'off']),
);

const config = [
  {
    ignores: [
      '.next',
      'node_modules',
      'next-env.d.ts',
      'coverage',
      'playwright-report',
      'test-results',
      'components/ui/**',
      'components/ai-elements/**',
    ],
  },
  ...base,
  browser,
  node,
  react,
  jsxA11y,
  typescript,
  vitest,
  prettier,
  perfectionist,
  {
    // App Router files legitimately export non-components (metadata, route
    // handlers, …), so Fast Refresh's "only export components" rule misfires here.
    files: ['app/**/*.{ts,tsx}'],
    rules: {
      'react-refresh/only-export-components': 'off',
    },
  },
  {
    files: ['e2e/**/*.{ts,tsx}'],
    rules: vitestRulesOff,
  },
];

export default config;
