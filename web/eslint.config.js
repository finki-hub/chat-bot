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

// The imperium `vitest` preset is applied globally, but Playwright specs use
// their own `test`/`expect` from @playwright/test (not the vitest API), so those
// rules misfire on valid e2e code. Derive an off-map for every vitest/* rule the
// preset enables and disable them just for the e2e scope.
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
    // App Router files legitimately export non-components (metadata,
    // generateMetadata, route handlers GET/POST, …), so Fast Refresh's
    // "only export components" rule does not apply to them.
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
