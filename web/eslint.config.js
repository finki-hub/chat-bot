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
    files: ['app/**/*.{ts,tsx}'],
    rules: {
      'react-refresh/only-export-components': 'off',
    },
  },
  {
    files: ['test/**/*.{ts,tsx}'],
    rules: {
      'vitest/prefer-to-be-falsy': 'off',
      'vitest/prefer-to-be-truthy': 'off',
    },
  },
  {
    files: ['e2e/**/*.{ts,tsx}'],
    rules: vitestRulesOff,
  },
];

export default config;
