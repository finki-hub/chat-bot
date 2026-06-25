import { MoonIcon, SunIcon } from 'lucide-react';
import { useEffect, useState } from 'react';

import { IconButton } from '@/components/ui/icon-controls';
import { t } from '@/lib/i18n';

type Theme = 'dark' | 'light';

const storageKey = 'theme';

const isTheme = (value: null | string): value is Theme =>
  value === 'dark' || value === 'light';

const prefersDark = (): boolean =>
  typeof matchMedia === 'function' &&
  matchMedia('(prefers-color-scheme: dark)').matches;

const readStoredTheme = (): Theme => {
  const stored = localStorage.getItem(storageKey);
  if (isTheme(stored)) {
    return stored;
  }

  return prefersDark() ? 'dark' : 'light';
};

const applyTheme = (theme: Theme) => {
  document.documentElement.dataset['kbTheme'] = theme;
  localStorage.setItem(storageKey, theme);
};

export const ThemeToggle = () => {
  const [theme, setTheme] = useState<Theme>('light');

  useEffect(() => {
    setTheme(readStoredTheme());
  }, []);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  return (
    <IconButton
      aria-label={t('header.theme')}
      onClick={() => {
        setTheme((currentTheme) =>
          currentTheme === 'dark' ? 'light' : 'dark',
        );
      }}
    >
      {theme === 'dark' ? (
        <SunIcon
          aria-hidden="true"
          className="h-4 w-4"
        />
      ) : (
        <MoonIcon
          aria-hidden="true"
          className="h-4 w-4"
        />
      )}
    </IconButton>
  );
};
