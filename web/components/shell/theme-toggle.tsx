import { MoonIcon, SunIcon } from 'lucide-react';
import { useEffect, useState } from 'react';

import { IconButton } from '@/components/ui/icon-controls';
import { t } from '@/lib/i18n';

type Theme = 'dark' | 'light';

const storageKey = 'theme';
const themeColors = {
  dark: '#0a0a0a',
  light: '#ffffff',
} satisfies Record<Theme, string>;

const isTheme = (value: null | string): value is Theme =>
  value === 'dark' || value === 'light';

const prefersDark = (): boolean =>
  typeof matchMedia === 'function' &&
  matchMedia('(prefers-color-scheme: dark)').matches;

const readStoredTheme = (): Theme => {
  if (typeof localStorage === 'undefined') {
    return 'light';
  }

  const stored = localStorage.getItem(storageKey);
  if (isTheme(stored)) {
    return stored;
  }

  return prefersDark() ? 'dark' : 'light';
};

const syncThemeChrome = (theme: Theme) => {
  document.documentElement.style.colorScheme = theme;
  document
    .querySelector<HTMLMetaElement>('meta[name="theme-color"]')
    ?.setAttribute('content', themeColors[theme]);
};

const applyTheme = (theme: Theme) => {
  document.documentElement.dataset['kbTheme'] = theme;
  syncThemeChrome(theme);
  localStorage.setItem(storageKey, theme);
};

export const ThemeToggle = () => {
  const [theme, setTheme] = useState<null | Theme>(null);

  useEffect(() => {
    setTheme(readStoredTheme());
  }, []);

  useEffect(() => {
    if (theme === null) {
      return;
    }

    applyTheme(theme);
  }, [theme]);

  const displayedTheme = theme ?? 'light';

  return (
    <IconButton
      aria-label={t('header.theme')}
      onClick={() => {
        setTheme((currentTheme) => {
          const activeTheme = currentTheme ?? readStoredTheme();

          return activeTheme === 'dark' ? 'light' : 'dark';
        });
      }}
    >
      {displayedTheme === 'dark' ? (
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
