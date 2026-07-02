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

const fallbackTheme = (): Theme => (prefersDark() ? 'dark' : 'light');

const readStoredThemeValue = (): null | string => {
  try {
    return localStorage.getItem(storageKey);
  } catch {
    return null;
  }
};

const writeStoredTheme = (theme: Theme): boolean => {
  try {
    localStorage.setItem(storageKey, theme);
    return true;
  } catch {
    return false;
  }
};

const readStoredTheme = (): Theme => {
  if (typeof localStorage === 'undefined') {
    return fallbackTheme();
  }

  const stored = readStoredThemeValue();
  if (isTheme(stored)) {
    return stored;
  }

  return fallbackTheme();
};

const syncThemeChrome = (theme: Theme) => {
  document.documentElement.style.colorScheme = theme;
  document
    .querySelectorAll<HTMLMetaElement>('meta[name="theme-color"]')
    .forEach((meta) => {
      meta.setAttribute('content', themeColors[theme]);
    });
};

const applyTheme = (theme: Theme) => {
  document.documentElement.dataset['kbTheme'] = theme;
  syncThemeChrome(theme);
  writeStoredTheme(theme);
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
