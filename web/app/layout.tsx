import type { Metadata, Viewport } from 'next';
import type { ReactNode } from 'react';

import Script from 'next/script';

import './globals.css';

import { Providers } from '@/app/providers';
import { t } from '@/lib/i18n';

const TITLE = 'ФИНКИ Хаб / Чат';
const DESCRIPTION =
  'Разговарај со ФИНКИ Хаб асистентот за прашања поврзани со студиите на ФИНКИ.';
const SITE_URL = process.env['SITE_URL'] ?? 'https://chat.finki-hub.com';
const OG_IMAGE = `${SITE_URL}/favicon-96x96.png`;
const LIGHT_THEME_COLOR = '#ffffff';
const DARK_THEME_COLOR = '#0a0a0a';

export const metadata: Metadata = {
  alternates: {
    canonical: SITE_URL,
  },
  authors: [{ name: 'ФИНКИ Хаб' }],
  description: DESCRIPTION,
  icons: {
    apple: { sizes: '180x180', url: '/apple-touch-icon.png' },
    icon: [
      { sizes: '96x96', type: 'image/png', url: '/favicon-96x96.png' },
      { type: 'image/svg+xml', url: '/favicon.svg' },
    ],
    shortcut: '/favicon.ico',
  },
  manifest: '/site.webmanifest',
  metadataBase: new URL(SITE_URL),
  openGraph: {
    description: DESCRIPTION,
    images: [OG_IMAGE],
    locale: 'mk_MK',
    title: TITLE,
    type: 'website',
    url: SITE_URL,
  },
  title: TITLE,
  twitter: {
    card: 'summary',
    description: DESCRIPTION,
    images: [OG_IMAGE],
    title: TITLE,
  },
};

export const viewport: Viewport = {
  themeColor: [
    { color: LIGHT_THEME_COLOR, media: '(prefers-color-scheme: light)' },
    { color: DARK_THEME_COLOR, media: '(prefers-color-scheme: dark)' },
  ],
};

const noFlashTheme = `(() => {
  try {
    const themeColors = { dark: '${DARK_THEME_COLOR}', light: '${LIGHT_THEME_COLOR}' };
    const stored = localStorage.getItem('theme');
    const theme =
      stored === 'dark' || stored === 'light'
        ? stored
        : typeof matchMedia === 'function' && matchMedia('(prefers-color-scheme: dark)').matches
          ? 'dark'
          : 'light';
    const root = document.documentElement;
    root.dataset.kbTheme = theme;
    root.style.colorScheme = theme;
    document
      .querySelectorAll('meta[name="theme-color"]')
      .forEach((meta) => {
        meta.setAttribute('content', themeColors[theme]);
      });
  } catch {}
})();`;

const RootLayout = ({ children }: Readonly<{ children: ReactNode }>) => (
  <html
    lang="mk"
    suppressHydrationWarning
  >
    <head>
      <Script
        id="no-flash-theme"
        strategy="beforeInteractive"
      >
        {noFlashTheme}
      </Script>
    </head>
    <body>
      <a
        className="fixed left-4 top-4 z-50 -translate-y-24 rounded-md border border-border bg-background px-4 py-2 text-sm font-medium text-foreground shadow-lg transition-transform focus:translate-y-0 focus:outline-none focus:ring-2 focus:ring-ring"
        href="#main-content"
      >
        {t('navigation.skipToContent')}
      </a>
      <Providers>{children}</Providers>
    </body>
  </html>
);

export default RootLayout;
