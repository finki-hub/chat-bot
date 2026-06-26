import type { Metadata, Viewport } from 'next';
import type { ReactNode } from 'react';

import Script from 'next/script';

import './globals.css';

import { Providers } from '@/app/providers';

const TITLE = 'ФИНКИ Хаб / Чат';
const DESCRIPTION =
  'Разговарај со ФИНКИ Хаб асистентот за прашања поврзани со студиите на ФИНКИ.';
const SITE_URL = process.env['SITE_URL'] ?? 'https://chat.finki-hub.com';
const OG_IMAGE = `${SITE_URL}/favicon-96x96.png`;

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
    { color: '#ffffff', media: '(prefers-color-scheme: light)' },
    { color: '#0a0a0a', media: '(prefers-color-scheme: dark)' },
  ],
};

const noFlashTheme = `(() => {
  try {
    const stored = localStorage.getItem('theme');
    const theme =
      stored === 'dark' || stored === 'light'
        ? stored
        : matchMedia('(prefers-color-scheme: dark)').matches
          ? 'dark'
          : 'light';
    document.documentElement.dataset.kbTheme = theme;
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
      <Providers>{children}</Providers>
    </body>
  </html>
);

export default RootLayout;
