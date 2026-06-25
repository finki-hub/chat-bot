import type { Metadata } from 'next';
import type { ReactNode } from 'react';

import Script from 'next/script';

import './globals.css';

import { Providers } from '@/app/providers';

export const metadata: Metadata = {
  description: 'FINKI Hub чат асистент',
  icons: {
    apple: '/apple-touch-icon.png',
    icon: [
      { rel: 'icon', url: '/favicon.ico' },
      { type: 'image/svg+xml', url: '/favicon.svg' },
      { sizes: '96x96', type: 'image/png', url: '/favicon-96x96.png' },
    ],
  },
  title: 'FINKI Hub Chat',
};

// Resolve the theme before first paint to avoid a flash of the wrong theme.
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
