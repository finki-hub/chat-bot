import type { Metadata } from 'next';
import type { ReactNode } from 'react';

import './globals.css';

import { Providers } from '@/app/providers';

export const metadata: Metadata = {
  description: 'FINKI Hub чат асистент',
  title: 'FINKI Hub Chat',
};

const RootLayout = ({ children }: Readonly<{ children: ReactNode }>) => (
  <html lang="mk">
    <body>
      <Providers>{children}</Providers>
    </body>
  </html>
);

export default RootLayout;
