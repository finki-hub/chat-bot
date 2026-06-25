import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  output: 'standalone',
  outputFileTracingRoot: import.meta.dirname,
  reactStrictMode: true,
};

export default nextConfig;
