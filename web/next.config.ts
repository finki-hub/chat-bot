import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // Lint is a separate gate (`npm run lint`); the build is type-check only, but type errors still fail it.
  eslint: {
    ignoreDuringBuilds: true,
  },
  // Self-contained server bundle (.next/standalone) for the Docker runtime image.
  output: 'standalone',
  // Pin the workspace root to web/ so an unrelated lockfile in a parent
  // directory does not get inferred as the file-tracing root.
  outputFileTracingRoot: import.meta.dirname,
  reactStrictMode: true,
};

export default nextConfig;
