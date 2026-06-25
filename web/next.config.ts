import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // Lint is a separate gate (`npm run lint`, eslint-config-imperium), matching the
  // other finki-hub frontends whose build is type-check only. Type errors still fail the build.
  eslint: {
    ignoreDuringBuilds: true,
  },
  // Pin the workspace root to web/ so an unrelated lockfile in a parent
  // directory does not get inferred as the file-tracing root.
  outputFileTracingRoot: import.meta.dirname,
  reactStrictMode: true,
};

export default nextConfig;
