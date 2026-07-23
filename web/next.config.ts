import type { NextConfig } from 'next';

const configuredDeploymentId = process.env['NEXT_DEPLOYMENT_ID']?.trim();
const deploymentId =
  configuredDeploymentId === '' ? undefined : configuredDeploymentId;

const nextConfig: NextConfig = {
  deploymentId,
  output: 'standalone',
  outputFileTracingRoot: import.meta.dirname,
  reactStrictMode: true,
};

export default nextConfig;
