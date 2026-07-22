import { afterEach, describe, expect, it, vi } from 'vitest';

describe('next config', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.resetModules();
  });

  it('uses the build deployment ID', async () => {
    // Given
    vi.stubEnv('NEXT_DEPLOYMENT_ID', 'deployment-commit');

    // When
    const { default: nextConfig } = await import('../next.config');

    // Then
    expect(nextConfig.deploymentId).toBe('deployment-commit');
  });

  it('leaves deployment version detection disabled for blank IDs', async () => {
    // Given
    vi.stubEnv('NEXT_DEPLOYMENT_ID', ' '.repeat(3));

    // When
    const { default: nextConfig } = await import('../next.config');

    // Then
    expect(nextConfig.deploymentId).toBeUndefined();
  });
});
