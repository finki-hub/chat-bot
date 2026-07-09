import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { authMock, isPlaywrightAuthBypassEnabledMock, redirectMock } =
  vi.hoisted(() => ({
    authMock: vi.fn<() => Promise<unknown>>(),
    isPlaywrightAuthBypassEnabledMock: vi.fn<() => boolean>(() => false),
    redirectMock: vi.fn<(url: string) => never>((url) => {
      throw new Error(`redirect:${url}`);
    }),
  }));

vi.mock('@/auth', () => ({
  auth: authMock,
  isAuthConfigured: () => true,
  isPlaywrightAuthBypassEnabled: isPlaywrightAuthBypassEnabledMock,
}));

vi.mock('@/components/chat/chat-screen', () => ({
  ChatScreen: () => <div>chat-screen</div>,
}));

vi.mock('next/navigation', () => ({
  redirect: redirectMock,
}));

describe('HomePage auth gate', () => {
  beforeEach(() => {
    authMock.mockReset();
    isPlaywrightAuthBypassEnabledMock.mockReset();
    isPlaywrightAuthBypassEnabledMock.mockReturnValue(false);
    redirectMock.mockClear();
  });

  afterEach(() => {
    cleanup();
  });

  it('redirects unauthenticated users to the custom sign-in page', async () => {
    authMock.mockResolvedValueOnce(null);
    const { default: HomePage } = await import('@/app/page');

    await expect(HomePage()).rejects.toThrow('redirect:/signin?callbackUrl=/');
    expect(redirectMock).toHaveBeenCalledWith('/signin?callbackUrl=/');
  });

  it('renders the chat screen for authenticated users', async () => {
    authMock.mockResolvedValueOnce({ user: { email: 'student@example.com' } });
    const { default: HomePage } = await import('@/app/page');
    const result = await HomePage();

    render(result);

    expect(screen.getByText('chat-screen')).toBeVisible();
  });

  it('renders the chat screen when the Playwright auth bypass is enabled', async () => {
    isPlaywrightAuthBypassEnabledMock.mockReturnValue(true);
    const { default: HomePage } = await import('@/app/page');
    const result = await HomePage();

    render(result);

    expect(screen.getByText('chat-screen')).toBeVisible();
    expect(authMock).not.toHaveBeenCalled();
  });
});
