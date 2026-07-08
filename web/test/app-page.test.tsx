import { isValidElement } from 'react';
import { describe, expect, it, vi } from 'vitest';

const { authMock, redirectMock } = vi.hoisted(() => ({
  authMock: vi.fn<() => Promise<unknown>>(),
  redirectMock: vi.fn<(url: string) => never>((url) => {
    throw new Error(`redirect:${url}`);
  }),
}));

vi.mock('@/auth', () => ({
  auth: authMock,
  isAuthConfigured: () => true,
}));

vi.mock('@/components/chat/chat-screen', () => ({
  ChatScreen: () => 'chat-screen',
}));

vi.mock('next/navigation', () => ({
  redirect: redirectMock,
}));

describe('HomePage auth gate', () => {
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

    expect(isValidElement(result)).toBe(true);
  });
});
