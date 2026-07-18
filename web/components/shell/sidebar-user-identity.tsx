'use client';

import {
  ChevronsUpDown,
  CircleUserRound,
  KeyRound,
  LogOut,
} from 'lucide-react';
import { signOut, useSession } from 'next-auth/react';

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { t } from '@/lib/i18n';

type SidebarUserIdentityProps = {
  readonly onOpenCredentials: () => void;
};

export const SidebarUserIdentity = ({
  onOpenCredentials,
}: SidebarUserIdentityProps) => {
  const { data: session, status } = useSession();

  if (status !== 'authenticated') {
    return null;
  }
  const username = session.user?.name ?? null;
  const userEmail = session.user?.email ?? null;
  const userLabel = username ?? userEmail ?? t('sidebar.accountFallback');
  const accountMenuLabel =
    userEmail && userEmail !== userLabel
      ? `${t('sidebar.accountMenu')}: ${userLabel}, ${userEmail}`
      : `${t('sidebar.accountMenu')}: ${userLabel}`;

  return (
    <div
      className="border-t border-border/60 pt-2"
      data-testid="sidebar-user-identity"
    >
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            aria-label={accountMenuLabel}
            className="group flex min-h-11 w-full min-w-0 items-center gap-2 rounded-xl px-2 py-2 text-left transition-colors duration-200 hover:bg-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            type="button"
          >
            <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-background text-muted-foreground ring-1 ring-border/70 transition-colors duration-200 group-hover:text-foreground">
              <CircleUserRound
                aria-hidden="true"
                className="size-4"
              />
            </span>
            <span className="flex min-w-0 flex-1 flex-col">
              <span
                className="truncate text-sm font-medium text-foreground"
                title={userLabel}
              >
                {userLabel}
              </span>
              {username && userEmail ? (
                <span
                  className="truncate text-xs text-muted-foreground"
                  title={userEmail}
                >
                  {userEmail}
                </span>
              ) : null}
            </span>
            <ChevronsUpDown
              aria-hidden="true"
              className="size-4 shrink-0 text-muted-foreground"
            />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align="start"
          className="w-(--radix-dropdown-menu-trigger-width) min-w-56"
          collisionPadding={12}
          side="top"
          sideOffset={8}
        >
          <DropdownMenuGroup>
            <DropdownMenuItem
              className="min-h-11"
              onSelect={onOpenCredentials}
            >
              <KeyRound aria-hidden="true" />
              {t('header.credentials')}
            </DropdownMenuItem>
          </DropdownMenuGroup>
          <DropdownMenuSeparator />
          <DropdownMenuGroup>
            <DropdownMenuItem
              className="min-h-11"
              onSelect={() => {
                void signOut();
              }}
              variant="destructive"
            >
              <LogOut aria-hidden="true" />
              {t('auth.signOut')}
            </DropdownMenuItem>
          </DropdownMenuGroup>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
};
