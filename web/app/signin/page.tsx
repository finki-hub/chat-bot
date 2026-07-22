import type { IconType } from 'react-icons';

import { ArrowRight } from 'lucide-react';
import { AuthError } from 'next-auth';
import { redirect } from 'next/navigation';
import { BsGoogle, BsMicrosoft } from 'react-icons/bs';

import {
  auth,
  type AuthProviderId,
  isAuthConfigured,
  providerMap,
  signIn,
} from '@/auth';
import { ChatShowcase } from '@/components/auth/chat-showcase';
import { getSafeCallbackUrl } from '@/lib/callback-url';

type SignInPageProps = {
  readonly searchParams: Promise<{
    readonly callbackUrl?: string;
    readonly error?: string;
  }>;
};

const providerIcons = {
  google: BsGoogle,
  'microsoft-entra-id': BsMicrosoft,
} satisfies Record<AuthProviderId, IconType>;

const SignInPage = async ({ searchParams }: SignInPageProps) => {
  const [{ callbackUrl, error }, session] = await Promise.all([
    searchParams,
    isAuthConfigured() ? auth() : null,
  ]);
  const safeCallbackUrl = getSafeCallbackUrl(callbackUrl);

  if (session !== null) {
    redirect(safeCallbackUrl);
  }

  return (
    <main
      className="bg-background text-foreground"
      id="main-content"
      tabIndex={-1}
    >
      <div className="mx-auto flex w-full max-w-6xl px-6 py-10 lg:min-h-dvh lg:items-center lg:px-8">
        <div className="grid w-full gap-10 lg:grid-cols-[0.95fr_1.05fr] lg:gap-16">
          <section className="space-y-8 motion-safe:animate-in motion-safe:fade-in-0 motion-safe:slide-in-from-bottom-2 motion-safe:duration-500 motion-safe:fill-mode-both">
            <div className="flex items-center gap-3">
              <img
                alt=""
                aria-hidden="true"
                className="h-10 w-10 object-contain"
                height={40}
                src="/logo.png"
                width={40}
              />
              <span className="text-lg font-bold tracking-tight">
                ФИНКИ Хаб
              </span>
            </div>

            <div className="max-w-xl space-y-4">
              <h1 className="text-balance text-4xl font-bold tracking-tight sm:text-5xl">
                Имаш прашање за ФИНКИ?
              </h1>
              <p className="text-pretty text-base leading-7 text-muted-foreground sm:text-lg sm:leading-8">
                Од запишување до дипломирање — прашај било што, било кога.
              </p>
            </div>

            <div className="max-w-sm space-y-3">
              {error === undefined ? null : (
                <p
                  className="rounded-xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive"
                  role="alert"
                >
                  Не успеавме да те најавиме. Обиди се повторно.
                </p>
              )}

              {providerMap.length === 0 ? (
                <p className="rounded-xl border border-border bg-muted/60 px-4 py-3 text-sm text-muted-foreground">
                  Најавувањето моментално не е достапно. Обиди се повторно
                  подоцна.
                </p>
              ) : (
                providerMap.map((provider) => {
                  const ProviderIcon = providerIcons[provider.id];

                  return (
                    <form
                      action={async () => {
                        'use server';

                        try {
                          await signIn(provider.id, {
                            redirectTo: safeCallbackUrl,
                          });
                        } catch (signInError) {
                          if (signInError instanceof AuthError) {
                            redirect('/signin?error=OAuthSignin');
                          }
                          throw signInError;
                        }
                      }}
                      key={provider.id}
                    >
                      <button
                        className="group flex w-full items-center justify-between rounded-2xl border border-border bg-card px-4 py-3 text-left text-sm font-medium transition-[border-color,background-color,transform] hover:border-primary/60 hover:bg-primary/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring active:translate-y-px motion-reduce:transform-none motion-reduce:transition-none"
                        type="submit"
                      >
                        <span className="flex items-center gap-3">
                          <ProviderIcon
                            aria-hidden="true"
                            className="h-4 w-4 shrink-0"
                          />
                          <span>Продолжи со {provider.name}</span>
                        </span>
                        <ArrowRight
                          aria-hidden="true"
                          className="h-4 w-4 text-muted-foreground group-hover:text-primary motion-safe:transition-[color,transform] motion-safe:group-hover:translate-x-0.5"
                        />
                      </button>
                    </form>
                  );
                })
              )}
            </div>
          </section>

          <section className="relative motion-safe:animate-in motion-safe:fade-in-0 motion-safe:slide-in-from-bottom-2 motion-safe:duration-500 motion-safe:fill-mode-both stagger-1 lg:self-stretch">
            <div className="h-[26rem] lg:absolute lg:inset-0 lg:h-full">
              <ChatShowcase />
            </div>
          </section>
        </div>
      </div>
    </main>
  );
};

export default SignInPage;
