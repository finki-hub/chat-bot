import type { IconType } from 'react-icons';

import { ArrowRight, Bot, CircleCheck, GraduationCap } from 'lucide-react';
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
import { getSafeCallbackUrl } from '@/lib/callback-url';

type SignInPageProps = {
  readonly searchParams: Promise<{
    readonly callbackUrl?: string;
    readonly error?: string;
  }>;
};

const featureItems = [
  'Студии, правила\u{A0}и\u{A0}постапки на\u{A0}едно\u{A0}место.',
  'Разговорите ти се достапни на\u{A0}сите\u{A0}уреди.',
] as const;

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
    <main className="relative min-h-dvh overflow-hidden bg-background text-foreground">
      <div className="absolute inset-0 -z-10 bg-[radial-gradient(circle_at_top_left,hsl(var(--primary)/0.24),transparent_34rem),linear-gradient(135deg,hsl(var(--background)),hsl(var(--muted)/0.72))]" />
      <div className="mx-auto grid min-h-dvh w-full max-w-6xl gap-8 px-6 py-8 md:grid-cols-[1.05fr_0.95fr] md:items-center md:gap-10 md:py-10 lg:px-8">
        <section className="space-y-6 md:space-y-8">
          <div className="inline-flex items-center gap-2 rounded-full border border-border bg-card/80 px-3 py-1.5 text-sm text-muted-foreground shadow-sm backdrop-blur">
            <GraduationCap
              aria-hidden="true"
              className="h-4 w-4 text-primary"
            />
            ФИНКИ Хаб асистент
          </div>

          <div className="max-w-2xl space-y-5">
            <h1 className="text-balance text-4xl font-bold tracking-tight sm:text-5xl lg:text-5xl">
              Одговори за ФИНКИ, кога ти требаат.
            </h1>
            <p className="max-w-xl text-pretty text-base leading-7 text-muted-foreground sm:text-lg sm:leading-8">
              Прашај за предмети,{' '}
              <span className="whitespace-nowrap">правила и постапки</span>.
              Продолжи ги разговорите од кој било уред.
            </p>
          </div>

          <div className="grid gap-3 text-sm text-muted-foreground">
            {featureItems.map((item) => (
              <div
                className="flex items-start gap-3"
                key={item}
              >
                <CircleCheck
                  aria-hidden="true"
                  className="mt-0.5 h-4 w-4 shrink-0 text-primary"
                />
                <span>{item}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-3xl border border-border bg-card/86 p-6 shadow-2xl shadow-primary/10 backdrop-blur md:p-8">
          <div className="mb-6 flex items-center gap-3 md:mb-8">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-primary text-primary-foreground shadow-lg shadow-primary/25">
              <Bot
                aria-hidden="true"
                className="h-6 w-6"
              />
            </div>
            <div>
              <h2 className="text-xl font-semibold">Започни разговор</h2>
              <p className="text-sm text-muted-foreground">
                Избери начин на најава за да продолжиш.
              </p>
            </div>
          </div>

          {error === undefined ? null : (
            <p
              className="mb-4 rounded-xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive"
              role="alert"
            >
              Не успеавме да те најавиме. Обиди се повторно.
            </p>
          )}

          {providerMap.length === 0 ? (
            <p className="rounded-xl border border-border bg-muted/60 px-4 py-3 text-sm text-muted-foreground">
              Најавувањето моментално не е достапно. Обиди се повторно подоцна.
            </p>
          ) : (
            <div className="space-y-3">
              {providerMap.map((provider) => {
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
                      className="group flex w-full items-center justify-between rounded-2xl border border-border bg-background px-4 py-3 text-left text-sm font-medium transition-[border-color,background-color,transform] hover:border-primary/60 hover:bg-primary/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring active:translate-y-px motion-reduce:transform-none motion-reduce:transition-none"
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
                        className="h-4 w-4 text-muted-foreground transition-[color,transform] group-hover:translate-x-0.5 group-hover:text-primary"
                      />
                    </button>
                  </form>
                );
              })}
            </div>
          )}
        </section>
      </div>
    </main>
  );
};

export default SignInPage;
