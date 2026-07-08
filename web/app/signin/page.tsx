import { ArrowRight, Bot, GraduationCap, ShieldCheck } from 'lucide-react';
import { AuthError } from 'next-auth';
import { redirect } from 'next/navigation';

import { auth, isAuthConfigured, providerMap, signIn } from '@/auth';
import { getSafeCallbackUrl } from '@/lib/callback-url';

type SignInPageProps = {
  readonly searchParams: Promise<{
    readonly callbackUrl?: string;
    readonly error?: string;
  }>;
};

const featureItems = [
  'Историјата е врзана со твојата најава, не со уредот.',
  'Разговорите се чуваат безбедно за да можеш да продолжиш подоцна.',
  'ФИНКИ Хаб асистентот останува фокусиран на студиски прашања.',
] as const;

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
      <div className="mx-auto grid min-h-dvh w-full max-w-6xl gap-10 px-6 py-10 md:grid-cols-[1.05fr_0.95fr] md:items-center lg:px-8">
        <section className="space-y-8">
          <div className="inline-flex items-center gap-2 rounded-full border border-border bg-card/80 px-3 py-1.5 text-sm text-muted-foreground shadow-sm backdrop-blur">
            <GraduationCap
              aria-hidden="true"
              className="h-4 w-4 text-primary"
            />
            ФИНКИ Хаб / Чат
          </div>

          <div className="max-w-2xl space-y-5">
            <h1 className="text-balance text-4xl font-bold tracking-tight sm:text-5xl lg:text-6xl">
              Најави се за да разговараш со{' '}
              <span className="whitespace-nowrap">ФИНКИ Хаб</span> асистентот.
            </h1>
            <p className="max-w-xl text-pretty text-lg leading-8 text-muted-foreground">
              Чатот е достапен само за најавени корисници за да ја зачува
              твојата историја и да спречи неовластена употреба.
            </p>
          </div>

          <div className="grid gap-3 text-sm text-muted-foreground">
            {featureItems.map((item) => (
              <div
                className="flex items-start gap-3"
                key={item}
              >
                <ShieldCheck
                  aria-hidden="true"
                  className="mt-0.5 h-4 w-4 shrink-0 text-primary"
                />
                <span>{item}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-3xl border border-border bg-card/86 p-6 shadow-2xl shadow-primary/10 backdrop-blur md:p-8">
          <div className="mb-8 flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-primary text-primary-foreground shadow-lg shadow-primary/25">
              <Bot
                aria-hidden="true"
                className="h-6 w-6"
              />
            </div>
            <div>
              <h2 className="text-xl font-semibold">Продолжи со најава</h2>
              <p className="text-sm text-muted-foreground">
                Избери еден од достапните провајдери.
              </p>
            </div>
          </div>

          {error === undefined ? null : (
            <p className="mb-4 rounded-xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
              Најавата не успеа. Обиди се повторно.
            </p>
          )}

          {providerMap.length === 0 ? (
            <p className="rounded-xl border border-border bg-muted/60 px-4 py-3 text-sm text-muted-foreground">
              Најавата не е конфигурирана на овој сервер.
            </p>
          ) : (
            <div className="space-y-3">
              {providerMap.map((provider) => (
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
                    className="group flex w-full items-center justify-between rounded-2xl border border-border bg-background px-4 py-3 text-left text-sm font-medium transition hover:border-primary/60 hover:bg-primary/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    type="submit"
                  >
                    <span>Најави се со {provider.name}</span>
                    <ArrowRight
                      aria-hidden="true"
                      className="h-4 w-4 text-muted-foreground transition group-hover:translate-x-0.5 group-hover:text-primary"
                    />
                  </button>
                </form>
              ))}
            </div>
          )}
        </section>
      </div>
    </main>
  );
};

export default SignInPage;
