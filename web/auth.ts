import NextAuth from 'next-auth';
import Google from 'next-auth/providers/google';

const authEnv = (name: string): string => process.env[name] ?? '';

export const isAuthConfigured = (): boolean =>
  authEnv('AUTH_GOOGLE_ID').length > 0 &&
  authEnv('AUTH_GOOGLE_SECRET').length > 0 &&
  authEnv('AUTH_SECRET').length > 0;

export const { auth, handlers, signIn, signOut } = NextAuth({
  callbacks: {
    jwt({ account, profile, token }) {
      if (account?.provider === 'google') {
        token['googleSubject'] = account.providerAccountId;
      }
      if (typeof profile?.sub === 'string') {
        token['googleSubject'] = profile.sub;
      }
      return token;
    },
    session({ session, token }) {
      if (typeof token['googleSubject'] === 'string') {
        session.user.googleSubject = token['googleSubject'];
      }
      return session;
    },
  },
  providers: [
    Google({
      clientId: authEnv('AUTH_GOOGLE_ID'),
      clientSecret: authEnv('AUTH_GOOGLE_SECRET'),
    }),
  ],
  secret: authEnv('AUTH_SECRET'),
  session: { maxAge: 30 * 24 * 60 * 60, strategy: 'jwt' },
  trustHost: true,
});
