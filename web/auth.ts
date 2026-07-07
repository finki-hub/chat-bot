import NextAuth from 'next-auth';
import Google from 'next-auth/providers/google';

const requiredAuthEnv = (name: string): string => {
  const value = process.env[name];

  if (value === undefined || value.length === 0) {
    throw new Error(`Missing required auth env var ${name}`);
  }

  return value;
};

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
      clientId: requiredAuthEnv('AUTH_GOOGLE_ID'),
      clientSecret: requiredAuthEnv('AUTH_GOOGLE_SECRET'),
    }),
  ],
  secret: requiredAuthEnv('AUTH_SECRET'),
  session: { maxAge: 30 * 24 * 60 * 60, strategy: 'jwt' },
  trustHost: true,
});
