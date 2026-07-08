import NextAuth from 'next-auth';
import Google from 'next-auth/providers/google';
import MicrosoftEntraID from 'next-auth/providers/microsoft-entra-id';

const authEnv = (name: string): string => process.env[name] ?? '';

const isGoogleConfigured = (): boolean =>
  authEnv('AUTH_GOOGLE_ID').length > 0 &&
  authEnv('AUTH_GOOGLE_SECRET').length > 0;

const isMicrosoftConfigured = (): boolean =>
  authEnv('AUTH_MICROSOFT_ENTRA_ID_ID').length > 0 &&
  authEnv('AUTH_MICROSOFT_ENTRA_ID_SECRET').length > 0 &&
  authEnv('AUTH_MICROSOFT_ENTRA_ID_ISSUER').length > 0;

export const isAuthConfigured = (): boolean =>
  (isGoogleConfigured() || isMicrosoftConfigured()) &&
  authEnv('AUTH_SECRET').length > 0;

const authProviders = [
  ...(isGoogleConfigured()
    ? [
        Google({
          clientId: authEnv('AUTH_GOOGLE_ID'),
          clientSecret: authEnv('AUTH_GOOGLE_SECRET'),
        }),
      ]
    : []),
  ...(isMicrosoftConfigured()
    ? [
        MicrosoftEntraID({
          clientId: authEnv('AUTH_MICROSOFT_ENTRA_ID_ID'),
          clientSecret: authEnv('AUTH_MICROSOFT_ENTRA_ID_SECRET'),
          issuer: authEnv('AUTH_MICROSOFT_ENTRA_ID_ISSUER'),
        }),
      ]
    : []),
];

export const { auth, handlers, signIn, signOut } = NextAuth({
  callbacks: {
    jwt({ account, token }) {
      if (typeof account?.provider === 'string') {
        token['provider'] = account.provider;
        token['providerSubject'] = account.providerAccountId;
      }
      return token;
    },
    session({ session, token }) {
      if (
        typeof token['provider'] === 'string' &&
        typeof token['providerSubject'] === 'string'
      ) {
        session.user.provider = token['provider'];
        session.user.providerSubject = token['providerSubject'];
      }
      return session;
    },
  },
  providers: authProviders,
  secret: authEnv('AUTH_SECRET'),
  session: { maxAge: 30 * 24 * 60 * 60, strategy: 'jwt' },
  trustHost: true,
});
