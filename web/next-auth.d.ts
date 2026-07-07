import type { DefaultSession } from 'next-auth';

declare module 'next-auth' {
  // eslint-disable-next-line @typescript-eslint/consistent-type-definitions -- NextAuth module augmentation requires interface merging.
  interface Session {
    user?: DefaultSession['user'] & {
      googleSubject?: string;
    };
  }
}

declare module 'next-auth/jwt' {
  // eslint-disable-next-line @typescript-eslint/consistent-type-definitions -- NextAuth JWT module augmentation requires interface merging.
  interface JWT {
    googleSubject?: string;
  }
}
