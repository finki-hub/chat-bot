# Web development

See the [repository README](../README.md) for setup and environment variables.
Use Node 26 and `npm ci` to install the locked dependencies.

## TypeScript compiler and tooling

`npm run check` and `npm run typecheck` explicitly run TypeScript 7.0.2 through
the `@typescript/native` npm alias. The explicit path avoids depending on which
package owns the shared `tsc` executable after installation.

The regular `typescript` dependency intentionally remains at 6.0.3, the latest
6.x version: TypeScript 7 does not provide the legacy JavaScript compiler API
required by Next.js tooling and typescript-eslint (whose peer range is currently
`>=4.8.4 <6.1.0`). This follows the upstream recommendation to install the native
compiler alongside TypeScript 6 rather than disabling lint checks or forcing an
unsupported peer version. Remove this compatibility dependency only when those
tools support the new API.

References:

- [TypeScript native compiler](https://github.com/microsoft/typescript-go#readme)
- [TypeScript 7 alias resolution tests](https://github.com/microsoft/typescript-go/blob/main/_extension/test/tsdkPackage.test.ts)
- [typescript-eslint dependency requirements](https://typescript-eslint.io/users/dependency-versions/)
- [Imperium 5 breaking changes](https://github.com/Delemangi/eslint-config-imperium/releases/tag/v5.0.0)

Auth.js intentionally stays on `next-auth@5.0.0-beta.32`: the registry's `latest`
tag is the older 4.x API, not an upgrade from the project's v5 prerelease.
