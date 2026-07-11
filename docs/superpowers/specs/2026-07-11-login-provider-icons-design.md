# Login Provider Icons Design

## Goal

Make the configured Google and Microsoft login methods easier to recognize while preserving the existing sign-in layout and authentication behavior.

## Design

- Keep the existing auth card, provider-button spacing, semantic colors, hover treatment, focus ring, and trailing arrow.
- Add a leading Google or Microsoft brand icon from `react-icons/si` beside each provider label. Icons are decorative and use `aria-hidden="true"`.
- Rename the Microsoft provider display name from `Microsoft Entra ID` to `Microsoft`.
- Keep the Auth.js provider ID `microsoft-entra-id`, environment variable names, issuer configuration, callbacks, and persisted provider claims unchanged.
- Render no locally authored SVG assets.

## Provider Scope

Do not add another OIDC provider. Repository documentation and identity types name only Google and Microsoft, and no additional institutional issuer or user demand is documented. A future campus OIDC provider should be added only when its issuer, client registration, and intended audience are known.

## Verification

- Unit-test the provider map display names.
- Run type checking, linting, unit tests, and the production build.
- Render `/signin` with both providers configured and inspect the page at mobile, tablet, and desktop widths.
- Verify both provider controls retain keyboard focus visibility and submit through their unchanged provider IDs.
