# FINKI Hub Chat Design System

## 1. Purpose

The chat UI is an operational product surface: fast, quiet, and readable for students asking factual questions. New controls should preserve the existing sidebar rhythm rather than introduce a new visual language.

## 2. Tokens

- Color uses the existing Tailwind semantic tokens: `background`, `foreground`, `muted`, `muted-foreground`, `border`, `card`, `primary`, `destructive`, and `ring`.
- Spacing follows the existing 4px Tailwind scale. Sidebar row controls use `gap-1`, `px-2.5`, `py-1.5`, and icon padding `p-1`. Auth cards use `p-6` on mobile and `p-8` on wider screens.
- Radius follows existing rounded surfaces: `rounded-md` for icon controls, `rounded-lg` for sidebar rows, `rounded-xl` for larger sidebar buttons, `rounded-2xl` for auth CTAs, and `rounded-3xl` for the auth panel.

## 3. Typography

- Sidebar rows use `text-sm`; section labels use `text-xs`, uppercase, and tracking-wide.
- Conversation titles are single-line, truncated, and left-aligned.
- Auth copy leads with the student task and concrete study value, not authentication mechanics or server terminology. Keep the auth headline balanced at `text-4xl`, `sm:text-5xl`, and `lg:text-5xl`.

## 4. Motion

- Motion is limited to explicitly named color, shadow, opacity, grid-row, and transform transitions. Broad `transition-all` is not used.
- Loading indicators may rotate, but only to communicate an in-progress action.

## 5. Components

- Sidebar conversation row: selectable title button plus compact row-action icon buttons that reveal on hover/focus for desktop and stay visible on mobile.
- Row action button: Lucide icon, `size-3.5`, semantic aria label, focus-visible ring, muted default color, stronger hover color matching the action intent.
- Destructive row action: use destructive hover color only for delete.
- Auth sign-in panel: card surface using `card`, `border`, `background`, `primary`, and `muted-foreground` tokens, with provider buttons that visibly change border/background on hover and use `focus-visible:ring-ring`.
- Auth sign-in content: limit the supporting value list to 2 concise benefits so the primary action remains visible on small screens. The panel uses `Започни разговор`, and provider actions follow `Продолжи со {provider.name}`.
- Credential settings dialog: modal card using existing `Dialog`, `Input`, and `Button` primitives. Provider rows use `border`, `card`, `muted`, and `muted-foreground` tokens, with saved state indicated by text and destructive delete action only when a credential exists.
- Model selector option: group models only by provider in catalog order. Models without saved provider credentials remain visible but disabled, use `muted-foreground`, and pair a Lucide key icon with visible key-required text rather than relying on color or hover help.
- Scrollable select menu: keep both scroll-arrow rows mounted while overflow exists so reaching a boundary never moves the popup. Show the unavailable direction with `muted-foreground` and without pointer behavior; hide both rows when the menu does not overflow.
- Mobile sidebar: a modal Radix dialog that traps focus, closes on Escape or overlay interaction, and restores focus to the header trigger. Desktop retains the static complementary landmark.
- Conversation loading state: preserve the current list/thread during transient failures and present an inline alert with a retry action. Only a confirmed missing conversation clears the selection.
- Composer submission failure: retain the draft and show an inline retryable error; clear the draft only after the message is accepted.

## 6. Accessibility

- Every icon-only action must have an `aria-label` from `web/lib/i18n.ts`.
- Primary mobile controls and conversation row actions expose at least 44px touch targets.
- Mobile header and drawer honor safe-area insets; decorative logo and GitHub shortcut are hidden below `sm` to protect the title and primary actions.
- Modal drawers contain keyboard focus and restore it to their invoking control.
- Disabled async actions must expose `aria-busy` when work is pending.
- Auth failures must use `role="alert"` and include a direct recovery step. Missing provider configuration must be described as temporary unavailability without exposing server terminology.

## 7. Accepted debt

- The app currently has no richer brand reference document. This file captures the existing implicit system so small sidebar actions can remain consistent without a redesign.
- Attachments, sharing/export, message search, edit/branching, command palette, font replacement, and Streamdown bundle splitting remain deferred product/performance work.
