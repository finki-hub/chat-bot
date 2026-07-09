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

## 4. Motion

- Motion is limited to opacity, color, and transform transitions already present in the sidebar.
- Loading indicators may rotate, but only to communicate an in-progress action.

## 5. Components

- Sidebar conversation row: selectable title button plus compact row-action icon buttons that reveal on hover/focus for desktop and stay visible on mobile.
- Row action button: Lucide icon, `size-3.5`, semantic aria label, focus-visible ring, muted default color, stronger hover color matching the action intent.
- Destructive row action: use destructive hover color only for delete.
- Auth sign-in panel: card surface using `card`, `border`, `background`, `primary`, and `muted-foreground` tokens, with provider buttons that visibly change border/background on hover and use `focus-visible:ring-ring`.
- Credential settings dialog: modal card using existing `Dialog`, `Input`, and `Button` primitives. Provider rows use `border`, `card`, `muted`, and `muted-foreground` tokens, with saved state indicated by text and destructive delete action only when a credential exists.

## 6. Accessibility

- Every icon-only action must have an `aria-label` from `web/lib/i18n.ts`.
- Disabled async actions must expose `aria-busy` when work is pending.

## 7. Accepted debt

- The app currently has no richer brand reference document. This file captures the existing implicit system so small sidebar actions can remain consistent without a redesign.
