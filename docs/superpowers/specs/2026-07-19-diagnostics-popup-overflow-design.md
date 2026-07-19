# Diagnostics popup overflow design

## Problem

The message diagnostics hover card constrains its width but not its height. On short or narrow screens, a diagnostics payload with every section can extend beyond the viewport while the surrounding chat shell intentionally prevents page-level overflow, leaving some rows unreachable.

## Approved behavior

- Preserve the existing Radix hover-card interaction on pointer hover and keyboard focus.
- Keep the popup within a 16px dynamic-viewport gutter on every side.
- Cap the popup width and height to the available viewport.
- Let the preferred minimum width collapse to the available width on ultra-narrow screens.
- Make the popup content the only vertical scroll owner when its content exceeds the height cap.
- Contain overscroll so reaching the popup boundary does not scroll the conversation behind it.
- Let a focused trigger scroll the popup with Arrow, Page Up/Down, Home, and End keys.
- Preserve the existing desktop width, visual tokens, section order, close delays, and Escape dismissal.

## Implementation boundary

Apply the responsive bounds to the diagnostics hover-card consumer rather than changing the shared hover-card primitive. Other hover cards may have different content and interaction needs, so a global overflow policy would be broader than this defect requires.

## Verification

- Add an end-to-end regression that renders the complete diagnostics payload at short and ultra-narrow viewports, opens the popup through its existing trigger, and proves the popup remains within the viewport while real keyboard input can scroll to the final row.
- Run type checking, linting, unit tests, the targeted browser test, and the production build.
- Manually exercise and capture the popup at 375px, 768px, and 1280px widths, including scrolling and Escape dismissal.
