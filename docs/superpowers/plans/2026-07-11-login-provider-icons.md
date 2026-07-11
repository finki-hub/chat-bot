# Login Provider Icons Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add package-rendered Google and Microsoft icons to the sign-in methods and present Microsoft under its consumer-facing name without changing OIDC behavior.

**Architecture:** Keep `web/auth.ts` as the source of provider IDs and display names. Render provider-specific icons directly in the existing server-rendered sign-in button so no client boundary or reusable abstraction is introduced for two fixed providers.

**Tech Stack:** Next.js 16, React 19, Auth.js 5 beta, Tailwind CSS 4, `react-icons`, Vitest, Playwright

## Global Constraints

- Keep the provider ID `microsoft-entra-id` and all `AUTH_MICROSOFT_ENTRA_ID_*` configuration unchanged.
- Use package-provided icons only; add no local SVG assets.
- Preserve the existing sign-in card, provider-button hover/focus states, and redirect flow.
- Add no third OIDC provider without a documented issuer and user need.

---

### Task 1: Provider Display Name

**Files:**
- Modify: `web/test/auth-route.test.ts:134`
- Modify: `web/auth.ts:53`

**Interfaces:**
- Consumes: `providerMap: readonly { id: string; name: string }[]`
- Produces: Microsoft entry `{ id: 'microsoft-entra-id', name: 'Microsoft' }`

- [ ] **Step 1: Change the provider-map expectation to the consumer-facing name**

```typescript
expect(providerMap).toStrictEqual([
  { id: 'google', name: 'Google' },
  { id: 'microsoft-entra-id', name: 'Microsoft' },
]);
```

- [ ] **Step 2: Run the targeted test and confirm it fails because the old display name remains**

Run: `npm test -- test/auth-route.test.ts`
Expected: FAIL showing `Microsoft Entra ID` instead of `Microsoft`.

- [ ] **Step 3: Rename only the Microsoft display name**

```typescript
{
  id: 'microsoft-entra-id',
  name: 'Microsoft',
}
```

- [ ] **Step 4: Run the targeted test**

Run: `npm test -- test/auth-route.test.ts`
Expected: PASS.

### Task 2: Provider Brand Icons

**Files:**
- Modify: `web/package.json`
- Modify: `web/package-lock.json`
- Modify: `web/app/signin/page.tsx:1`

**Interfaces:**
- Consumes: provider IDs `google` and `microsoft-entra-id`
- Produces: decorative `SiGoogle` and `SiMicrosoft` React icon elements within the existing submit buttons

- [ ] **Step 1: Install the approved icon dependency**

Run: `npm install react-icons`
Expected: `react-icons` appears in dependencies and the lockfile updates.

- [ ] **Step 2: Import the two packaged brand icons**

```typescript
import { SiGoogle, SiMicrosoft } from 'react-icons/si';
```

- [ ] **Step 3: Add the icon beside the provider label without changing submission behavior**

```tsx
<span className="flex items-center gap-3">
  {provider.id === 'google' ? (
    <SiGoogle aria-hidden="true" className="h-4 w-4 shrink-0" />
  ) : (
    <SiMicrosoft aria-hidden="true" className="h-4 w-4 shrink-0" />
  )}
  <span>Најави се со {provider.name}</span>
</span>
```

- [ ] **Step 4: Run static and unit validation**

Run: `npm run check && npm run lint && npm test`
Expected: all commands exit 0.

- [ ] **Step 5: Build the production application**

Run: `npm run build`
Expected: Next.js production build exits 0.

### Task 3: Browser and OIDC Surface Verification

**Files:**
- Verify: `web/app/signin/page.tsx`
- Verify: `web/auth.ts`

**Interfaces:**
- Consumes: configured Google and Microsoft environment values
- Produces: rendered `/signin` page with two correctly labeled submit controls

- [ ] **Step 1: Start the app with non-secret test provider values**

Run: set `AUTH_SECRET`, Google credentials, Microsoft credentials, and Microsoft issuer to local test values, then run `npm run dev`.
Expected: `/signin` renders both configured methods without contacting either provider until submission.

- [ ] **Step 2: Inspect mobile, tablet, and desktop widths in Playwright**

Expected at 375, 768, and 1280 pixels: both icons render, text is unclipped, the Microsoft label is `Microsoft`, and the card layout is unchanged.

- [ ] **Step 3: Exercise focus and submission surfaces**

Expected: keyboard focus remains visible; Google submits provider ID `google`; Microsoft submits provider ID `microsoft-entra-id`; both requests reach the Auth.js authorization route even though placeholder credentials cannot complete upstream OAuth.

- [ ] **Step 4: Review, commit, push, and open the PR**

Run repository status/diff/log checks, commit only the design, plan, dependency, UI, and test files, push `feat/login-provider-icons`, and create a PR targeting `main`.
Expected: clean worktree and a GitHub PR URL.
