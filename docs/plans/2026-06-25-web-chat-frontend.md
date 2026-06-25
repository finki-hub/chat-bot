# FINKI Hub Web Chat — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pin the shared contract (env access, API/UI-message types, and a verified AI SDK v5 + AI Elements cheat-sheet) that every task author copies from to build the web/ Next.js + BFF chat app against the protocol-v2 SSE backend.

**Architecture:** Browser runs a Next.js (App Router, React 19) client using useChat<MyUIMessage> + DefaultChatTransport, talking only to same-origin /api/* Route Handlers (the BFF). The BFF holds API_BASE_URL and the x-api-key, fetches the Python chat API at POST /chat/ (trailing slash), parses the protocol-v2 named SSE events (token/status/reset/error/done), and re-emits them through a manual createUIMessageStream writer (text-start/text-delta/text-end for the answer, transient data-status/data-error parts, and a start chunk carrying responseId/inferenceModel metadata). Dexie/IndexedDB owns multi-conversation history client-side in UIMessage shape.

**Tech Stack:** Next.js (App Router, React 19, TypeScript); AI SDK v5 (ai + @ai-sdk/react) for createUIMessageStream/createUIMessageStreamResponse and useChat; AI Elements (shadcn registry, vendored into components/ai-elements) + Streamdown + Tailwind v4 + lucide-react; Dexie (IndexedDB) for local history; TanStack Query (models), Zustand (UI state); Vitest + React Testing Library + fake-indexeddb + Playwright; npm.

**Design spec:** `docs/superpowers/specs/2026-06-25-web-chat-frontend-design.md`

## Global Constraints

- AI SDK pinned: `ai@^5` with `@ai-sdk/react@^2` — the `@ai-sdk/react` major is decoupled from `ai`; the release that pairs with `ai@5` is the `2.x` line (there is NO `@ai-sdk/react@5` on npm). AI Elements vendored at the matching v5 release and version-locked to the AI SDK. NOTE: the ecosystem's latest major is now v6 (`ai@6` / `@ai-sdk/react@3`), so install AI Elements + shadcn pinned to v5-compatible versions, not blindly `@latest`.
- Next.js App Router + React 19 + TypeScript; package manager is npm.
- Python API has NO `/api` prefix; the streaming endpoint is `POST /chat/` WITH a trailing slash (a bare `/chat` 307-redirects and drops the body).
- Browser calls ONLY same-origin Next.js `/api/*` routes; it never calls the Python API and never sees API_BASE_URL.
- `x-api-key` (the master key) is server-only via CHAT_API_KEY; the browser hits the same-origin BFF route `POST /api/feedback`, which injects the key and forwards to the Python upstream `POST /chat/feedback` (NOT `/api/feedback` — the Python API has no `/api` prefix). It must never reach the browser and there are NO NEXT_PUBLIC_* vars in v1.
- Request caps the client enforces and the BFF re-validates: <= 50 messages total, <= 8000 chars per turn; messages oldest-first and the last element MUST be role:'user' (else API 422).
- responseId comes ONLY from the `X-Response-Id` response header on POST /chat/; when absent, hide like/dislike for that turn.
- Chrome/UI copy is Macedonian-only (structured so EN can be added later); answers are Macedonian Cyrillic.
- Answer rendering: bare URLs only (autolinked), NO Markdown tables and no `[text](url)` links. Streamdown renders GFM tables/links by DEFAULT, so "defaults suffice" only because the bot's output already avoids them — to HARD-enforce the no-table/no-link constraint, configure Streamdown's allowed components/remark plugins explicitly and verify against the installed version.
- Feedback always sends client:'web' and a non-empty user_id (stable per-browser anon UUID from localStorage); upsert key is (response_id, client, user_id).
- data parts are named EXACTLY `data-status` and `data-error`; both are written `transient:true` (delivered via onData, never persisted into message.parts).
- MyUIMessage metadata is exactly { responseId?: string; inferenceModel?: string } and is attached via a manual `start` chunk's `messageMetadata`.
- BFF parser is tolerant of the legacy plain-text stream: a bare `data:` line with no `event:` is treated as a token (un-escape literal \n to newline).
- Conventional Commits for all commits.

## API Cheat-Sheet (shared reference for all tasks)

## AI SDK v5 + AI Elements + Dexie — verified cheat-sheet (copy from here)

Verified against AI SDK v5 docs (`/websites/ai-sdk_dev_v5`), AI Elements (`/vercel/ai-elements`), Dexie (`/websites/dexie`). Installed packages: `ai@^5`, `@ai-sdk/react@^2` (the line that pairs with `ai@5`; there is no `@ai-sdk/react@5`), AI Elements vendored at the matching v5 release.

### 1) BFF — manual UI message stream WITHOUT streamText

Key facts: text part chunks are `text-start` / `text-delta` / `text-end` and ALL share one `id`; the delta field is named **`delta`** (not `text`). Message metadata on a manual stream is a **`start` chunk** carrying `messageMetadata` (the `messageMetadata` *callback* only exists on `toUIMessageStreamResponse`, which we are NOT using). Transient data parts use `transient: true` and arrive on the client via `onData` only.

```ts
// web/app/api/chat/route.ts  (Node runtime — needs server env)
import { createUIMessageStream, createUIMessageStreamResponse } from 'ai';
import type { MyUIMessage } from '@/lib/api-types';

export async function POST(req: Request) {
  // ...build ChatSchema from UIMessage[] (see api-types), then:
  const upstream = await fetch(`${API_BASE_URL}/chat/`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(chatBody),
  });

  // Pre-stream JSON errors (422/503/500) are NOT SSE — branch before reading.
  if (!upstream.ok || !upstream.headers.get('content-type')?.includes('text/event-stream')) {
    const detail = await upstream.json().catch(() => ({}));
    const stream = createUIMessageStream<MyUIMessage>({
      execute: ({ writer }) => {
        writer.write({ type: 'start' }); // open the assistant message
        writer.write({
          type: 'data-error',
          data: { code: 'pre_stream', message: (detail as any)?.detail ?? 'Request failed' },
          transient: true,
        });
      },
    });
    return createUIMessageStreamResponse({ stream });
  }

  const responseId = upstream.headers.get('X-Response-Id') ?? undefined;
  const inferenceModel = chatBody.inference_model;

  const stream = createUIMessageStream<MyUIMessage>({
    execute: async ({ writer }) => {
      // Attach message metadata via the START chunk (manual-stream way to set metadata):
      writer.write({ type: 'start', messageMetadata: { responseId, inferenceModel } });

      let textId: string | null = null;
      const startText = () => { textId = crypto.randomUUID(); writer.write({ type: 'text-start', id: textId }); };
      const endText = () => { if (textId) { writer.write({ type: 'text-end', id: textId }); textId = null; } };

      // for each parsed protocol-v2 event (see lib/sse.ts):
      //   token {text}  -> if (!textId) startText(); writer.write({ type:'text-delta', id: textId!, delta: text });
      //   status{label,tool} -> writer.write({ type:'data-status', data:{ label, tool }, transient:true });
      //   reset{}       -> endText(); startText();   // preamble drop: end old part, open a new one
      //   error{code,message} -> writer.write({ type:'data-error', data:{ code, message }, transient:true });
      //                          if (code !== 'interrupted') endText();   // hard-stop the text part
      //   done{}        -> endText();                // finalize; execute() returning closes the stream
    },
    onError: (e) => (e instanceof Error ? e.message : 'stream error'),
  });

  return createUIMessageStreamResponse({ stream });
}
```

### 2) Client — typed useChat + transport + onData

`prepareSendMessagesRequest({ id, messages, trigger, messageId })` returns `{ body, headers? }`. Extra request fields go in `body`. Transient data parts surface only in `onData`.

```ts
// web/lib/transport.ts
import { DefaultChatTransport } from 'ai';
import type { MyUIMessage } from '@/lib/api-types';

export function buildChatTransport(getExtras: () => { model: string; embeddingsModel?: string; queryTransformModel?: string; temperature?: number; topP?: number; maxTokens?: number }) {
  return new DefaultChatTransport<MyUIMessage>({
    api: '/api/chat',
    prepareSendMessagesRequest: ({ messages, id, trigger, messageId }) => ({
      body: { messages, id, trigger, messageId, ...getExtras() },
    }),
  });
}
```

```tsx
// in the chat component
import { useChat } from '@ai-sdk/react';
import type { MyUIMessage } from '@/lib/api-types';

const { messages, sendMessage, status, stop, regenerate } = useChat<MyUIMessage>({
  transport: buildChatTransport(() => ({ model: activeModel, temperature: 0.3 })),
  onData: (part) => {
    if (part.type === 'data-status') setStatusChip(part.data);          // {label, tool?}
    if (part.type === 'data-error')  setInlineError(part.data);         // {code, message}
  },
  onFinish: ({ message }) => { /* persist to Dexie */ },
});
// metadata: message.metadata?.responseId / message.metadata?.inferenceModel
```

### 3) AI Elements — import paths + render the LAST text part + status chip

Components are vendored under `components/ai-elements/*` and imported via the `@/` alias. `MessageResponse` wraps Streamdown for streaming Markdown. `PromptInputSubmit` takes a `status` prop ('ready' | 'streaming' | 'submitted' | 'error').

```tsx
import { Conversation, ConversationContent, ConversationEmptyState, ConversationScrollButton } from '@/components/ai-elements/conversation';
import { Message, MessageContent, MessageResponse } from '@/components/ai-elements/message';
import { PromptInput, type PromptInputMessage, PromptInputTextarea, PromptInputSubmit } from '@/components/ai-elements/prompt-input';

{messages.map((m) => {
  const textParts = m.parts.filter((p) => p.type === 'text');
  const last = textParts.at(-1); // render-last = preamble drop (§5.2)
  return (
    <Message from={m.role} key={m.id}>
      <MessageContent>
        {last ? <MessageResponse>{last.text}</MessageResponse> : null}
        {/* custom chip fed by onData data-status, e.g. <SearchStatus label tool /> */}
      </MessageContent>
    </Message>
  );
})}

<PromptInput onSubmit={(msg: PromptInputMessage) => msg.text?.trim() && sendMessage({ text: msg.text })}>
  <PromptInputTextarea />
  <PromptInputSubmit status={status} onClick={() => status === 'streaming' && stop()} />
</PromptInput>
```

### 4) Dexie schema + query + fake-indexeddb under Vitest

```ts
// web/lib/db.ts
import { Dexie, type EntityTable } from 'dexie';
import type { MyUIMessage } from '@/lib/api-types';

export interface ConversationRow { id: string; title: string; model: string; createdAt: number; updatedAt: number; }
export interface MessageRow { id: string; conversationId: string; role: MyUIMessage['role']; parts: MyUIMessage['parts']; metadata?: MyUIMessage['metadata']; createdAt: number; }

export const db = new Dexie('finkiHubChat') as Dexie & {
  conversations: EntityTable<ConversationRow, 'id'>;
  messages: EntityTable<MessageRow, 'id'>;
};
db.version(1).stores({
  conversations: 'id, updatedAt',                 // primary key + index
  messages: 'id, conversationId, createdAt',      // query by conversation, ordered by time
});

export const loadMessages = (cid: string) =>
  db.messages.where('conversationId').equals(cid).sortBy('createdAt');
// saveMessages(conversationId, messages: MyUIMessage[]) maps MyUIMessage[] -> MessageRow[]
// and bumps the conversation's updatedAt in a txn — see Task 8 for the authoritative impl.
```

```ts
// web/vitest.setup.ts  — polyfills global indexedDB + IDBKeyRange for jsdom
import 'fake-indexeddb/auto';
// vitest.config.ts: test: { environment: 'jsdom', setupFiles: ['./vitest.setup.ts'] }
```

```ts
// example test — each test gets a clean DB
import { beforeEach, expect, it } from 'vitest';
import { db } from '@/lib/db';
beforeEach(async () => { await db.delete(); await db.open(); });
it('round-trips a message', async () => {
  await db.messages.bulkPut([{ id: 'm1', conversationId: 'c1', role: 'assistant', parts: [{ type: 'text', text: 'здраво' }], createdAt: 1 }]);
  expect((await db.messages.where('conversationId').equals('c1').toArray())[0].parts[0]).toEqual({ type: 'text', text: 'здраво' });
});
```

---
I have the full spec. Now I'll write Tasks 1, 2, and 3 with complete code and TDD steps.

### Task 1: Scaffold the web/ Next.js app

**Files:**
- Create: `web/package.json`
- Create: `web/tsconfig.json`
- Create: `web/next.config.ts`
- Create: `web/postcss.config.mjs`
- Create: `web/app/globals.css` (Tailwind v4 entry)
- Create: `web/vitest.config.ts`
- Create: `web/vitest.setup.ts`
- Create: `web/playwright.config.ts`
- Create: `web/app/layout.tsx`
- Create: `web/app/page.tsx` (placeholder)
- Create: `web/.gitignore`
- Create: `web/next-env.d.ts` (generated by Next on `dev`/`build`; gitignored per Next convention and regenerated — NOT committed. If `tsc` runs on a clean checkout before a build, run `npx next typegen` first — see Step 15.)
- Test: `web/test/smoke.test.ts`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: a working npm app rooted at `web/` with the `@/*` path alias (`@/* -> ./*`), Tailwind v4, and the test toolchain. Later tasks rely on: `npm` scripts `dev`/`build`/`typecheck`/`test`/`test:e2e`; Vitest config with `environment: 'jsdom'`, `setupFiles: ['./vitest.setup.ts']`, and the `@` alias resolving to the `web/` root; `fake-indexeddb/auto` loaded in setup (used by Tasks 8/13); `app/globals.css` importing Tailwind.

**Steps:**

- [ ] **Step 1: Create the `web/` directory and the npm manifest.** Create `web/package.json`. This pins AI SDK v5, React 19, Next.js, Tailwind v4, Dexie, TanStack Query, Zustand, and the test toolchain so every later task installs against the same versions.

```json
{
  "name": "finki-hub-web",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "typecheck": "tsc --noEmit",
    "test": "vitest run",
    "test:watch": "vitest",
    "test:e2e": "playwright test"
  },
  "dependencies": {
    "@ai-sdk/react": "^2.0.0",
    "@tanstack/react-query": "^5.59.0",
    "ai": "^5.0.0",
    "dexie": "^4.0.10",
    "lucide-react": "^0.460.0",
    "next": "^15.1.0",
    "react": "^19.0.0",
    "react-dom": "^19.0.0",
    "server-only": "^0.0.1",
    "streamdown": "^1.0.0",
    "zustand": "^5.0.2"
  },
  "devDependencies": {
    "@playwright/test": "^1.49.0",
    "@tailwindcss/postcss": "^4.0.0",
    "@testing-library/jest-dom": "^6.6.3",
    "@testing-library/react": "^16.1.0",
    "@testing-library/user-event": "^14.5.2",
    "@types/node": "^22.10.0",
    "@types/react": "^19.0.0",
    "@types/react-dom": "^19.0.0",
    "@vitejs/plugin-react": "^4.3.4",
    "fake-indexeddb": "^6.0.0",
    "jsdom": "^25.0.1",
    "tailwindcss": "^4.0.0",
    "typescript": "^5.7.2",
    "vitest": "^2.1.8"
  }
}
```

- [ ] **Step 2: Add the TypeScript config (strict + `@/` alias).** Create `web/tsconfig.json`. `paths` maps `@/*` to the `web/` root, which every later task imports through.

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["dom", "dom.iterable", "ES2022"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "verbatimModuleSyntax": true,
    "plugins": [{ "name": "next" }],
    "types": ["node", "vitest/globals", "@testing-library/jest-dom"],
    "paths": {
      "@/*": ["./*"]
    }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

- [ ] **Step 3: Add the Next.js config.** Create `web/next.config.ts`.

```ts
import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  reactStrictMode: true,
};

export default nextConfig;
```

- [ ] **Step 4: Add the Tailwind v4 PostCSS config.** Create `web/postcss.config.mjs`. Tailwind v4 ships its PostCSS plugin as `@tailwindcss/postcss`; no separate `tailwind.config` file is required for v4 — configuration is CSS-first via `@import "tailwindcss"`.

```js
const config = {
  plugins: {
    '@tailwindcss/postcss': {},
  },
};

export default config;
```

- [ ] **Step 5: Add the Tailwind v4 CSS entry.** Create `web/app/globals.css`. The single `@import` line is the whole Tailwind v4 setup; later UI tasks add tokens here.

```css
@import 'tailwindcss';
```

- [ ] **Step 6: Add the Vitest config (jsdom + `@` alias + setup).** Create `web/vitest.config.ts`. This wires the React plugin, jsdom, the `@` alias resolving to the `web/` root, and the setup file Tasks 8/13 rely on for IndexedDB.

```ts
import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./', import.meta.url)),
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./vitest.setup.ts'],
    include: ['test/**/*.test.{ts,tsx}'],
  },
});
```

- [ ] **Step 7: Add the Vitest setup file.** Create `web/vitest.setup.ts`. Loads jest-dom matchers and the `fake-indexeddb` global polyfill (used by Dexie tests in Tasks 8/13).

```ts
import '@testing-library/jest-dom/vitest';
import 'fake-indexeddb/auto';
```

- [ ] **Step 8: Add the Playwright config.** Create `web/playwright.config.ts`. Points e2e (Task 16) at the `e2e/` dir and boots `next dev` on port 3000.

```ts
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
```

- [ ] **Step 9: Add the root layout.** Create `web/app/layout.tsx`. Macedonian `lang`, imports `globals.css`, renders children.

```tsx
import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'FINKI Hub Chat',
  description: 'FINKI Hub чат асистент',
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="mk">
      <body>{children}</body>
    </html>
  );
}
```

- [ ] **Step 10: Add the placeholder page.** Create `web/app/page.tsx`. Replaced wholesale by the app shell in Task 13; for now it proves the app renders.

```tsx
export default function Page() {
  return (
    <main>
      <h1>FINKI Hub Chat</h1>
    </main>
  );
}
```

- [ ] **Step 11: Add `web/.gitignore`.** Create `web/.gitignore` so build/test artifacts and local env never get committed.

```gitignore
node_modules
.next
out
coverage
playwright-report
test-results
.env*.local
*.tsbuildinfo
next-env.d.ts
```

- [ ] **Step 12: Install dependencies.** Run from inside `web/`:

```bash
cd web && npm install
```

Expected: a `web/package-lock.json` is created and `node_modules` is populated with no peer-dependency errors that block install.

- [ ] **Step 13: Write the failing smoke test.** Create `web/test/smoke.test.ts`.

```ts
import { describe, expect, it } from 'vitest';

describe('toolchain smoke', () => {
  it('runs vitest with the expected math', () => {
    expect(1 + 1).toBe(2);
  });

  it('has the jsdom environment available', () => {
    expect(typeof document).toBe('object');
    expect(document.createElement('div').tagName).toBe('DIV');
  });

  it('has the fake-indexeddb global polyfill', () => {
    expect(typeof indexedDB).not.toBe('undefined');
  });
});
```

- [ ] **Step 14: Run the smoke test — expect PASS (it is a self-contained sanity check).** Run from inside `web/`:

```bash
cd web && npm test
```

Expected: `smoke.test.ts` passes 3 tests. If `indexedDB is not defined`, the setup file (Step 7) is not being loaded — fix `setupFiles` in `vitest.config.ts`. This single command verifies jsdom, the setup file, and the `fake-indexeddb` polyfill are all wired.

- [ ] **Step 15: Verify the typecheck passes.** Run from inside `web/`:

```bash
cd web && npm run typecheck
```

Expected: `tsc --noEmit` exits 0 with no errors. (If `next-env.d.ts` is missing, run `npm run build` once or `npx next typegen` to generate it, then re-run.)

- [ ] **Step 16: Verify the production build succeeds.** Run from inside `web/`:

```bash
cd web && npm run build
```

Expected: `next build` completes and reports a successful compile with the `/` route. This is the deliverable verification (typecheck + smoke test + build).

- [ ] **Step 17: Commit the scaffold.**

```bash
git add web/
git commit -m "chore(web): scaffold standalone next.js app with vitest and playwright"
```

---

### Task 2: Install shadcn/ui + AI Elements

**Files:**
- Create: `web/components.json`
- Create: `web/lib/utils.ts` (shadcn `cn` helper — required by every shadcn/AI-Elements component)
- Create/Modify: `web/app/globals.css` (shadcn design tokens appended by `shadcn init`)
- Create: `web/components/ui/` (shadcn primitives vendored by the AI Elements registry)
- Create: `web/components/ai-elements/` (vendored AI Elements)
- Test: `web/test/ai-elements.smoke.test.tsx`

**Interfaces:**
- Consumes: Task 1 (the scaffolded app, the `@/` alias, the Vitest jsdom config).
- Produces: vendored components under `@/components/ai-elements/*` and `@/components/ui/*`, plus `@/lib/utils` exporting `cn(...inputs: ClassValue[]): string`. Later tasks (11, 12) import: `@/components/ai-elements/conversation` (`Conversation`, `ConversationContent`, `ConversationEmptyState`, `ConversationScrollButton`), `@/components/ai-elements/message` (`Message`, `MessageContent`, `MessageResponse`), `@/components/ai-elements/prompt-input` (`PromptInput`, `PromptInputTextarea`, `PromptInputSubmit`, `PromptInputSelect`, and the `PromptInputMessage` type).

**Steps:**

- [ ] **Step 1: Initialize shadcn/ui.** Run from inside `web/` (non-interactive flags so it does not prompt). This writes `components.json`, creates `lib/utils.ts` with `cn`, appends the design tokens / base layer to `app/globals.css`, and installs `clsx` + `tailwind-merge`.

```bash
cd web && npx shadcn@latest init --yes
```

NOTE: the shadcn CLI is version-sensitive — run `npx shadcn@latest init --help` FIRST to confirm flags. Current versions have NO `--base-color` flag (the base color `neutral` lives in `components.json`, Step 2); the real flags are `-d/--defaults`, `-b/--base <radix|base>`, `--css-variables`. Run it inside the existing `web/` (Task 1) so it configures the project in place rather than scaffolding a new one.

Expected: `web/components.json` and `web/lib/utils.ts` exist; `app/globals.css` now contains `@layer base` tokens in addition to the Tailwind import.

- [ ] **Step 2: Verify `components.json` uses the `@/` aliases.** Read `web/components.json` and confirm it contains the alias block below (so the registry vendors into the spec's paths). If `tailwind.config` / `css` differ, this is fine for v4; the alias block is what matters.

```json
{
  "$schema": "https://ui.shadcn.com/schema.json",
  "style": "new-york",
  "rsc": true,
  "tsx": true,
  "tailwind": {
    "config": "",
    "css": "app/globals.css",
    "baseColor": "neutral",
    "cssVariables": true
  },
  "aliases": {
    "components": "@/components",
    "utils": "@/lib/utils",
    "ui": "@/components/ui"
  }
}
```

- [ ] **Step 3: Vendor the AI Elements components from their registry.** Run from inside `web/`. This pulls the full AI Elements set into `components/ai-elements/` and any shadcn primitives they depend on (button, textarea, select, …) into `components/ui/`, and installs `streamdown` / `lucide-react` if not already present.

```bash
cd web && npx ai-elements@latest
```

NOTE: bare `ai-elements@latest` (no `add`) installs the full component set; it prompts to scaffold only when there's no `package.json` (Task 1 already created one). Pin to a **v5-compatible** AI Elements release rather than `@latest` — latest now tracks `ai@6` / `@ai-sdk/react@3` and will MISMATCH the pinned `ai@5` + `@ai-sdk/react@2`. Verify the invocation with `npx ai-elements@latest --help`.

Expected: `web/components/ai-elements/` contains at least `conversation.tsx`, `message.tsx`, and `prompt-input.tsx`; `web/components/ui/` contains the shadcn primitives they import.

- [ ] **Step 4: Confirm the required exports exist (no code change, a grep check).** Run from inside `web/`:

```bash
cd web && grep -RE "export (function|const) (Conversation|Message|MessageResponse|PromptInput|PromptInputTextarea|PromptInputSubmit)" components/ai-elements
```

Expected: matches for `Conversation`, `Message`, `MessageResponse`, `PromptInput`, `PromptInputTextarea`, `PromptInputSubmit`. If `MessageResponse` is absent, the installed AI Elements version exposes it under a different name — re-run Step 3 against the pinned release tag and re-check; the cheat-sheet import paths are authoritative.

- [ ] **Step 5: Write the failing render smoke test.** Create `web/test/ai-elements.smoke.test.tsx`. It renders one AI Elements component to prove the vendored source compiles and mounts under jsdom (`Response` is the streaming-Markdown renderer behind `MessageResponse`).

```tsx
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { Message, MessageContent } from '@/components/ai-elements/message';

describe('ai-elements smoke', () => {
  it('renders a Message with text content', () => {
    render(
      <Message from="assistant">
        <MessageContent>здраво свет</MessageContent>
      </Message>,
    );
    expect(screen.getByText('здраво свет')).toBeInTheDocument();
  });
});
```

- [ ] **Step 6: Run the smoke test — expect PASS.** Run from inside `web/`:

```bash
cd web && npm test -- ai-elements.smoke
```

Expected: 1 test passes. If you hit `Cannot find module '@/components/ai-elements/message'`, Step 3 vendored to a different path — confirm the directory and adjust the import; if you hit a missing peer (e.g. `clsx`), run `npm install` and re-run. If `MessageContent` is not exported, render `Message` alone with a child `<span>` and assert on it.

- [ ] **Step 7: Verify the typecheck still passes with the vendored sources.** Run from inside `web/`:

```bash
cd web && npm run typecheck
```

Expected: exit 0. Vendored AI Elements occasionally reference `@/lib/utils` — Step 1 created it, so this should pass; if a vendored file imports a primitive that was not installed, re-run Step 3.

- [ ] **Step 8: Commit shadcn + AI Elements.**

```bash
git add web/components.json web/lib/utils.ts web/app/globals.css web/components/ web/package.json web/package-lock.json
git commit -m "chore(web): vendor shadcn/ui primitives and ai elements"
```

---

### Task 3: Shared env + types

**Files:**
- Create: `web/lib/env.ts`
- Create: `web/lib/api-types.ts`
- Test: `web/test/env.test.ts`

**Interfaces:**
- Consumes: Task 1 (the scaffolded app + `@/` alias + Vitest config) and the `server-only` dependency declared in Task 1's `package.json`.
- Produces (relied on by Tasks 5, 6, 7, 9, 10, 11, 14):
  - From `@/lib/env`: `API_BASE_URL: string`, `CHAT_API_KEY: string`, and `env: { API_BASE_URL: string; CHAT_API_KEY: string }`.
  - From `@/lib/api-types`: types `ModelId`, `ConversationRole`, `ConversationTurn`, `ChatRequestBody`, `ProtocolV2Event`, `ChatErrorCode`, `FeedbackType`, `FeedbackSchema`, `FeedbackAck`, `FeedbackClientPayload`, `MyMetadata`, `MyDataParts`, `MyUIMessage`; and the constants `MAX_MESSAGES = 50`, `MAX_CHARS_PER_TURN = 8000`.

**Steps:**

- [ ] **Step 1: Add the `server-only` test stub so jsdom can import `env.ts`.** The `server-only` package throws if imported into a client bundle; under Vitest there is no bundler boundary, so we alias it to a no-op for tests. Create `web/test/stubs/server-only.ts`:

```ts
// No-op stand-in for the `server-only` package under Vitest (no bundler boundary).
export {};
```

Then add the alias to `web/vitest.config.ts` (extend the existing `resolve.alias` from Task 1):

```ts
import { fileURLToPath } from 'node:url';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./', import.meta.url)),
      'server-only': fileURLToPath(new URL('./test/stubs/server-only.ts', import.meta.url)),
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./vitest.setup.ts'],
    include: ['test/**/*.test.{ts,tsx}'],
  },
});
```

- [ ] **Step 2: Write the failing env test.** Create `web/test/env.test.ts`. It asserts `env.ts` throws when a var is unset and reads it when set. Because `env.ts` reads `process.env` at module-eval time, each case uses `vi.resetModules()` + a dynamic `import()` so the module re-evaluates against the current environment.

```ts
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const ORIGINAL = { ...process.env };

describe('lib/env', () => {
  beforeEach(() => {
    vi.resetModules();
    process.env = { ...ORIGINAL };
  });

  afterEach(() => {
    process.env = { ...ORIGINAL };
  });

  it('throws when API_BASE_URL is missing', async () => {
    delete process.env.API_BASE_URL;
    process.env.CHAT_API_KEY = 'k';
    await expect(import('@/lib/env')).rejects.toThrow(/API_BASE_URL/);
  });

  it('throws when CHAT_API_KEY is missing', async () => {
    process.env.API_BASE_URL = 'http://api:8880';
    delete process.env.CHAT_API_KEY;
    await expect(import('@/lib/env')).rejects.toThrow(/CHAT_API_KEY/);
  });

  it('exposes both values when set', async () => {
    process.env.API_BASE_URL = 'http://api:8880';
    process.env.CHAT_API_KEY = 'secret-key';
    const mod = await import('@/lib/env');
    expect(mod.API_BASE_URL).toBe('http://api:8880');
    expect(mod.CHAT_API_KEY).toBe('secret-key');
    expect(mod.env).toEqual({
      API_BASE_URL: 'http://api:8880',
      CHAT_API_KEY: 'secret-key',
    });
  });
});
```

- [ ] **Step 3: Run the env test — expect FAIL (module does not exist yet).** Run from inside `web/`:

```bash
cd web && npm test -- env
```

Expected: FAIL with `Failed to load url @/lib/env` / `Cannot find module '@/lib/env'` (the file does not exist yet).

- [ ] **Step 4: Create `web/lib/env.ts` with the EXACT contract content.** Create `web/lib/env.ts`:

```ts
// web/lib/env.ts
// Server-only access to BFF env vars. NEVER import this from a Client Component
// (no 'use client'); it must only run in Route Handlers / server code.
// There are no NEXT_PUBLIC_* vars in v1 — the browser only calls same-origin /api/*.
import 'server-only';

function required(name: string): string {
  const value = process.env[name];
  if (!value || value.length === 0) {
    throw new Error(
      `Missing required server env var ${name}. Set it in the BFF environment ` +
        `(e.g. .env.local or the compose service env); it must never be exposed to the browser.`,
    );
  }
  return value;
}

/** Base URL of the Python chat API (protocol-v2), e.g. http://api:8880. No /api prefix; /chat/ has a trailing slash. */
export const API_BASE_URL = required('API_BASE_URL');

/** Master x-api-key for POST /chat/feedback. Server-only — injected by the BFF, never sent to the browser. */
export const CHAT_API_KEY = required('CHAT_API_KEY');

export const env = { API_BASE_URL, CHAT_API_KEY } as const;
```

- [ ] **Step 5: Run the env test — expect PASS.** Run from inside `web/`:

```bash
cd web && npm test -- env
```

Expected: 3 tests pass. If the "throws when …" cases instead resolve, the `server-only` alias from Step 1 is not applied — confirm `vitest.config.ts` has the `server-only` alias and re-run.

- [ ] **Step 6: Write the failing type-level test for `api-types`.** Create `web/test/api-types.test.ts`. It asserts the runtime constants and exercises the exported types so a type regression fails the build (the `satisfies` lines fail typecheck if a type drifts; the runtime `expect` lines fail the test run if the constants drift).

```ts
import { describe, expect, it } from 'vitest';
import {
  MAX_CHARS_PER_TURN,
  MAX_MESSAGES,
  type ChatRequestBody,
  type FeedbackClientPayload,
  type FeedbackSchema,
  type MyUIMessage,
  type ProtocolV2Event,
} from '@/lib/api-types';

describe('lib/api-types', () => {
  it('exposes the wire caps', () => {
    expect(MAX_MESSAGES).toBe(50);
    expect(MAX_CHARS_PER_TURN).toBe(8000);
  });

  it('models a ChatRequestBody (oldest-first, last is user)', () => {
    const body = {
      messages: [
        { role: 'user', content: 'здраво' },
        { role: 'assistant', content: 'здраво!' },
        { role: 'user', content: 'кога е испитот?' },
      ],
      temperature: 0.3,
    } satisfies ChatRequestBody;
    expect(body.messages.at(-1)?.role).toBe('user');
  });

  it('models a protocol-v2 token event', () => {
    const ev = { event: 'token', data: { text: 'збор' } } satisfies ProtocolV2Event;
    expect(ev.data.text).toBe('збор');
  });

  it('models the feedback wire + client payloads', () => {
    const wire = {
      response_id: '00000000-0000-4000-8000-000000000000',
      client: 'web',
      user_id: 'anon-1',
      feedback_type: 'like',
    } satisfies FeedbackSchema;
    const payload = {
      responseId: wire.response_id,
      feedbackType: 'like',
      userId: 'anon-1',
    } satisfies FeedbackClientPayload;
    expect(wire.client).toBe('web');
    expect(payload.feedbackType).toBe('like');
  });

  it('models a typed UIMessage with metadata + data-status part', () => {
    const msg: MyUIMessage = {
      id: 'm1',
      role: 'assistant',
      metadata: { responseId: 'r1', inferenceModel: 'claude-sonnet-4-6' },
      parts: [
        { type: 'text', text: 'одговор' },
        { type: 'data-status', data: { label: '🔍 Пребарувам…', tool: 'search' } },
      ],
    };
    expect(msg.metadata?.responseId).toBe('r1');
    expect(msg.parts[0].type).toBe('text');
  });
});
```

- [ ] **Step 7: Run the api-types test — expect FAIL (module does not exist yet).** Run from inside `web/`:

```bash
cd web && npm test -- api-types
```

Expected: FAIL with `Failed to load url @/lib/api-types` / `Cannot find module '@/lib/api-types'`.

- [ ] **Step 8: Create `web/lib/api-types.ts` with the EXACT contract content.** Create `web/lib/api-types.ts`:

```ts
// web/lib/api-types.ts
// Single source of truth for the wire contract + the typed UIMessage.
// Mirrors the Python API: ChatSchema (api/app/schemas/chat.py) and
// FeedbackSchema (api/app/schemas/feedback.py), per spec §3.2 / §3.4.
import type { UIMessage } from 'ai';

// ---------------------------------------------------------------------------
// Model ids: GET /chat/models returns a flat sorted string[]; no display meta.
// ---------------------------------------------------------------------------
export type ModelId = string;

// ---------------------------------------------------------------------------
// POST /chat/ request body (ChatSchema). messages: 1..50, OLDEST-FIRST, last
// element MUST be role:"user". content <= 8000 chars/turn. snake_case on wire.
// ---------------------------------------------------------------------------
export type ConversationRole = 'user' | 'assistant';

export interface ConversationTurn {
  role: ConversationRole;
  content: string; // <= 8000 chars
}

export interface ChatRequestBody {
  messages: ConversationTurn[]; // 1..50, oldest-first, last is role:"user"
  system_prompt?: string | null; // default null -> server FINKI agent prompt
  embeddings_model?: ModelId; // default BAAI/bge-m3
  inference_model?: ModelId; // default claude-sonnet-4-6 — picks streaming LLM
  query_transform_model?: ModelId; // default gpt-5.4-mini
  temperature?: number; // 0.0..1.0 (default 0.3)
  top_p?: number; // 0.0..1.0 (default 1.0)
  max_tokens?: number; // >= 1 (default 4096)
}

// Client-side caps the BFF re-validates.
export const MAX_MESSAGES = 50;
export const MAX_CHARS_PER_TURN = 8000;

// ---------------------------------------------------------------------------
// protocol-v2 SSE events (named SSE: `event: <name>\ndata: <JSON>\n\n`).
// ---------------------------------------------------------------------------
export type ProtocolV2Event =
  | { event: 'token'; data: { text: string } }
  | { event: 'status'; data: { state: string; label: string; tool?: string } }
  | { event: 'reset'; data: Record<string, never> }
  | { event: 'error'; data: { code: 'no_answer' | 'interrupted' | 'agent_error'; message: string } }
  | { event: 'done'; data: Record<string, never> };

export type ChatErrorCode = 'no_answer' | 'interrupted' | 'agent_error';

// ---------------------------------------------------------------------------
// POST /chat/feedback (FeedbackSchema). client is the literal "web"; user_id
// is required (>=1 char). x-api-key is injected server-side by the BFF.
// ---------------------------------------------------------------------------
export type FeedbackType = 'like' | 'dislike';

export interface FeedbackSchema {
  response_id: string; // UUID from X-Response-Id
  client: 'web'; // required literal for this app
  user_id: string; // required, min length 1 (anon per-browser id)
  feedback_type: FeedbackType;
  question_text?: string;
  answer_text?: string;
  inference_model?: string;
  embeddings_model?: string;
  query_transform_model?: string;
  client_ref?: string;
  channel_id?: string;
  guild_id?: string;
}

export interface FeedbackAck {
  id: string; // UUID — FeedbackAckSchema.id is a UUID, not an int
  response_id: string; // UUID
  feedback_type: FeedbackType;
}

// Client -> BFF feedback payload; the BFF adds client:"web", user_id, x-api-key.
export interface FeedbackClientPayload {
  responseId: string;
  feedbackType: FeedbackType;
  userId: string; // anon per-browser id (BFF maps to user_id)
  questionText?: string;
  answerText?: string;
  inferenceModel?: string;
}

// ---------------------------------------------------------------------------
// Typed UIMessage: metadata + custom data parts. UIMessage<METADATA, DATA, TOOLS>.
// Data parts are named EXACTLY data-status and data-error; both transient.
// ---------------------------------------------------------------------------
export interface MyMetadata {
  responseId?: string;
  inferenceModel?: string;
}

export interface MyDataParts {
  status: { label: string; tool?: string }; // -> part type "data-status"
  error: { code: string; message: string }; // -> part type "data-error"
}

export type MyUIMessage = UIMessage<MyMetadata, MyDataParts>;
```

- [ ] **Step 9: Run the api-types test — expect PASS.** Run from inside `web/`:

```bash
cd web && npm test -- api-types
```

Expected: 5 tests pass. If `data-status` errors at typecheck because the part's `data` is inferred unknown, confirm `MyDataParts.status` is `{ label: string; tool?: string }` and that `MyUIMessage` is `UIMessage<MyMetadata, MyDataParts>` exactly as above.

- [ ] **Step 10: Verify the whole suite + typecheck pass together.** Run from inside `web/`:

```bash
cd web && npm test && npm run typecheck
```

Expected: all tests green (smoke + ai-elements + env + api-types) and `tsc --noEmit` exits 0. The `satisfies`/typed-`MyUIMessage` lines in the tests make a type regression fail `typecheck`, giving the type-level coverage the task requires.

- [ ] **Step 11: Commit the shared env + types.**

```bash
git add web/lib/env.ts web/lib/api-types.ts web/test/env.test.ts web/test/api-types.test.ts web/test/stubs/server-only.ts web/vitest.config.ts
git commit -m "feat(web): add server-only env access and shared api wire types"
```

I have the full spec. Now I have everything I need to write Tasks 4 and 5. Let me produce the markdown for these two tasks, drawing precisely from the contract types, the cheat-sheet, and the spec's protocol-v2 event table.

```markdown
### Task 4: Protocol-v2 SSE parser (`web/lib/sse.ts`)

A pure async-iterator parser that turns the Python chat API's SSE byte/text stream into typed protocol-v2 events. It MUST also tolerate the legacy bare `data: <text>` form (a `data:` line with no preceding `event:` line) by mapping it to a `token` event, un-escaping literal `\n` → newline (spec §10, §3.2). This is consumed by the BFF translator in Task 5.

**Files:**
- Create: `web/lib/sse.ts`
- Test: `web/test/sse.test.ts`

**Interfaces:**
- Consumes (from Task 3, `@/lib/api-types`): `ChatErrorCode = 'no_answer' | 'interrupted' | 'agent_error'`.
- Produces (other tasks import these from `@/lib/sse`):
  - `type ParsedEvent =`
    - `| { type: 'token'; text: string }`
    - `| { type: 'status'; state: string; label: string; tool?: string }`
    - `| { type: 'reset' }`
    - `| { type: 'error'; code: ChatErrorCode; message: string }`
    - `| { type: 'done' }`
  - `type SseSource = AsyncIterable<Uint8Array | string> | ReadableStream<Uint8Array>`
  - `function parseProtocolV2(source: SseSource): AsyncGenerator<ParsedEvent, void, unknown>`

**Steps:**

- [ ] **Step 1: Write the failing test for the happy-path token/done stream.**
  Create `web/test/sse.test.ts` with a helper that yields a fixture as one string chunk, plus the first assertion:

  ```ts
  // web/test/sse.test.ts
  import { describe, expect, it } from 'vitest';
  import { parseProtocolV2, type ParsedEvent } from '@/lib/sse';

  // Wraps fixture chunks as an AsyncIterable<string>, mimicking a decoded stream.
  function source(...chunks: string[]): AsyncIterable<string> {
    return {
      async *[Symbol.asyncIterator]() {
        for (const c of chunks) yield c;
      },
    };
  }

  async function collect(...chunks: string[]): Promise<ParsedEvent[]> {
    const out: ParsedEvent[] = [];
    for await (const ev of parseProtocolV2(source(...chunks))) out.push(ev);
    return out;
  }

  describe('parseProtocolV2', () => {
    it('parses a plain token stream ending in done', async () => {
      const events = await collect(
        'event: token\ndata: {"text":"Здраво"}\n\n',
        'event: token\ndata: {"text":", свете"}\n\n',
        'event: done\ndata: {}\n\n',
      );
      expect(events).toEqual([
        { type: 'token', text: 'Здраво' },
        { type: 'token', text: ', свете' },
        { type: 'done' },
      ]);
    });
  });
  ```

- [ ] **Step 2: Run the test, expect FAIL.**
  Run `cd web && npx vitest run test/sse.test.ts` (or from `web/`: `npx vitest run test/sse.test.ts`).
  Expected failure: `Failed to resolve import "@/lib/sse"` / `parseProtocolV2 is not a function` — the module does not exist yet.

- [ ] **Step 3: Minimal implementation of `parseProtocolV2`.**
  Create `web/lib/sse.ts`. It normalizes the source to a string async-iterable, buffers across chunks, splits on blank lines into SSE frames, and parses each frame's `event:` + `data:` lines. A `data:` line with no `event:` line maps to a `token` (legacy fallback, un-escaping `\n`).

  ```ts
  // web/lib/sse.ts
  // Protocol-v2 SSE parser. Frames are `event: <name>\ndata: <JSON>\n\n`.
  // Tolerant of the legacy bare `data: <text>` form (no event: line) → token,
  // un-escaping literal `\n` to a real newline (spec §10).
  import type { ChatErrorCode } from '@/lib/api-types';

  export type ParsedEvent =
    | { type: 'token'; text: string }
    | { type: 'status'; state: string; label: string; tool?: string }
    | { type: 'reset' }
    | { type: 'error'; code: ChatErrorCode; message: string }
    | { type: 'done' };

  export type SseSource =
    | AsyncIterable<Uint8Array | string>
    | ReadableStream<Uint8Array>;

  const ERROR_CODES: readonly ChatErrorCode[] = ['no_answer', 'interrupted', 'agent_error'];

  function toErrorCode(value: unknown): ChatErrorCode {
    return ERROR_CODES.includes(value as ChatErrorCode) ? (value as ChatErrorCode) : 'agent_error';
  }

  // Normalize any supported source into an async-iterable of decoded strings.
  async function* toStringChunks(source: SseSource): AsyncGenerator<string> {
    const decoder = new TextDecoder();
    if (source instanceof ReadableStream) {
      const reader = source.getReader();
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          if (value) yield decoder.decode(value, { stream: true });
        }
        const tail = decoder.decode();
        if (tail) yield tail;
      } finally {
        reader.releaseLock();
      }
      return;
    }
    for await (const chunk of source as AsyncIterable<Uint8Array | string>) {
      yield typeof chunk === 'string' ? chunk : decoder.decode(chunk, { stream: true });
    }
  }

  // Parse one SSE frame (the text between blank-line separators) into an event.
  function parseFrame(frame: string): ParsedEvent | null {
    let eventName: string | null = null;
    const dataLines: string[] = [];
    for (const rawLine of frame.split('\n')) {
      const line = rawLine.replace(/\r$/, '');
      if (line === '' || line.startsWith(':')) continue; // blank / comment
      if (line.startsWith('event:')) {
        eventName = line.slice('event:'.length).trim();
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice('data:'.length).replace(/^ /, ''));
      }
    }
    if (dataLines.length === 0 && eventName === null) return null;
    const dataRaw = dataLines.join('\n');

    // Legacy fallback: a data-only frame with no event name is a token.
    if (eventName === null) {
      return { type: 'token', text: dataRaw.replace(/\\n/g, '\n') };
    }

    let data: unknown = {};
    if (dataRaw.length > 0) {
      try {
        data = JSON.parse(dataRaw);
      } catch {
        // Named frame with non-JSON data: only meaningful for token text.
        if (eventName === 'token') return { type: 'token', text: dataRaw.replace(/\\n/g, '\n') };
        data = {};
      }
    }
    const obj = (data ?? {}) as Record<string, unknown>;

    switch (eventName) {
      case 'token':
        return { type: 'token', text: typeof obj.text === 'string' ? obj.text : '' };
      case 'status':
        return {
          type: 'status',
          state: typeof obj.state === 'string' ? obj.state : '',
          label: typeof obj.label === 'string' ? obj.label : '',
          ...(typeof obj.tool === 'string' ? { tool: obj.tool } : {}),
        };
      case 'reset':
        return { type: 'reset' };
      case 'error':
        return {
          type: 'error',
          code: toErrorCode(obj.code),
          message: typeof obj.message === 'string' ? obj.message : '',
        };
      case 'done':
        return { type: 'done' };
      default:
        return null; // unknown event name → ignore
    }
  }

  export async function* parseProtocolV2(source: SseSource): AsyncGenerator<ParsedEvent, void, unknown> {
    let buffer = '';
    for await (const chunk of toStringChunks(source)) {
      buffer += chunk;
      // SSE frames are separated by a blank line. Normalize CRLF first.
      buffer = buffer.replace(/\r\n/g, '\n');
      let sepIndex = buffer.indexOf('\n\n');
      while (sepIndex !== -1) {
        const frame = buffer.slice(0, sepIndex);
        buffer = buffer.slice(sepIndex + 2);
        const ev = parseFrame(frame);
        if (ev) yield ev;
        sepIndex = buffer.indexOf('\n\n');
      }
    }
    // Flush a trailing frame with no terminating blank line.
    const tail = buffer.replace(/\r\n/g, '\n').trim();
    if (tail) {
      const ev = parseFrame(tail);
      if (ev) yield ev;
    }
  }
  ```

- [ ] **Step 4: Run the test, expect PASS.**
  Run `npx vitest run test/sse.test.ts` from `web/`. Expected: the `parses a plain token stream ending in done` test passes.

- [ ] **Step 5: Write the failing test for the tool path (status → reset → answer) and chunk-splitting.**
  Append to `web/test/sse.test.ts`. This covers the agent tool sequence AND a frame split across two source chunks (the parser must buffer):

  ```ts
  it('parses the tool path: status, reset, then answer tokens', async () => {
    const events = await collect(
      'event: status\ndata: {"state":"tool_call","label":"🔍 Пребарувам…","tool":"search_docs"}\n\n',
      'event: token\ndata: {"text":"некаков преамбула"}\n\n',
      'event: reset\ndata: {}\n\n',
      'event: token\ndata: {"text":"вистински одговор"}\n\n',
      'event: done\ndata: {}\n\n',
    );
    expect(events).toEqual([
      { type: 'status', state: 'tool_call', label: '🔍 Пребарувам…', tool: 'search_docs' },
      { type: 'token', text: 'некаков преамбула' },
      { type: 'reset' },
      { type: 'token', text: 'вистински одговор' },
      { type: 'done' },
    ]);
  });

  it('buffers a frame split across multiple chunks', async () => {
    const events = await collect('event: token\nda', 'ta: {"text":"спл', 'ит"}\n\n', 'event: done\ndata: {}\n\n');
    expect(events).toEqual([{ type: 'token', text: 'сплит' }, { type: 'done' }]);
  });
  ```

- [ ] **Step 6: Run the test, expect PASS (no new implementation needed).**
  Run `npx vitest run test/sse.test.ts` from `web/`. Expected: both new tests pass — buffering and the tool path are already handled by Step 3. If the split-chunk test fails, the buffer logic in `parseProtocolV2` is the culprit; do not change the tests.

- [ ] **Step 7: Write the failing test for the error event and the legacy bare-`data:` fallback.**
  Append to `web/test/sse.test.ts`:

  ```ts
  it('maps error frames to typed error events and clamps unknown codes', async () => {
    const events = await collect(
      'event: error\ndata: {"code":"interrupted","message":"одговорот е прекинат"}\n\n',
      'event: error\ndata: {"code":"weird","message":"boom"}\n\n',
      'event: done\ndata: {}\n\n',
    );
    expect(events).toEqual([
      { type: 'error', code: 'interrupted', message: 'одговорот е прекинат' },
      { type: 'error', code: 'agent_error', message: 'boom' },
      { type: 'done' },
    ]);
  });

  it('treats a bare data: line (no event:) as a token and un-escapes \\n', async () => {
    const events = await collect('data: прв ред\\nвтор ред\n\n');
    expect(events).toEqual([{ type: 'token', text: 'прв ред\nвтор ред' }]);
  });

  it('ignores unknown named events', async () => {
    const events = await collect('event: ping\ndata: {}\n\n', 'event: done\ndata: {}\n\n');
    expect(events).toEqual([{ type: 'done' }]);
  });
  ```

- [ ] **Step 8: Run the test, expect PASS (no new implementation needed).**
  Run `npx vitest run test/sse.test.ts` from `web/`. Expected: all SSE tests pass — `toErrorCode` clamps `"weird"` → `agent_error`, the eventless `data:` frame becomes a token with `\n` un-escaped, and `ping` is ignored. If any fail, fix `sse.ts` (not the tests).

- [ ] **Step 9: Commit.**
  ```sh
  git add web/lib/sse.ts web/test/sse.test.ts
  git commit -m "feat(web): add protocol-v2 SSE parser with legacy fallback"
  ```

---

### Task 5: BFF `/api/chat` translator (`web/app/api/chat/route.ts`)

The core piece. A **pure** function `translateToUiStream` drains an `AsyncIterable<ParsedEvent>` (from Task 4) and drives a writer-like sink, emitting the AI SDK v5 UI-message-stream parts: lazy `text-start`/`text-delta`/`text-end`, `reset` → end current part + start a fresh one (preamble drop, spec §5.2), `status` → transient `data-status`, `error` → transient `data-error` (hard-stop the text part unless `interrupted`), `done` → finalize. `route.ts` wires `fetch` + `parseProtocolV2` + `createUIMessageStream`/`createUIMessageStreamResponse` around the pure function. The request-body → `ChatSchema` conversion (oldest-first, last-is-user, 50/8000 caps) is a second pure function, `toChatRequestBody`.

**Files:**
- Create: `web/lib/chat-translate.ts`
- Create: `web/app/api/chat/route.ts`
- Test: `web/test/chat-translate.test.ts`
- Test: `web/test/api-chat.route.test.ts`

**Interfaces:**
- Consumes:
  - From Task 4 (`@/lib/sse`): `parseProtocolV2`, `type ParsedEvent`.
  - From Task 3 (`@/lib/api-types`): `type MyUIMessage`, `type ChatRequestBody`, `type ConversationTurn`, `MAX_MESSAGES`, `MAX_CHARS_PER_TURN`.
  - From Task 3 (`@/lib/env`): `API_BASE_URL`.
- Produces (used by tests and `route.ts`):
  - `interface UiStreamWriter { write(part: UiStreamPart): void }` — the writer-like sink shape (a subset of the AI SDK `createUIMessageStream` writer).
  - `type UiStreamPart` — the union of parts this translator emits (`start` / `text-start` / `text-delta` / `text-end` / `data-status` / `data-error`).
  - `function translateToUiStream(events: AsyncIterable<ParsedEvent>, writer: UiStreamWriter, meta: { responseId?: string; inferenceModel?: string }, idGen?: () => string): Promise<void>`
  - `interface ChatClientBody { messages: MyUIMessage[]; model?: string; embeddingsModel?: string; queryTransformModel?: string; temperature?: number; topP?: number; maxTokens?: number; userId?: string }`
  - `function toChatRequestBody(body: ChatClientBody): ChatRequestBody`

**Steps:**

- [ ] **Step 1: Write the failing test for `toChatRequestBody` (message conversion + caps).**
  Create `web/test/chat-translate.test.ts`:

  ```ts
  // web/test/chat-translate.test.ts
  import { describe, expect, it } from 'vitest';
  import type { MyUIMessage } from '@/lib/api-types';
  import { toChatRequestBody, type ChatClientBody } from '@/lib/chat-translate';

  function msg(role: MyUIMessage['role'], ...texts: string[]): MyUIMessage {
    return { id: crypto.randomUUID(), role, parts: texts.map((text) => ({ type: 'text', text })) };
  }

  describe('toChatRequestBody', () => {
    it('joins text parts and forwards sampling params', () => {
      const body: ChatClientBody = {
        messages: [msg('user', 'Кога е ', 'испитот?')],
        model: 'claude-sonnet-4-6',
        embeddingsModel: 'BAAI/bge-m3',
        queryTransformModel: 'gpt-5.4-mini',
        temperature: 0.5,
        topP: 0.9,
        maxTokens: 2048,
      };
      expect(toChatRequestBody(body)).toEqual({
        messages: [{ role: 'user', content: 'Кога е испитот?' }],
        inference_model: 'claude-sonnet-4-6',
        embeddings_model: 'BAAI/bge-m3',
        query_transform_model: 'gpt-5.4-mini',
        temperature: 0.5,
        top_p: 0.9,
        max_tokens: 2048,
      });
    });

    it('keeps only the last 50 messages (oldest-first)', () => {
      const messages = Array.from({ length: 60 }, (_, i) =>
        msg(i % 2 === 0 ? 'user' : 'assistant', `m${i}`),
      );
      messages[59] = msg('user', 'last'); // ensure last is a user turn
      const out = toChatRequestBody({ messages });
      expect(out.messages).toHaveLength(50);
      expect(out.messages[0].content).toBe('m10');
      expect(out.messages.at(-1)).toEqual({ role: 'user', content: 'last' });
    });

    it('truncates each turn to 8000 chars', () => {
      const out = toChatRequestBody({ messages: [msg('user', 'я'.repeat(9000))] });
      expect(out.messages[0].content).toHaveLength(8000);
    });

    it('omits undefined sampling params', () => {
      const out = toChatRequestBody({ messages: [msg('user', 'hi')] });
      expect(out).toEqual({ messages: [{ role: 'user', content: 'hi' }] });
    });
  });
  ```

- [ ] **Step 2: Run the test, expect FAIL.**
  Run `npx vitest run test/chat-translate.test.ts` from `web/`.
  Expected failure: `Failed to resolve import "@/lib/chat-translate"` — the module does not exist.

- [ ] **Step 3: Implement `toChatRequestBody` (and declare the shared types).**
  Create `web/lib/chat-translate.ts` with the types and the body converter. The translator function is added in Step 6.

  ```ts
  // web/lib/chat-translate.ts
  // Pure translation logic for the BFF /api/chat route, unit-testable without
  // Next.js: (1) toChatRequestBody maps the client request → Python ChatSchema;
  // (2) translateToUiStream drains protocol-v2 events into AI SDK UI-message-
  // stream parts (lazy text part, preamble drop on reset, transient data parts).
  import type {
    ChatRequestBody,
    ConversationTurn,
    MyUIMessage,
  } from '@/lib/api-types';
  import { MAX_CHARS_PER_TURN, MAX_MESSAGES } from '@/lib/api-types';
  import type { ParsedEvent } from '@/lib/sse';

  export interface ChatClientBody {
    messages: MyUIMessage[];
    model?: string;
    embeddingsModel?: string;
    queryTransformModel?: string;
    temperature?: number;
    topP?: number;
    maxTokens?: number;
    userId?: string;
  }

  function joinText(message: MyUIMessage): string {
    const text = message.parts
      .filter((p): p is { type: 'text'; text: string } => p.type === 'text')
      .map((p) => p.text)
      .join('');
    return text.length > MAX_CHARS_PER_TURN ? text.slice(0, MAX_CHARS_PER_TURN) : text;
  }

  export function toChatRequestBody(body: ChatClientBody): ChatRequestBody {
    const trimmed = body.messages.slice(-MAX_MESSAGES);
    const messages: ConversationTurn[] = trimmed.map((m) => ({
      role: m.role === 'assistant' ? 'assistant' : 'user',
      content: joinText(m),
    }));

    const out: ChatRequestBody = { messages };
    if (body.model !== undefined) out.inference_model = body.model;
    if (body.embeddingsModel !== undefined) out.embeddings_model = body.embeddingsModel;
    if (body.queryTransformModel !== undefined) out.query_transform_model = body.queryTransformModel;
    if (body.temperature !== undefined) out.temperature = body.temperature;
    if (body.topP !== undefined) out.top_p = body.topP;
    if (body.maxTokens !== undefined) out.max_tokens = body.maxTokens;
    return out;
  }
  ```

- [ ] **Step 4: Run the test, expect PASS.**
  Run `npx vitest run test/chat-translate.test.ts` from `web/`. Expected: all four `toChatRequestBody` tests pass.

- [ ] **Step 5: Write the failing test for `translateToUiStream` (token, tool path, error, done).**
  Append to `web/test/chat-translate.test.ts`. A `FakeWriter` records the emitted parts; a deterministic `idGen` makes `text-start` ids assertable. The test uses an async generator of `ParsedEvent` as input.

  ```ts
  import type { ParsedEvent } from '@/lib/sse';
  import { translateToUiStream, type UiStreamPart } from '@/lib/chat-translate';

  class FakeWriter {
    parts: UiStreamPart[] = [];
    write(part: UiStreamPart) {
      this.parts.push(part);
    }
  }

  async function* events(...evs: ParsedEvent[]): AsyncGenerator<ParsedEvent> {
    for (const e of evs) yield e;
  }

  function ids() {
    let n = 0;
    return () => `t${++n}`;
  }

  describe('translateToUiStream', () => {
    it('emits start metadata, lazy text part, and finalizes on done', async () => {
      const w = new FakeWriter();
      await translateToUiStream(
        events({ type: 'token', text: 'Здраво' }, { type: 'token', text: '!' }, { type: 'done' }),
        w,
        { responseId: 'r1', inferenceModel: 'claude-sonnet-4-6' },
        ids(),
      );
      expect(w.parts).toEqual([
        { type: 'start', messageMetadata: { responseId: 'r1', inferenceModel: 'claude-sonnet-4-6' } },
        { type: 'text-start', id: 't1' },
        { type: 'text-delta', id: 't1', delta: 'Здраво' },
        { type: 'text-delta', id: 't1', delta: '!' },
        { type: 'text-end', id: 't1' },
      ]);
    });

    it('drops the preamble on reset by ending the old part and starting a new one', async () => {
      const w = new FakeWriter();
      await translateToUiStream(
        events(
          { type: 'status', state: 'tool_call', label: '🔍 Пребарувам…', tool: 'search_docs' },
          { type: 'token', text: 'преамбула' },
          { type: 'reset' },
          { type: 'token', text: 'одговор' },
          { type: 'done' },
        ),
        w,
        { responseId: 'r2', inferenceModel: 'claude-sonnet-4-6' },
        ids(),
      );
      expect(w.parts).toEqual([
        { type: 'start', messageMetadata: { responseId: 'r2', inferenceModel: 'claude-sonnet-4-6' } },
        { type: 'data-status', data: { label: '🔍 Пребарувам…', tool: 'search_docs' }, transient: true },
        { type: 'text-start', id: 't1' },
        { type: 'text-delta', id: 't1', delta: 'преамбула' },
        { type: 'text-end', id: 't1' },
        { type: 'text-start', id: 't2' },
        { type: 'text-delta', id: 't2', delta: 'одговор' },
        { type: 'text-end', id: 't2' },
      ]);
    });

    it('emits data-error and hard-stops the text part on a non-interrupted error', async () => {
      const w = new FakeWriter();
      await translateToUiStream(
        events(
          { type: 'token', text: 'half' },
          { type: 'error', code: 'agent_error', message: 'boom' },
          { type: 'done' },
        ),
        w,
        {},
        ids(),
      );
      expect(w.parts).toEqual([
        { type: 'start', messageMetadata: {} },
        { type: 'text-start', id: 't1' },
        { type: 'text-delta', id: 't1', delta: 'half' },
        { type: 'data-error', data: { code: 'agent_error', message: 'boom' }, transient: true },
        { type: 'text-end', id: 't1' },
      ]);
    });

    it('keeps the partial text part open on interrupted (no extra text-end before done)', async () => {
      const w = new FakeWriter();
      await translateToUiStream(
        events(
          { type: 'token', text: 'half' },
          { type: 'error', code: 'interrupted', message: 'прекинат' },
          { type: 'done' },
        ),
        w,
        {},
        ids(),
      );
      expect(w.parts).toEqual([
        { type: 'start', messageMetadata: {} },
        { type: 'text-start', id: 't1' },
        { type: 'text-delta', id: 't1', delta: 'half' },
        { type: 'data-error', data: { code: 'interrupted', message: 'прекинат' }, transient: true },
        { type: 'text-end', id: 't1' },
      ]);
    });

    it('emits only start + data-error when the stream errors before any token', async () => {
      const w = new FakeWriter();
      await translateToUiStream(
        events({ type: 'error', code: 'no_answer', message: 'нема одговор' }, { type: 'done' }),
        w,
        {},
        ids(),
      );
      expect(w.parts).toEqual([
        { type: 'start', messageMetadata: {} },
        { type: 'data-error', data: { code: 'no_answer', message: 'нема одговор' }, transient: true },
      ]);
    });
  });
  ```

- [ ] **Step 6: Implement `translateToUiStream` and its part types.**
  Append to `web/lib/chat-translate.ts`. Key rules from spec §5.1/§5.2: emit `start` with metadata first; `text-start` is lazy (on the first token after a (re)start); `reset` ends the current part and arms a fresh one; `error` writes a transient `data-error` and, unless `interrupted`, ends the current text part (hard stop); `done` ends any open text part. An open text part is always closed by the end (so `interrupted` and `done` both finalize cleanly).

  ```ts
  export type UiStreamPart =
    | { type: 'start'; messageMetadata: { responseId?: string; inferenceModel?: string } }
    | { type: 'text-start'; id: string }
    | { type: 'text-delta'; id: string; delta: string }
    | { type: 'text-end'; id: string }
    | { type: 'data-status'; data: { label: string; tool?: string }; transient: true }
    | { type: 'data-error'; data: { code: string; message: string }; transient: true };

  export interface UiStreamWriter {
    write(part: UiStreamPart): void;
  }

  export async function translateToUiStream(
    events: AsyncIterable<ParsedEvent>,
    writer: UiStreamWriter,
    meta: { responseId?: string; inferenceModel?: string },
    idGen: () => string = () => crypto.randomUUID(),
  ): Promise<void> {
    writer.write({ type: 'start', messageMetadata: meta });

    let textId: string | null = null;
    let stopped = false; // a non-interrupted error halts further text

    const startText = () => {
      textId = idGen();
      writer.write({ type: 'text-start', id: textId });
    };
    const endText = () => {
      if (textId) {
        writer.write({ type: 'text-end', id: textId });
        textId = null;
      }
    };

    for await (const ev of events) {
      switch (ev.type) {
        case 'token': {
          if (stopped) break;
          if (!textId) startText();
          writer.write({ type: 'text-delta', id: textId!, delta: ev.text });
          break;
        }
        case 'status': {
          writer.write({
            type: 'data-status',
            data: { label: ev.label, ...(ev.tool !== undefined ? { tool: ev.tool } : {}) },
            transient: true,
          });
          break;
        }
        case 'reset': {
          // Preamble drop: end the current part, lazily open a new one on the
          // next token (render-last shows only the post-reset answer, §5.2).
          endText();
          break;
        }
        case 'error': {
          writer.write({
            type: 'data-error',
            data: { code: ev.code, message: ev.message },
            transient: true,
          });
          if (ev.code !== 'interrupted') {
            endText(); // hard stop the text part
            stopped = true;
          }
          break;
        }
        case 'done': {
          endText();
          break;
        }
      }
    }

    endText(); // finalize any still-open text part (e.g. interrupted, or no done)
  }
  ```

- [ ] **Step 7: Run the test, expect PASS.**
  Run `npx vitest run test/chat-translate.test.ts` from `web/`. Expected: all `translateToUiStream` tests pass. Note the assertions are tight — `reset` produces a fresh `text-start id: 't2'`, `interrupted` keeps a single `text-start/text-end` pair, and the pre-token `no_answer` case emits no text part at all.

- [ ] **Step 8: Write the failing route test (mock global fetch with a fake Python SSE).**
  Create `web/test/api-chat.route.test.ts`. It mocks `@/lib/env` (so the route imports without real env vars), mocks `global.fetch` to return a `text/event-stream` body with an `X-Response-Id` header, calls the route's `POST`, and reads the resulting UI-message-stream back to assert the translated SSE contains the answer text and the response-id metadata.

  ```ts
  // web/test/api-chat.route.test.ts
  import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

  vi.mock('@/lib/env', () => ({
    API_BASE_URL: 'http://api:8880',
    CHAT_API_KEY: 'test-key',
    env: { API_BASE_URL: 'http://api:8880', CHAT_API_KEY: 'test-key' },
  }));

  function sseBody(...frames: string[]): ReadableStream<Uint8Array> {
    const enc = new TextEncoder();
    return new ReadableStream({
      start(controller) {
        for (const f of frames) controller.enqueue(enc.encode(f));
        controller.close();
      },
    });
  }

  async function readAll(res: Response): Promise<string> {
    return await res.text();
  }

  describe('POST /api/chat', () => {
    beforeEach(() => {
      vi.restoreAllMocks();
    });
    afterEach(() => {
      vi.restoreAllMocks();
    });

    it('translates a python SSE answer into a UI message stream with metadata', async () => {
      const fetchMock = vi.fn(async () =>
        new Response(
          sseBody(
            'event: token\ndata: {"text":"Здраво"}\n\n',
            'event: token\ndata: {"text":"!"}\n\n',
            'event: done\ndata: {}\n\n',
          ),
          { status: 200, headers: { 'content-type': 'text/event-stream', 'X-Response-Id': 'resp-123' } },
        ),
      );
      vi.stubGlobal('fetch', fetchMock);

      const { POST } = await import('@/app/api/chat/route');
      const req = new Request('http://localhost/api/chat', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          messages: [{ id: 'u1', role: 'user', parts: [{ type: 'text', text: 'Здраво' }] }],
          model: 'claude-sonnet-4-6',
        }),
      });

      const res = await POST(req);
      expect(res.headers.get('content-type')).toContain('text/event-stream');

      // Assert the upstream python API was called correctly.
      expect(fetchMock).toHaveBeenCalledTimes(1);
      const [url, init] = fetchMock.mock.calls[0];
      expect(String(url)).toBe('http://api:8880/chat/');
      const sentBody = JSON.parse((init as RequestInit).body as string);
      expect(sentBody.messages).toEqual([{ role: 'user', content: 'Здраво' }]);
      expect(sentBody.inference_model).toBe('claude-sonnet-4-6');

      // The translated UI stream carries the answer text and the response id.
      const out = await readAll(res);
      expect(out).toContain('Здраво');
      expect(out).toContain('resp-123');
      expect(out).toContain('text-delta');
    });

    it('surfaces a pre-stream JSON error (503) as a data-error', async () => {
      const fetchMock = vi.fn(async () =>
        new Response(JSON.stringify({ detail: 'Failed to retrieve or re-rank context for the query.' }), {
          status: 503,
          headers: { 'content-type': 'application/json' },
        }),
      );
      vi.stubGlobal('fetch', fetchMock);

      const { POST } = await import('@/app/api/chat/route');
      const req = new Request('http://localhost/api/chat', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ messages: [{ id: 'u1', role: 'user', parts: [{ type: 'text', text: 'hi' }] }] }),
      });

      const res = await POST(req);
      const out = await readAll(res);
      expect(out).toContain('data-error');
      expect(out).toContain('Failed to retrieve or re-rank context');
    });
  });
  ```

- [ ] **Step 9: Run the route test, expect FAIL.**
  Run `npx vitest run test/api-chat.route.test.ts` from `web/`.
  Expected failure: `Failed to resolve import "@/app/api/chat/route"` — `route.ts` does not exist yet.

- [ ] **Step 10: Implement the route handler.**
  Create `web/app/api/chat/route.ts`. It parses the client body, builds the `ChatSchema`, POSTs to the Python API, branches on content-type (pre-stream JSON error → a `data-error`-only UI stream per the cheat-sheet), reads `X-Response-Id`, and otherwise wires `parseProtocolV2(upstream.body)` into `translateToUiStream` inside `createUIMessageStream`.

  ```ts
  // web/app/api/chat/route.ts
  import { createUIMessageStream, createUIMessageStreamResponse } from 'ai';
  import { API_BASE_URL } from '@/lib/env';
  import type { MyUIMessage } from '@/lib/api-types';
  import {
    toChatRequestBody,
    translateToUiStream,
    type ChatClientBody,
  } from '@/lib/chat-translate';
  import { parseProtocolV2 } from '@/lib/sse';

  // The Python API needs a server runtime (env, streaming fetch), not edge.
  export const runtime = 'nodejs';

  export async function POST(req: Request): Promise<Response> {
    const clientBody = (await req.json()) as ChatClientBody;
    const chatBody = toChatRequestBody(clientBody);
    const inferenceModel = chatBody.inference_model;

    const upstream = await fetch(`${API_BASE_URL}/chat/`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(chatBody),
    });

    const contentType = upstream.headers.get('content-type') ?? '';

    // Pre-stream JSON errors (422/503/500) are NOT SSE — branch before reading.
    if (!upstream.ok || !contentType.includes('text/event-stream')) {
      const detail = (await upstream.json().catch(() => ({}))) as { detail?: string };
      const stream = createUIMessageStream<MyUIMessage>({
        execute: ({ writer }) => {
          writer.write({ type: 'start', messageMetadata: { inferenceModel } });
          writer.write({
            type: 'data-error',
            data: { code: 'pre_stream', message: detail?.detail ?? 'Request failed' },
            transient: true,
          });
        },
      });
      return createUIMessageStreamResponse({ stream });
    }

    const responseId = upstream.headers.get('X-Response-Id') ?? undefined;
    const upstreamBody = upstream.body;

    const stream = createUIMessageStream<MyUIMessage>({
      execute: async ({ writer }) => {
        if (!upstreamBody) {
          writer.write({ type: 'start', messageMetadata: { responseId, inferenceModel } });
          writer.write({
            type: 'data-error',
            data: { code: 'agent_error', message: 'Empty stream from API' },
            transient: true,
          });
          return;
        }
        await translateToUiStream(parseProtocolV2(upstreamBody), writer, {
          responseId,
          inferenceModel,
        });
      },
      onError: (e) => (e instanceof Error ? e.message : 'stream error'),
    });

    return createUIMessageStreamResponse({ stream });
  }
  ```

  Note: `translateToUiStream` and the `createUIMessageStream` `writer` agree on the part shapes used here (`start` with `messageMetadata`, `text-start`/`text-delta`/`text-end`, transient `data-status`/`data-error`); the `writer` accepts these AI SDK v5 UI-message-stream chunks directly, so no adapter is needed.

- [ ] **Step 11: Run the route test, expect PASS.**
  Run `npx vitest run test/api-chat.route.test.ts` from `web/`. Expected: both route tests pass — the happy path forwards the correct `ChatSchema` to `http://api:8880/chat/` and the translated stream contains `Здраво`, `text-delta`, and `resp-123`; the 503 path yields a `data-error` carrying the `detail`.

- [ ] **Step 12: Typecheck the new modules.**
  Run `npx tsc --noEmit` from `web/`. Expected: no type errors. If `translateToUiStream`'s `writer.write` argument is rejected by the AI SDK writer type in `route.ts`, the `UiStreamPart` union in `chat-translate.ts` is out of sync with the installed `ai` types — reconcile the part shapes (do not loosen with `any`).

- [ ] **Step 13: Commit.**
  ```sh
  git add web/lib/chat-translate.ts web/app/api/chat/route.ts web/test/chat-translate.test.ts web/test/api-chat.route.test.ts
  git commit -m "feat(web): add BFF /api/chat protocol-v2 to UI-message-stream translator"
  ```
```

I have enough context from the spec and the detailed contract provided. Tasks 6 and 7 are the BFF route handlers for `/api/models` and `/api/feedback`. Let me write these two tasks following the exact skeleton, contract, and TDD rules.

### Task 6: BFF `/api/models` route — proxy `GET /chat/models`

**Files:**
- Create: `web/app/api/models/route.ts`
- Test: `web/test/api-models.route.test.ts`

**Interfaces:**

Consumes (from Task 3, `web/lib/env.ts`):
- `API_BASE_URL: string`

Consumes (from Task 3, `web/lib/api-types.ts`):
- `type ModelId = string`

Produces (consumed by Task 12 `web/lib/use-models.ts` via `GET /api/models`):
- HTTP `GET /api/models` → `200` JSON body `ModelId[]` (a flat `string[]`), with `Cache-Control: public, max-age=300, stale-while-revalidate=600`.
- On upstream failure (non-2xx or fetch throw) → `200` JSON body `[]` (empty list never breaks the picker; the client falls back to defaults). Header `X-Models-Source: error` is set so tests can assert the fallback path.

**Steps:**

- [ ] **Step 1: Write the failing route test.** Create `web/test/api-models.route.test.ts` with the full test below. It mocks `web/lib/env.ts` (so the route doesn't throw on missing env) and stubs `global.fetch`, then calls the exported `GET` handler directly.

```ts
// web/test/api-models.route.test.ts
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@/lib/env', () => ({
  API_BASE_URL: 'http://api:8880',
  CHAT_API_KEY: 'test-key',
  env: { API_BASE_URL: 'http://api:8880', CHAT_API_KEY: 'test-key' },
}));

import { GET } from '@/app/api/models/route';

const okJson = (body: unknown, init?: ResponseInit) =>
  new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
    ...init,
  });

describe('GET /api/models', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('proxies the upstream model list and returns a string[]', async () => {
    const models = ['claude-sonnet-4-6', 'gpt-5.4-mini', 'BAAI/bge-m3'];
    const fetchMock = vi.fn().mockResolvedValue(okJson(models));
    vi.stubGlobal('fetch', fetchMock);

    const res = await GET();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('http://api:8880/chat/models');
    expect(init?.method ?? 'GET').toBe('GET');

    expect(res.status).toBe(200);
    expect(res.headers.get('cache-control')).toBe(
      'public, max-age=300, stale-while-revalidate=600',
    );
    await expect(res.json()).resolves.toEqual(models);
  });

  it('returns [] with an error source header when upstream is non-2xx', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response('nope', { status: 503 }));
    vi.stubGlobal('fetch', fetchMock);

    const res = await GET();

    expect(res.status).toBe(200);
    expect(res.headers.get('x-models-source')).toBe('error');
    await expect(res.json()).resolves.toEqual([]);
  });

  it('returns [] with an error source header when fetch throws', async () => {
    const fetchMock = vi.fn().mockRejectedValue(new Error('network down'));
    vi.stubGlobal('fetch', fetchMock);

    const res = await GET();

    expect(res.status).toBe(200);
    expect(res.headers.get('x-models-source')).toBe('error');
    await expect(res.json()).resolves.toEqual([]);
  });

  it('returns [] when upstream JSON is not an array of strings', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson({ models: 'oops' }));
    vi.stubGlobal('fetch', fetchMock);

    const res = await GET();

    expect(res.status).toBe(200);
    expect(res.headers.get('x-models-source')).toBe('error');
    await expect(res.json()).resolves.toEqual([]);
  });
});
```

- [ ] **Step 2: Run the test, expect FAIL.** Run:

```bash
cd web && npx vitest run test/api-models.route.test.ts
```

Expected failure: a module-resolution error, e.g. `Failed to resolve import "@/app/api/models/route"` (the route file does not exist yet).

- [ ] **Step 3: Implement the route handler (minimal, makes the test pass).** Create `web/app/api/models/route.ts` with the full code below.

```ts
// web/app/api/models/route.ts
// BFF: proxy GET {API_BASE_URL}/chat/models -> a flat string[] of model ids.
// Server-only (Route Handler): API_BASE_URL never reaches the browser.
// The upstream list changes rarely, so we cache briefly. Any upstream failure
// degrades to [] (the model picker falls back to its defaults) and is marked
// with X-Models-Source: error so callers/tests can detect the fallback path.
import { API_BASE_URL } from '@/lib/env';
import type { ModelId } from '@/lib/api-types';

// Always run on the server; never statically prerender (needs server env).
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const CACHE_CONTROL = 'public, max-age=300, stale-while-revalidate=600';

function isStringArray(value: unknown): value is ModelId[] {
  return Array.isArray(value) && value.every((item) => typeof item === 'string');
}

function fallback(): Response {
  return new Response(JSON.stringify([] as ModelId[]), {
    status: 200,
    headers: {
      'content-type': 'application/json',
      'x-models-source': 'error',
    },
  });
}

export async function GET(): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(`${API_BASE_URL}/chat/models`, {
      method: 'GET',
      headers: { accept: 'application/json' },
    });
  } catch {
    return fallback();
  }

  if (!upstream.ok) {
    return fallback();
  }

  let body: unknown;
  try {
    body = await upstream.json();
  } catch {
    return fallback();
  }

  if (!isStringArray(body)) {
    return fallback();
  }

  return new Response(JSON.stringify(body), {
    status: 200,
    headers: {
      'content-type': 'application/json',
      'cache-control': CACHE_CONTROL,
    },
  });
}
```

- [ ] **Step 4: Run the test, expect PASS.** Run:

```bash
cd web && npx vitest run test/api-models.route.test.ts
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit.**

```bash
git add web/app/api/models/route.ts web/test/api-models.route.test.ts
git commit -m "feat(web): add BFF GET /api/models proxy with caching and graceful fallback"
```

---

### Task 7: BFF `/api/feedback` route — inject `x-api-key` and proxy `POST /chat/feedback`

**Files:**
- Create: `web/app/api/feedback/route.ts`
- Test: `web/test/api-feedback.route.test.ts`

**Interfaces:**

Consumes (from Task 3, `web/lib/env.ts`):
- `API_BASE_URL: string`
- `CHAT_API_KEY: string`

Consumes (from Task 3, `web/lib/api-types.ts`):
- `interface FeedbackClientPayload { responseId: string; feedbackType: FeedbackType; userId: string; questionText?: string; answerText?: string; inferenceModel?: string }`
- `type FeedbackType = 'like' | 'dislike'`
- `interface FeedbackSchema { response_id: string; client: 'web'; user_id: string; feedback_type: FeedbackType; question_text?: string; answer_text?: string; inference_model?: string; ... }`
- `interface FeedbackAck { id: number; response_id: string; feedback_type: FeedbackType }`

Produces (consumed by Task 14 `web/components/chat/answer-actions.tsx` via `POST /api/feedback`):
- HTTP `POST /api/feedback`, request body = `FeedbackClientPayload` (camelCase). The handler builds the snake_case `FeedbackSchema` (adds `client: "web"`, maps `userId`→`user_id`), injects `x-api-key: CHAT_API_KEY` server-side, and forwards to `POST {API_BASE_URL}/chat/feedback`.
- Responses: `200` JSON `FeedbackAck` on success; `400` JSON `{ error }` for invalid client payload (missing `responseId`/`userId`/`feedbackType`); `502` JSON `{ error }` when upstream fails.

**Steps:**

- [ ] **Step 1: Write the failing route test.** Create `web/test/api-feedback.route.test.ts` with the full test below. It mocks env, stubs `global.fetch`, and posts `FeedbackClientPayload` to the exported `POST` handler — asserting the injected `x-api-key` header and the assembled snake_case body.

```ts
// web/test/api-feedback.route.test.ts
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('@/lib/env', () => ({
  API_BASE_URL: 'http://api:8880',
  CHAT_API_KEY: 'super-secret-key',
  env: { API_BASE_URL: 'http://api:8880', CHAT_API_KEY: 'super-secret-key' },
}));

import { POST } from '@/app/api/feedback/route';
import type { FeedbackAck, FeedbackClientPayload } from '@/lib/api-types';

const jsonRequest = (body: unknown) =>
  new Request('http://localhost/api/feedback', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });

const ack: FeedbackAck = { id: 7, response_id: 'r-123', feedback_type: 'like' };

const okJson = (body: unknown) =>
  new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });

const validPayload: FeedbackClientPayload = {
  responseId: 'r-123',
  feedbackType: 'like',
  userId: 'anon-abc',
  questionText: 'Кога е испитот?',
  answerText: 'Испитот е на 1 јуни.',
  inferenceModel: 'claude-sonnet-4-6',
};

describe('POST /api/feedback', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('injects x-api-key and forwards a snake_case FeedbackSchema, returning the ack', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson(ack));
    vi.stubGlobal('fetch', fetchMock);

    const res = await POST(jsonRequest(validPayload));

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe('http://api:8880/chat/feedback');
    expect(init.method).toBe('POST');

    const headers = new Headers(init.headers);
    expect(headers.get('x-api-key')).toBe('super-secret-key');
    expect(headers.get('content-type')).toBe('application/json');

    expect(JSON.parse(init.body as string)).toEqual({
      response_id: 'r-123',
      client: 'web',
      user_id: 'anon-abc',
      feedback_type: 'like',
      question_text: 'Кога е испитот?',
      answer_text: 'Испитот е на 1 јуни.',
      inference_model: 'claude-sonnet-4-6',
    });

    expect(res.status).toBe(200);
    await expect(res.json()).resolves.toEqual(ack);
  });

  it('omits optional fields when not provided', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson(ack));
    vi.stubGlobal('fetch', fetchMock);

    await POST(
      jsonRequest({
        responseId: 'r-123',
        feedbackType: 'dislike',
        userId: 'anon-abc',
      } satisfies FeedbackClientPayload),
    );

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(JSON.parse(init.body as string)).toEqual({
      response_id: 'r-123',
      client: 'web',
      user_id: 'anon-abc',
      feedback_type: 'dislike',
    });
  });

  it('never leaks the api key to the response or to the browser-facing body', async () => {
    const fetchMock = vi.fn().mockResolvedValue(okJson(ack));
    vi.stubGlobal('fetch', fetchMock);

    const res = await POST(jsonRequest(validPayload));
    expect(res.headers.get('x-api-key')).toBeNull();
  });

  it('returns 400 when responseId is missing', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    const res = await POST(
      jsonRequest({ feedbackType: 'like', userId: 'anon-abc' }),
    );

    expect(res.status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('returns 400 when userId is empty', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    const res = await POST(
      jsonRequest({ responseId: 'r-123', feedbackType: 'like', userId: '' }),
    );

    expect(res.status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('returns 400 when feedbackType is not like/dislike', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    const res = await POST(
      jsonRequest({
        responseId: 'r-123',
        feedbackType: 'meh',
        userId: 'anon-abc',
      }),
    );

    expect(res.status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('returns 502 when upstream fails', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response('boom', { status: 500 }));
    vi.stubGlobal('fetch', fetchMock);

    const res = await POST(jsonRequest(validPayload));

    expect(res.status).toBe(502);
    await expect(res.json()).resolves.toHaveProperty('error');
  });
});
```

- [ ] **Step 2: Run the test, expect FAIL.** Run:

```bash
cd web && npx vitest run test/api-feedback.route.test.ts
```

Expected failure: `Failed to resolve import "@/app/api/feedback/route"` (the route file does not exist yet).

- [ ] **Step 3: Implement the route handler (minimal, makes the test pass).** Create `web/app/api/feedback/route.ts` with the full code below.

```ts
// web/app/api/feedback/route.ts
// BFF: receive a camelCase FeedbackClientPayload from the browser, assemble the
// snake_case FeedbackSchema (adding client:"web" and mapping userId->user_id),
// inject the server-only x-api-key, and forward to POST {API_BASE_URL}/chat/feedback.
// The api key lives only in this server process and never reaches the browser.
import { API_BASE_URL, CHAT_API_KEY } from '@/lib/env';
import type {
  FeedbackAck,
  FeedbackClientPayload,
  FeedbackSchema,
  FeedbackType,
} from '@/lib/api-types';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function isFeedbackType(value: unknown): value is FeedbackType {
  return value === 'like' || value === 'dislike';
}

type ValidPayload = FeedbackClientPayload & { userId: string };

function parsePayload(value: unknown): ValidPayload | null {
  if (typeof value !== 'object' || value === null) {
    return null;
  }
  const candidate = value as Record<string, unknown>;
  const { responseId, feedbackType, userId } = candidate;

  if (typeof responseId !== 'string' || responseId.length === 0) {
    return null;
  }
  if (typeof userId !== 'string' || userId.length === 0) {
    return null;
  }
  if (!isFeedbackType(feedbackType)) {
    return null;
  }

  const payload: ValidPayload = { responseId, feedbackType, userId };

  if (typeof candidate.questionText === 'string') {
    payload.questionText = candidate.questionText;
  }
  if (typeof candidate.answerText === 'string') {
    payload.answerText = candidate.answerText;
  }
  if (typeof candidate.inferenceModel === 'string') {
    payload.inferenceModel = candidate.inferenceModel;
  }

  return payload;
}

function toSchema(payload: ValidPayload): FeedbackSchema {
  const schema: FeedbackSchema = {
    response_id: payload.responseId,
    client: 'web',
    user_id: payload.userId,
    feedback_type: payload.feedbackType,
  };

  if (payload.questionText !== undefined) {
    schema.question_text = payload.questionText;
  }
  if (payload.answerText !== undefined) {
    schema.answer_text = payload.answerText;
  }
  if (payload.inferenceModel !== undefined) {
    schema.inference_model = payload.inferenceModel;
  }

  return schema;
}

function jsonError(message: string, status: number): Response {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

export async function POST(req: Request): Promise<Response> {
  let raw: unknown;
  try {
    raw = await req.json();
  } catch {
    return jsonError('Invalid JSON body', 400);
  }

  const payload = parsePayload(raw);
  if (payload === null) {
    return jsonError(
      'Invalid feedback payload: responseId, userId, and feedbackType (like|dislike) are required.',
      400,
    );
  }

  const schema = toSchema(payload);

  let upstream: Response;
  try {
    upstream = await fetch(`${API_BASE_URL}/chat/feedback`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-api-key': CHAT_API_KEY,
      },
      body: JSON.stringify(schema),
    });
  } catch {
    return jsonError('Failed to reach the feedback service.', 502);
  }

  if (!upstream.ok) {
    return jsonError('The feedback service rejected the request.', 502);
  }

  let ack: FeedbackAck;
  try {
    ack = (await upstream.json()) as FeedbackAck;
  } catch {
    return jsonError('The feedback service returned an invalid response.', 502);
  }

  return new Response(JSON.stringify(ack), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}
```

- [ ] **Step 4: Run the test, expect PASS.** Run:

```bash
cd web && npx vitest run test/api-feedback.route.test.ts
```

Expected: all 7 tests pass.

- [ ] **Step 5: Commit.**

```bash
git add web/app/api/feedback/route.ts web/test/api-feedback.route.test.ts
git commit -m "feat(web): add BFF POST /api/feedback proxy injecting server-only x-api-key"
```

I have read the spec and have the full contract. Now I'll write Tasks 8, 9, and 10. These are pure library/utility tasks (Dexie store, client utils, transport wiring) that I can write directly from the contract and cheat-sheet.

Let me verify a couple of API details against the cheat-sheet to make sure signatures are exact, particularly the `DefaultChatTransport` usage and Dexie patterns. I have those in the cheat-sheet already. Let me write the plan.

```markdown
### Task 8: Dexie store (`web/lib/db.ts`)

**Files:**
- Create: `web/lib/db.ts`
- Test: `web/test/db.test.ts`

**Interfaces:**
- Consumes (from Task 3, `@/lib/api-types`):
  - `type MyUIMessage = UIMessage<MyMetadata, MyDataParts>` — `MyUIMessage['role']`, `MyUIMessage['parts']`, `MyUIMessage['metadata']` are the shapes stored.
- Consumes (Task 1 dev infra): `fake-indexeddb/auto` is loaded in `web/vitest.setup.ts`; Vitest `environment: 'jsdom'`.
- Produces (Tasks 11/13 rely on these exact signatures):
  - `interface ConversationRow { id: string; title: string; model: string; createdAt: number; updatedAt: number }`
  - `interface MessageRow { id: string; conversationId: string; role: MyUIMessage['role']; parts: MyUIMessage['parts']; metadata?: MyUIMessage['metadata']; createdAt: number }`
  - `db` (Dexie instance with `conversations` and `messages` tables)
  - `createConversation(input: { id?: string; title: string; model: string }): Promise<ConversationRow>`
  - `listConversations(): Promise<ConversationRow[]>` (newest `updatedAt` first)
  - `renameConversation(id: string, title: string): Promise<void>`
  - `deleteConversation(id: string): Promise<void>` (also deletes that conversation's messages)
  - `loadMessages(conversationId: string): Promise<MessageRow[]>` (oldest `createdAt` first)
  - `saveMessages(conversationId: string, messages: MyUIMessage[]): Promise<void>` (upserts rows + bumps the conversation's `updatedAt`)

- [ ] **Step 1: Write the failing test for `createConversation` + `listConversations`.**

  Create `web/test/db.test.ts`:

  ```ts
  import { beforeEach, describe, expect, it } from 'vitest';
  import {
    createConversation,
    deleteConversation,
    db,
    listConversations,
    loadMessages,
    renameConversation,
    saveMessages,
  } from '@/lib/db';
  import type { MyUIMessage } from '@/lib/api-types';

  beforeEach(async () => {
    await db.delete();
    await db.open();
  });

  describe('conversations', () => {
    it('creates a conversation with a generated id and timestamps', async () => {
      const row = await createConversation({ title: 'Здраво', model: 'claude-sonnet-4-6' });
      expect(row.id).toMatch(/[0-9a-f-]{36}/);
      expect(row.title).toBe('Здраво');
      expect(row.model).toBe('claude-sonnet-4-6');
      expect(row.createdAt).toBeGreaterThan(0);
      expect(row.updatedAt).toBe(row.createdAt);
    });

    it('honours an explicit id', async () => {
      const row = await createConversation({ id: 'c-fixed', title: 'X', model: 'm' });
      expect(row.id).toBe('c-fixed');
    });

    it('lists conversations newest-updated first', async () => {
      const a = await createConversation({ id: 'a', title: 'A', model: 'm' });
      const b = await createConversation({ id: 'b', title: 'B', model: 'm' });
      // bump a's updatedAt above b's
      await renameConversation('a', 'A2');
      const list = await listConversations();
      expect(list.map((c) => c.id)).toEqual(['a', 'b']);
      expect(list[0].title).toBe('A2');
      expect(a.id).toBe('a');
      expect(b.id).toBe('b');
    });
  });
  ```

- [ ] **Step 2: Run the test — expect FAIL (module not found).**

  Command:
  ```bash
  cd web && npx vitest run test/db.test.ts
  ```
  Expected failure: `Failed to resolve import "@/lib/db"` (or `Cannot find module '@/lib/db'`), because `web/lib/db.ts` does not exist yet.

- [ ] **Step 3: Implement `web/lib/db.ts` schema + the conversation helpers.**

  Create `web/lib/db.ts`:

  ```ts
  // web/lib/db.ts
  // Local (IndexedDB) conversation + message store. Stores messages in the AI SDK
  // UIMessage shape so metadata.responseId survives reloads (spec §7).
  import { Dexie, type EntityTable } from 'dexie';
  import type { MyUIMessage } from '@/lib/api-types';

  export interface ConversationRow {
    id: string;
    title: string;
    model: string;
    createdAt: number;
    updatedAt: number;
  }

  export interface MessageRow {
    id: string;
    conversationId: string;
    role: MyUIMessage['role'];
    parts: MyUIMessage['parts'];
    metadata?: MyUIMessage['metadata'];
    createdAt: number;
  }

  export const db = new Dexie('finkiHubChat') as Dexie & {
    conversations: EntityTable<ConversationRow, 'id'>;
    messages: EntityTable<MessageRow, 'id'>;
  };

  db.version(1).stores({
    conversations: 'id, updatedAt', // primary key + index for ordered listing
    messages: 'id, conversationId, createdAt', // query by conversation, ordered by time
  });

  export async function createConversation(input: {
    id?: string;
    title: string;
    model: string;
  }): Promise<ConversationRow> {
    const now = Date.now();
    const row: ConversationRow = {
      id: input.id ?? crypto.randomUUID(),
      title: input.title,
      model: input.model,
      createdAt: now,
      updatedAt: now,
    };
    await db.conversations.put(row);
    return row;
  }

  export function listConversations(): Promise<ConversationRow[]> {
    return db.conversations.orderBy('updatedAt').reverse().toArray();
  }

  export async function renameConversation(id: string, title: string): Promise<void> {
    await db.conversations.update(id, { title, updatedAt: Date.now() });
  }

  export async function deleteConversation(id: string): Promise<void> {
    await db.transaction('rw', db.conversations, db.messages, async () => {
      await db.messages.where('conversationId').equals(id).delete();
      await db.conversations.delete(id);
    });
  }

  export function loadMessages(conversationId: string): Promise<MessageRow[]> {
    return db.messages.where('conversationId').equals(conversationId).sortBy('createdAt');
  }

  export async function saveMessages(
    conversationId: string,
    messages: MyUIMessage[],
  ): Promise<void> {
    const base = Date.now();
    const rows: MessageRow[] = messages.map((m, i) => ({
      id: m.id,
      conversationId,
      role: m.role,
      parts: m.parts,
      metadata: m.metadata,
      createdAt: base + i, // preserve send order within one batch
    }));
    await db.transaction('rw', db.conversations, db.messages, async () => {
      await db.messages.bulkPut(rows);
      await db.conversations.update(conversationId, { updatedAt: Date.now() });
    });
  }
  ```

- [ ] **Step 4: Re-run the conversation test — expect PASS.**

  Command:
  ```bash
  cd web && npx vitest run test/db.test.ts
  ```
  Expected: the `conversations` describe block passes (3 tests green).

- [ ] **Step 5: Add the failing test for messages (`saveMessages`/`loadMessages`/`deleteConversation`).**

  Append to `web/test/db.test.ts`:

  ```ts
  function userMsg(id: string, text: string): MyUIMessage {
    return { id, role: 'user', parts: [{ type: 'text', text }] };
  }

  function assistantMsg(id: string, text: string, responseId?: string): MyUIMessage {
    return {
      id,
      role: 'assistant',
      parts: [{ type: 'text', text }],
      metadata: responseId ? { responseId } : undefined,
    };
  }

  describe('messages', () => {
    it('round-trips UIMessage parts and metadata, ordered by createdAt', async () => {
      await createConversation({ id: 'c1', title: 'C1', model: 'm' });
      await saveMessages('c1', [
        userMsg('m1', 'прашање'),
        assistantMsg('m2', 'одговор', 'resp-123'),
      ]);
      const rows = await loadMessages('c1');
      expect(rows.map((r) => r.id)).toEqual(['m1', 'm2']);
      expect(rows[0].parts).toEqual([{ type: 'text', text: 'прашање' }]);
      expect(rows[1].metadata).toEqual({ responseId: 'resp-123' });
    });

    it('upserts on re-save (same id) and bumps the conversation updatedAt', async () => {
      const conv = await createConversation({ id: 'c2', title: 'C2', model: 'm' });
      await saveMessages('c2', [assistantMsg('m1', 'прв')]);
      await saveMessages('c2', [assistantMsg('m1', 'втор')]);
      const rows = await loadMessages('c2');
      expect(rows).toHaveLength(1);
      expect(rows[0].parts).toEqual([{ type: 'text', text: 'втор' }]);
      const updated = (await listConversations()).find((c) => c.id === 'c2')!;
      expect(updated.updatedAt).toBeGreaterThanOrEqual(conv.updatedAt);
    });

    it('deletes a conversation and its messages', async () => {
      await createConversation({ id: 'c3', title: 'C3', model: 'm' });
      await saveMessages('c3', [userMsg('m1', 'x')]);
      await deleteConversation('c3');
      expect(await listConversations()).toHaveLength(0);
      expect(await loadMessages('c3')).toHaveLength(0);
    });
  });
  ```

- [ ] **Step 6: Run the full file — expect PASS.**

  Command:
  ```bash
  cd web && npx vitest run test/db.test.ts
  ```
  Expected: all `conversations` + `messages` tests pass (no implementation change needed; Step 3 already covers them).

- [ ] **Step 7: Commit.**

  ```bash
  git add web/lib/db.ts web/test/db.test.ts
  git commit -m "feat(web): add Dexie conversation/message store with helpers"
  ```

---

### Task 9: Client utils — anon id + message helpers (`web/lib/user.ts`, `web/lib/messages.ts`)

**Files:**
- Create: `web/lib/user.ts`
- Create: `web/lib/messages.ts`
- Test: `web/test/user.test.ts`
- Test: `web/test/messages.test.ts`

**Interfaces:**
- Consumes (from Task 3, `@/lib/api-types`):
  - `type MyUIMessage`
  - `const MAX_MESSAGES = 50`
  - `const MAX_CHARS_PER_TURN = 8000`
- Consumes (Task 1 dev infra): Vitest `environment: 'jsdom'` (provides `localStorage` + `crypto.randomUUID`).
- Produces (Tasks 10/13 rely on these exact signatures):
  - `getAnonUserId(): string` — returns a stable per-browser UUID, persisted in `localStorage` under key `finkiHub.anonUserId`.
  - `ANON_USER_ID_KEY = 'finkiHub.anonUserId'` (exported const)
  - `trimForRequest(messages: MyUIMessage[]): MyUIMessage[]` — keeps the newest ≤ `MAX_MESSAGES`, and truncates each message's text parts so its combined text is ≤ `MAX_CHARS_PER_TURN`.
  - `deriveTitle(firstUserText: string): string` — first line, trimmed, ≤ 60 chars (ellipsis if cut), falls back to `'Нов разговор'` when empty.

- [ ] **Step 1: Write the failing test for `getAnonUserId`.**

  Create `web/test/user.test.ts`:

  ```ts
  import { beforeEach, expect, it } from 'vitest';
  import { ANON_USER_ID_KEY, getAnonUserId } from '@/lib/user';

  beforeEach(() => {
    localStorage.clear();
  });

  it('mints a UUID and persists it under the namespaced key', () => {
    const id = getAnonUserId();
    expect(id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/);
    expect(localStorage.getItem(ANON_USER_ID_KEY)).toBe(id);
  });

  it('returns the same id on subsequent calls', () => {
    const first = getAnonUserId();
    const second = getAnonUserId();
    expect(second).toBe(first);
  });

  it('reuses an id already present in localStorage', () => {
    localStorage.setItem(ANON_USER_ID_KEY, 'preexisting-id');
    expect(getAnonUserId()).toBe('preexisting-id');
  });
  ```

- [ ] **Step 2: Run the test — expect FAIL (module not found).**

  Command:
  ```bash
  cd web && npx vitest run test/user.test.ts
  ```
  Expected failure: `Failed to resolve import "@/lib/user"` — `web/lib/user.ts` does not exist yet.

- [ ] **Step 3: Implement `web/lib/user.ts`.**

  Create `web/lib/user.ts`:

  ```ts
  // web/lib/user.ts
  // Stable anonymous per-browser id, persisted in localStorage. Sent to
  // /api/feedback as user_id (the backend requires a non-empty user_id, spec §7).
  export const ANON_USER_ID_KEY = 'finkiHub.anonUserId';

  export function getAnonUserId(): string {
    const existing = localStorage.getItem(ANON_USER_ID_KEY);
    if (existing && existing.length > 0) {
      return existing;
    }
    const id = crypto.randomUUID();
    localStorage.setItem(ANON_USER_ID_KEY, id);
    return id;
  }
  ```

- [ ] **Step 4: Re-run the user test — expect PASS.**

  Command:
  ```bash
  cd web && npx vitest run test/user.test.ts
  ```
  Expected: all 3 tests pass.

- [ ] **Step 5: Write the failing test for `trimForRequest` + `deriveTitle`.**

  Create `web/test/messages.test.ts`:

  ```ts
  import { describe, expect, it } from 'vitest';
  import type { MyUIMessage } from '@/lib/api-types';
  import { MAX_CHARS_PER_TURN, MAX_MESSAGES } from '@/lib/api-types';
  import { deriveTitle, trimForRequest } from '@/lib/messages';

  function textMsg(id: string, role: MyUIMessage['role'], text: string): MyUIMessage {
    return { id, role, parts: [{ type: 'text', text }] };
  }

  describe('trimForRequest', () => {
    it('keeps the newest MAX_MESSAGES when over the cap', () => {
      const msgs: MyUIMessage[] = Array.from({ length: MAX_MESSAGES + 5 }, (_, i) =>
        textMsg(`m${i}`, i % 2 === 0 ? 'user' : 'assistant', `t${i}`),
      );
      const trimmed = trimForRequest(msgs);
      expect(trimmed).toHaveLength(MAX_MESSAGES);
      // newest preserved, oldest dropped
      expect(trimmed[0].id).toBe('m5');
      expect(trimmed.at(-1)!.id).toBe(`m${MAX_MESSAGES + 4}`);
    });

    it('returns the same list when under the cap', () => {
      const msgs = [textMsg('a', 'user', 'hi')];
      expect(trimForRequest(msgs)).toHaveLength(1);
    });

    it('truncates an over-long turn to MAX_CHARS_PER_TURN', () => {
      const long = 'я'.repeat(MAX_CHARS_PER_TURN + 100);
      const trimmed = trimForRequest([textMsg('a', 'user', long)]);
      const part = trimmed[0].parts[0];
      expect(part.type).toBe('text');
      expect((part as { type: 'text'; text: string }).text).toHaveLength(MAX_CHARS_PER_TURN);
    });

    it('truncates across multiple text parts by combined length', () => {
      const a = 'a'.repeat(5000);
      const b = 'b'.repeat(5000);
      const msg: MyUIMessage = {
        id: 'm',
        role: 'user',
        parts: [
          { type: 'text', text: a },
          { type: 'text', text: b },
        ],
      };
      const trimmed = trimForRequest([msg]);
      const total = trimmed[0].parts
        .filter((p): p is { type: 'text'; text: string } => p.type === 'text')
        .reduce((n, p) => n + p.text.length, 0);
      expect(total).toBe(MAX_CHARS_PER_TURN);
    });
  });

  describe('deriveTitle', () => {
    it('uses the first line, trimmed', () => {
      expect(deriveTitle('  Која е оценката?\nвтор ред  ')).toBe('Која е оценката?');
    });

    it('truncates to 60 chars with an ellipsis', () => {
      const title = deriveTitle('a'.repeat(100));
      expect(title).toHaveLength(60);
      expect(title.endsWith('…')).toBe(true);
    });

    it('falls back when empty', () => {
      expect(deriveTitle('   ')).toBe('Нов разговор');
      expect(deriveTitle('')).toBe('Нов разговор');
    });
  });
  ```

- [ ] **Step 6: Run the test — expect FAIL (module not found).**

  Command:
  ```bash
  cd web && npx vitest run test/messages.test.ts
  ```
  Expected failure: `Failed to resolve import "@/lib/messages"` — `web/lib/messages.ts` does not exist yet.

- [ ] **Step 7: Implement `web/lib/messages.ts`.**

  Create `web/lib/messages.ts`:

  ```ts
  // web/lib/messages.ts
  // Client-side request shaping: enforce the API caps (≤ 50 messages,
  // ≤ 8000 chars/turn) before sending, and derive a conversation title.
  // The BFF re-validates; this keeps the UI honest and the payload small.
  import type { MyUIMessage } from '@/lib/api-types';
  import { MAX_CHARS_PER_TURN, MAX_MESSAGES } from '@/lib/api-types';

  const FALLBACK_TITLE = 'Нов разговор';
  const TITLE_MAX = 60;

  /** Truncate a message's text parts so their combined length ≤ MAX_CHARS_PER_TURN. */
  function capTextParts(message: MyUIMessage): MyUIMessage {
    let budget = MAX_CHARS_PER_TURN;
    let needsCap = false;
    let used = 0;
    for (const part of message.parts) {
      if (part.type === 'text') {
        used += part.text.length;
      }
    }
    if (used <= MAX_CHARS_PER_TURN) {
      return message;
    }
    needsCap = true;
    const parts = message.parts.map((part) => {
      if (part.type !== 'text') {
        return part;
      }
      if (budget <= 0) {
        return { ...part, text: '' };
      }
      const slice = part.text.slice(0, budget);
      budget -= slice.length;
      return { ...part, text: slice };
    });
    return needsCap ? { ...message, parts } : message;
  }

  /** Enforce ≤ MAX_MESSAGES (keep the newest) and ≤ MAX_CHARS_PER_TURN per turn. */
  export function trimForRequest(messages: MyUIMessage[]): MyUIMessage[] {
    const windowed =
      messages.length > MAX_MESSAGES ? messages.slice(messages.length - MAX_MESSAGES) : messages;
    return windowed.map(capTextParts);
  }

  /** First line of the first user message, trimmed to TITLE_MAX with an ellipsis. */
  export function deriveTitle(firstUserText: string): string {
    const firstLine = firstUserText.split('\n')[0]?.trim() ?? '';
    if (firstLine.length === 0) {
      return FALLBACK_TITLE;
    }
    if (firstLine.length <= TITLE_MAX) {
      return firstLine;
    }
    return `${firstLine.slice(0, TITLE_MAX - 1)}…`;
  }
  ```

- [ ] **Step 8: Re-run the messages test — expect PASS.**

  Command:
  ```bash
  cd web && npx vitest run test/messages.test.ts
  ```
  Expected: all `trimForRequest` + `deriveTitle` tests pass.

- [ ] **Step 9: Commit.**

  ```bash
  git add web/lib/user.ts web/lib/messages.ts web/test/user.test.ts web/test/messages.test.ts
  git commit -m "feat(web): add anon user id + request-trimming/title helpers"
  ```

---

### Task 10: Transport wiring (`web/lib/transport.ts`)

**Files:**
- Create: `web/lib/transport.ts`
- Test: `web/test/transport.test.ts`

**Interfaces:**
- Consumes (from Task 9, `@/lib/user`):
  - `getAnonUserId(): string`
- Consumes (from Task 3, `@/lib/api-types`):
  - `type MyUIMessage`
  - `type ModelId = string`
- Consumes (`ai` package): `DefaultChatTransport<MyUIMessage>` with option `prepareSendMessagesRequest({ messages, id, trigger, messageId }) => { body }`.
- Produces (Tasks 11/13 rely on these exact signatures):
  - `interface ChatExtras { model: ModelId; embeddingsModel?: ModelId; queryTransformModel?: ModelId; temperature?: number; topP?: number; maxTokens?: number }`
  - `buildChatTransport(getExtras: () => ChatExtras): DefaultChatTransport<MyUIMessage>` — a transport pointed at `/api/chat` whose request `body` carries `{ messages, id, trigger, messageId, ...extras, userId }`, with `userId` from `getAnonUserId()`.

- [ ] **Step 1: Write the failing test for the prepared request body.**

  The transport's `prepareSendMessagesRequest` is the testable seam. We exercise it directly (no live `useChat` needed) by reading it back off the constructed transport, and we stub `getAnonUserId` via the injected `getExtras` boundary plus a spy on the `@/lib/user` module.

  Create `web/test/transport.test.ts`:

  ```ts
  import { describe, expect, it, vi } from 'vitest';

  vi.mock('@/lib/user', () => ({
    ANON_USER_ID_KEY: 'finkiHub.anonUserId',
    getAnonUserId: () => 'anon-test-id',
  }));

  import type { MyUIMessage } from '@/lib/api-types';
  import { buildChatTransport, type ChatExtras } from '@/lib/transport';

  type PrepareArgs = {
    messages: MyUIMessage[];
    id: string;
    trigger: 'submit-message' | 'regenerate-message';
    messageId?: string;
  };

  // The transport exposes prepareSendMessagesRequest as a configured option we can call.
  function getPrepare(extras: ChatExtras) {
    const transport = buildChatTransport(() => extras) as unknown as {
      prepareSendMessagesRequest: (args: PrepareArgs) => { body: Record<string, unknown> };
    };
    return transport.prepareSendMessagesRequest.bind(transport);
  }

  const sampleMessages: MyUIMessage[] = [
    { id: 'u1', role: 'user', parts: [{ type: 'text', text: 'здраво' }] },
  ];

  describe('buildChatTransport', () => {
    it('puts messages, id, trigger, extras, and userId into the request body', () => {
      const prepare = getPrepare({ model: 'claude-sonnet-4-6', temperature: 0.3 });
      const { body } = prepare({
        messages: sampleMessages,
        id: 'conv-1',
        trigger: 'submit-message',
        messageId: 'm-1',
      });
      expect(body).toMatchObject({
        messages: sampleMessages,
        id: 'conv-1',
        trigger: 'submit-message',
        messageId: 'm-1',
        model: 'claude-sonnet-4-6',
        temperature: 0.3,
        userId: 'anon-test-id',
      });
    });

    it('forwards all sampling params when present', () => {
      const prepare = getPrepare({
        model: 'gpt-5.4-mini',
        embeddingsModel: 'BAAI/bge-m3',
        queryTransformModel: 'gpt-5.4-mini',
        temperature: 0.5,
        topP: 0.9,
        maxTokens: 2048,
      });
      const { body } = prepare({ messages: sampleMessages, id: 'c', trigger: 'submit-message' });
      expect(body).toMatchObject({
        model: 'gpt-5.4-mini',
        embeddingsModel: 'BAAI/bge-m3',
        queryTransformModel: 'gpt-5.4-mini',
        temperature: 0.5,
        topP: 0.9,
        maxTokens: 2048,
      });
    });

    it('reads extras lazily on every call (picks up model changes)', () => {
      let model = 'model-a';
      const transport = buildChatTransport(() => ({ model })) as unknown as {
        prepareSendMessagesRequest: (args: PrepareArgs) => { body: Record<string, unknown> };
      };
      const first = transport.prepareSendMessagesRequest({
        messages: sampleMessages,
        id: 'c',
        trigger: 'submit-message',
      });
      expect((first.body as { model: string }).model).toBe('model-a');
      model = 'model-b';
      const second = transport.prepareSendMessagesRequest({
        messages: sampleMessages,
        id: 'c',
        trigger: 'submit-message',
      });
      expect((second.body as { model: string }).model).toBe('model-b');
    });
  });
  ```

- [ ] **Step 2: Run the test — expect FAIL (module not found).**

  Command:
  ```bash
  cd web && npx vitest run test/transport.test.ts
  ```
  Expected failure: `Failed to resolve import "@/lib/transport"` — `web/lib/transport.ts` does not exist yet.

- [ ] **Step 3: Implement `web/lib/transport.ts`.**

  Create `web/lib/transport.ts`:

  ```ts
  // web/lib/transport.ts
  // The AI SDK v5 transport for useChat. Points at the same-origin BFF (/api/chat)
  // and injects the model/sampling params plus the anonymous userId into the
  // request body via prepareSendMessagesRequest (spec §6). Extras are read lazily
  // so the active model can change between turns without rebuilding the transport.
  import { DefaultChatTransport } from 'ai';
  import type { ModelId, MyUIMessage } from '@/lib/api-types';
  import { getAnonUserId } from '@/lib/user';

  export interface ChatExtras {
    model: ModelId;
    embeddingsModel?: ModelId;
    queryTransformModel?: ModelId;
    temperature?: number;
    topP?: number;
    maxTokens?: number;
  }

  export function buildChatTransport(
    getExtras: () => ChatExtras,
  ): DefaultChatTransport<MyUIMessage> {
    return new DefaultChatTransport<MyUIMessage>({
      api: '/api/chat',
      prepareSendMessagesRequest: ({ messages, id, trigger, messageId }) => ({
        body: {
          messages,
          id,
          trigger,
          messageId,
          ...getExtras(),
          userId: getAnonUserId(),
        },
      }),
    });
  }
  ```

- [ ] **Step 4: Re-run the transport test — expect PASS.**

  Command:
  ```bash
  cd web && npx vitest run test/transport.test.ts
  ```
  Expected: all 3 `buildChatTransport` tests pass.

  > Note: `DefaultChatTransport` stores `prepareSendMessagesRequest` on the instance, so the cast in the test reads it back directly. If a future AI SDK patch renames the field, this test is the canary — keep the option name aligned with the cheat-sheet (`prepareSendMessagesRequest`).

- [ ] **Step 5: Commit.**

  ```bash
  git add web/lib/transport.ts web/test/transport.test.ts
  git commit -m "feat(web): add useChat transport injecting model params + anon userId"
  ```
```

I have the spec. Now I'll write Tasks 11-15, drawing on the cheat-sheet and the contract types. Let me produce the markdown.

I have everything I need. The spec, contract types (api-types.ts, env.ts), and the cheat-sheet give me the exact interfaces. Now writing Tasks 11-15.

### Task 11: Thread view (Conversation/Message/MessageResponse, render-last, SearchStatus, inline error)

**Files:**
- Create: `web/components/chat/thread.tsx`, `web/components/chat/message.tsx`, `web/components/chat/search-status.tsx`
- Test: `web/test/thread.test.tsx`

**Interfaces:**

Consumes (Task 3, `@/lib/api-types`):
- `type MyUIMessage = UIMessage<MyMetadata, MyDataParts>` where `MyMetadata = { responseId?: string; inferenceModel?: string }`
- `MyDataParts = { status: { label: string; tool?: string }; error: { code: string; message: string } }`

Consumes (Task 10, `@/lib/transport`): `buildChatTransport()` — used by Task 13, not directly here. Task 11 is presentational: it receives `messages`, `status`, an active status part, an active error, and an `onRetry` callback as props so it is unit-testable without a live `useChat`.

Consumes (Task 2, vendored AI Elements): `Conversation`, `ConversationContent`, `ConversationEmptyState`, `ConversationScrollButton` from `@/components/ai-elements/conversation`; `Message`, `MessageContent`, `MessageResponse` from `@/components/ai-elements/message`.

Consumes (Task 15, `@/lib/i18n`): `t(key)` — **but Task 15 is authored after this in the plan**, so Task 11 uses literal Macedonian strings inline and Task 15's step wires `t()` in. To avoid a forward dependency that breaks compilation, Task 11 ships with inline literals; Task 15 includes the edit that swaps them for `t()`.

Produces (later tasks rely on these exact names/props):
- `SearchStatus` — `export function SearchStatus(props: { label: string; tool?: string }): JSX.Element` (`web/components/chat/search-status.tsx`)
- `AssistantMessage` — `export function AssistantMessage(props: { message: MyUIMessage; statusPart?: { label: string; tool?: string }; errorPart?: { code: string; message: string }; onRetry?: () => void; actions?: React.ReactNode }): JSX.Element` (`web/components/chat/message.tsx`)
- `Thread` — `export function Thread(props: { messages: MyUIMessage[]; status: 'ready' | 'streaming' | 'submitted' | 'error'; activeStatus?: { label: string; tool?: string }; activeError?: { code: string; message: string }; onRetry?: () => void; renderActions?: (message: MyUIMessage) => React.ReactNode }): JSX.Element` (`web/components/chat/thread.tsx`)

Notes baked into the design:
- **Render-last:** for an assistant message, render only the last `type:'text'` part (preamble drop, §5.2).
- The search chip shows only while the active status belongs to the streaming assistant turn **and** that turn has no answer text yet.
- Inline error shows a **Retry** button unless `code === 'interrupted'`; interrupted shows a soft notice and keeps the partial.

Steps:

- [ ] **Step 1: Write the failing test for `SearchStatus`.** Create `web/test/thread.test.tsx` with the first block:

```tsx
// web/test/thread.test.tsx
import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { SearchStatus } from '@/components/chat/search-status';

describe('SearchStatus', () => {
  it('renders the label and exposes a status role', () => {
    render(<SearchStatus label="🔍 Пребарувам…" tool="faq_search" />);
    const chip = screen.getByRole('status');
    expect(chip).toHaveTextContent('🔍 Пребарувам…');
  });

  it('renders without a tool', () => {
    render(<SearchStatus label="Размислувам…" />);
    expect(screen.getByRole('status')).toHaveTextContent('Размислувам…');
  });
});
```

- [ ] **Step 2: Run it, expect FAIL.** Run `cd web && npx vitest run test/thread.test.tsx`. Expect failure: `Failed to resolve import "@/components/chat/search-status"` (module does not exist yet).

- [ ] **Step 3: Implement `SearchStatus`.** Create `web/components/chat/search-status.tsx`:

```tsx
// web/components/chat/search-status.tsx
'use client';

import { Loader2 } from 'lucide-react';

export interface SearchStatusProps {
  label: string;
  tool?: string;
}

export function SearchStatus({ label, tool }: SearchStatusProps) {
  return (
    <div
      role="status"
      aria-live="polite"
      data-testid="search-status"
      data-tool={tool}
      className="inline-flex items-center gap-2 rounded-full border border-border bg-muted/60 px-3 py-1 text-sm text-muted-foreground"
    >
      <Loader2 className="size-4 animate-spin" aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}
```

- [ ] **Step 4: Run it, expect PASS.** Run `cd web && npx vitest run test/thread.test.tsx`. Expect the two `SearchStatus` tests to pass.

- [ ] **Step 5: Commit.**
```bash
git add web/components/chat/search-status.tsx web/test/thread.test.tsx
git commit -m "feat(web): add SearchStatus chip for streaming tool indicator"
```

- [ ] **Step 6: Write the failing test for `AssistantMessage` render-last + error.** Append to `web/test/thread.test.tsx`:

```tsx
import { AssistantMessage } from '@/components/chat/message';
import type { MyUIMessage } from '@/lib/api-types';

function assistantWithParts(parts: MyUIMessage['parts']): MyUIMessage {
  return { id: 'a1', role: 'assistant', parts, metadata: {} };
}

describe('AssistantMessage', () => {
  it('renders only the LAST text part (preamble drop)', () => {
    const msg = assistantWithParts([
      { type: 'text', text: 'Барам во базата…' },
      { type: 'text', text: 'Конечниот одговор е тука.' },
    ]);
    render(<AssistantMessage message={msg} />);
    expect(screen.getByText('Конечниот одговор е тука.')).toBeInTheDocument();
    expect(screen.queryByText('Барам во базата…')).not.toBeInTheDocument();
  });

  it('shows the search chip when a status is active and no text yet', () => {
    const msg = assistantWithParts([]);
    render(<AssistantMessage message={msg} statusPart={{ label: '🔍 Пребарувам…' }} />);
    expect(screen.getByRole('status')).toHaveTextContent('🔍 Пребарувам…');
  });

  it('hides the search chip once answer text has arrived', () => {
    const msg = assistantWithParts([{ type: 'text', text: 'Одговор' }]);
    render(<AssistantMessage message={msg} statusPart={{ label: '🔍 Пребарувам…' }} />);
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
  });

  it('renders a Retry button for a non-interrupted error', () => {
    const onRetry = vi.fn();
    const msg = assistantWithParts([]);
    render(
      <AssistantMessage
        message={msg}
        errorPart={{ code: 'agent_error', message: 'Се случи грешка.' }}
        onRetry={onRetry}
      />,
    );
    expect(screen.getByText('Се случи грешка.')).toBeInTheDocument();
    screen.getByRole('button', { name: 'Обиди се повторно' }).click();
    expect(onRetry).toHaveBeenCalledOnce();
  });

  it('shows a soft notice (no Retry) for an interrupted error', () => {
    const msg = assistantWithParts([{ type: 'text', text: 'Делумен одговор' }]);
    render(
      <AssistantMessage
        message={msg}
        errorPart={{ code: 'interrupted', message: 'прекинато' }}
      />,
    );
    expect(screen.getByText('Делумен одговор')).toBeInTheDocument();
    expect(screen.getByText('Одговорот е прекинат.')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Обиди се повторно' })).not.toBeInTheDocument();
  });
});
```

- [ ] **Step 7: Run it, expect FAIL.** Run `cd web && npx vitest run test/thread.test.tsx`. Expect failure: `Failed to resolve import "@/components/chat/message"`.

- [ ] **Step 8: Implement `AssistantMessage`.** Create `web/components/chat/message.tsx`:

```tsx
// web/components/chat/message.tsx
'use client';

import type { ReactNode } from 'react';
import { Message, MessageContent, MessageResponse } from '@/components/ai-elements/message';
import type { MyUIMessage } from '@/lib/api-types';
import { SearchStatus } from '@/components/chat/search-status';

export interface AssistantMessageProps {
  message: MyUIMessage;
  statusPart?: { label: string; tool?: string };
  errorPart?: { code: string; message: string };
  onRetry?: () => void;
  actions?: ReactNode;
}

/** Last text part only — drops the pre-tool preamble (spec §5.2). */
function lastText(message: MyUIMessage): string | null {
  const texts = message.parts.filter((p): p is { type: 'text'; text: string } => p.type === 'text');
  const last = texts.at(-1);
  return last ? last.text : null;
}

export function AssistantMessage({ message, statusPart, errorPart, onRetry, actions }: AssistantMessageProps) {
  const text = lastText(message);
  const showChip = !!statusPart && !text;
  const isInterrupted = errorPart?.code === 'interrupted';

  return (
    <Message from="assistant">
      <MessageContent>
        {text ? <div data-testid="answer-text"><MessageResponse>{text}</MessageResponse></div> : null}
        {showChip ? <SearchStatus label={statusPart!.label} tool={statusPart!.tool} /> : null}
        {errorPart ? (
          <div role="alert" className="mt-2 rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm">
            {isInterrupted ? (
              <p className="text-muted-foreground">Одговорот е прекинат.</p>
            ) : (
              <div className="flex flex-col gap-2">
                <p className="text-destructive">{errorPart.message}</p>
                {onRetry ? (
                  <button
                    type="button"
                    onClick={onRetry}
                    className="self-start rounded-md border border-border px-3 py-1 text-sm hover:bg-muted"
                  >
                    Обиди се повторно
                  </button>
                ) : null}
              </div>
            )}
          </div>
        ) : null}
        {actions ? <div className="mt-2">{actions}</div> : null}
      </MessageContent>
    </Message>
  );
}
```

- [ ] **Step 9: Run it, expect PASS.** Run `cd web && npx vitest run test/thread.test.tsx`. Expect all `AssistantMessage` tests to pass.

- [ ] **Step 10: Commit.**
```bash
git add web/components/chat/message.tsx web/test/thread.test.tsx
git commit -m "feat(web): render assistant message with last-text part, chip, and inline error"
```

- [ ] **Step 11: Write the failing test for `Thread`.** Append to `web/test/thread.test.tsx`:

```tsx
import { Thread } from '@/components/chat/thread';

function userMessage(text: string): MyUIMessage {
  return { id: 'u1', role: 'user', parts: [{ type: 'text', text }], metadata: {} };
}

describe('Thread', () => {
  it('shows the empty state when there are no messages', () => {
    render(<Thread messages={[]} status="ready" />);
    expect(screen.getByText('Започни разговор')).toBeInTheDocument();
  });

  it('renders a user turn and an assistant answer (last text part)', () => {
    const messages: MyUIMessage[] = [
      userMessage('Кога е роковниот испит?'),
      assistantWithParts([
        { type: 'text', text: 'преамбула' },
        { type: 'text', text: 'Роковниот испит е во јануари.' },
      ]),
    ];
    render(<Thread messages={messages} status="ready" />);
    expect(screen.getByText('Кога е роковниот испит?')).toBeInTheDocument();
    expect(screen.getByText('Роковниот испит е во јануари.')).toBeInTheDocument();
    expect(screen.queryByText('преамбула')).not.toBeInTheDocument();
  });

  it('passes the active status only to the LAST assistant message while streaming', () => {
    const messages: MyUIMessage[] = [userMessage('прашање'), assistantWithParts([])];
    render(
      <Thread
        messages={messages}
        status="streaming"
        activeStatus={{ label: '🔍 Пребарувам…' }}
      />,
    );
    expect(screen.getByRole('status')).toHaveTextContent('🔍 Пребарувам…');
  });

  it('renders per-message actions via renderActions', () => {
    const messages: MyUIMessage[] = [assistantWithParts([{ type: 'text', text: 'готово' }])];
    render(
      <Thread
        messages={messages}
        status="ready"
        renderActions={(m) => <span data-testid="acts">{m.id}</span>}
      />,
    );
    expect(screen.getByTestId('acts')).toHaveTextContent('a1');
  });
});
```

- [ ] **Step 12: Run it, expect FAIL.** Run `cd web && npx vitest run test/thread.test.tsx`. Expect failure: `Failed to resolve import "@/components/chat/thread"`.

- [ ] **Step 13: Implement `Thread`.** Create `web/components/chat/thread.tsx`:

```tsx
// web/components/chat/thread.tsx
'use client';

import type { ReactNode } from 'react';
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from '@/components/ai-elements/conversation';
import { Message, MessageContent, MessageResponse } from '@/components/ai-elements/message';
import type { MyUIMessage } from '@/lib/api-types';
import { AssistantMessage } from '@/components/chat/message';

export interface ThreadProps {
  messages: MyUIMessage[];
  status: 'ready' | 'streaming' | 'submitted' | 'error';
  activeStatus?: { label: string; tool?: string };
  activeError?: { code: string; message: string };
  onRetry?: () => void;
  renderActions?: (message: MyUIMessage) => ReactNode;
}

function userText(message: MyUIMessage): string {
  return message.parts
    .filter((p): p is { type: 'text'; text: string } => p.type === 'text')
    .map((p) => p.text)
    .join('');
}

export function Thread({ messages, status, activeStatus, activeError, onRetry, renderActions }: ThreadProps) {
  const lastAssistantId = messages.filter((m) => m.role === 'assistant').at(-1)?.id;
  const streaming = status === 'streaming' || status === 'submitted';

  return (
    <Conversation className="flex-1">
      <ConversationContent>
        {messages.length === 0 ? (
          <ConversationEmptyState
            title="Започни разговор"
            description="Прашај нешто за студиите на ФИНКИ."
          />
        ) : (
          messages.map((m) => {
            if (m.role === 'user') {
              return (
                <Message from="user" key={m.id}>
                  <MessageContent>
                    <MessageResponse>{userText(m)}</MessageResponse>
                  </MessageContent>
                </Message>
              );
            }
            const isLastAssistant = m.id === lastAssistantId;
            return (
              <AssistantMessage
                key={m.id}
                message={m}
                statusPart={isLastAssistant && streaming ? activeStatus : undefined}
                errorPart={isLastAssistant ? activeError : undefined}
                onRetry={onRetry}
                actions={renderActions ? renderActions(m) : undefined}
              />
            );
          })
        )}
      </ConversationContent>
      <ConversationScrollButton />
    </Conversation>
  );
}
```

- [ ] **Step 14: Run the full file, expect PASS.** Run `cd web && npx vitest run test/thread.test.tsx`. Expect all `SearchStatus`, `AssistantMessage`, and `Thread` tests green.

- [ ] **Step 15: Typecheck.** Run `cd web && npx tsc --noEmit`. Expect no errors.

- [ ] **Step 16: Commit.**
```bash
git add web/components/chat/thread.tsx web/test/thread.test.tsx
git commit -m "feat(web): add Thread view rendering AI Elements conversation with preamble drop"
```

### Task 12: Composer + model picker

**Files:**
- Create: `web/components/chat/composer.tsx`, `web/lib/use-models.ts`
- Test: `web/test/composer.test.tsx`

**Interfaces:**

Consumes (Task 2, vendored AI Elements): `PromptInput`, `PromptInputTextarea`, `PromptInputSubmit`, `type PromptInputMessage` from `@/components/ai-elements/prompt-input`; `PromptInputSelect` (model picker) — if the vendored set names it differently, the implementation below uses the shadcn `Select` primitives from `@/components/ui/select` directly so the task is self-contained.

Consumes (Task 6, BFF): `GET /api/models` → `string[]`.

Consumes (TanStack Query, Task 1/11 libs): `QueryClient`, `QueryClientProvider`, `useQuery` from `@tanstack/react-query`.

Produces (Task 13 relies on these):
- `useModels` — `export function useModels(): { data: string[] | undefined; isLoading: boolean; isError: boolean }` (`web/lib/use-models.ts`)
- `groupModelsByProvider` — `export function groupModelsByProvider(ids: string[]): Array<{ provider: string; models: string[] }>` (`web/lib/use-models.ts`)
- `Composer` — `export function Composer(props: { model: string; models: string[]; onModelChange: (model: string) => void; onSubmit: (text: string) => void; onStop: () => void; status: 'ready' | 'streaming' | 'submitted' | 'error'; disabled?: boolean }): JSX.Element` (`web/components/chat/composer.tsx`)

Provider inference: split the id on the first `/` (e.g. `BAAI/bge-m3` → `BAAI`); if no `/`, derive from the prefix before the first `-` (e.g. `claude-sonnet-4-6` → `claude`, `gpt-5.4-mini` → `gpt`); fallback `'other'`.

Steps:

- [ ] **Step 1: Write the failing test for `groupModelsByProvider`.** Create `web/test/composer.test.tsx`:

```tsx
// web/test/composer.test.tsx
import { describe, expect, it, vi } from 'vitest';
import { groupModelsByProvider } from '@/lib/use-models';

describe('groupModelsByProvider', () => {
  it('groups by the part before "/" when present', () => {
    const groups = groupModelsByProvider(['BAAI/bge-m3', 'BAAI/bge-large']);
    expect(groups).toEqual([{ provider: 'BAAI', models: ['BAAI/bge-large', 'BAAI/bge-m3'] }]);
  });

  it('infers the provider from the name prefix when there is no slash', () => {
    const groups = groupModelsByProvider(['claude-sonnet-4-6', 'gpt-5.4-mini', 'claude-opus-4-8']);
    expect(groups).toEqual([
      { provider: 'claude', models: ['claude-opus-4-8', 'claude-sonnet-4-6'] },
      { provider: 'gpt', models: ['gpt-5.4-mini'] },
    ]);
  });

  it('sorts providers and models and handles unknown ids', () => {
    const groups = groupModelsByProvider(['zeta', 'BAAI/bge-m3']);
    expect(groups).toEqual([
      { provider: 'BAAI', models: ['BAAI/bge-m3'] },
      { provider: 'zeta', models: ['zeta'] },
    ]);
  });
});
```

- [ ] **Step 2: Run it, expect FAIL.** Run `cd web && npx vitest run test/composer.test.tsx`. Expect failure: `Failed to resolve import "@/lib/use-models"`.

- [ ] **Step 3: Implement `useModels` + `groupModelsByProvider`.** Create `web/lib/use-models.ts`:

```ts
// web/lib/use-models.ts
'use client';

import { useQuery } from '@tanstack/react-query';
import type { ModelId } from '@/lib/api-types';

async function fetchModels(): Promise<ModelId[]> {
  const res = await fetch('/api/models');
  if (!res.ok) {
    throw new Error(`Failed to load models: ${res.status}`);
  }
  const data: unknown = await res.json();
  return Array.isArray(data) ? (data as ModelId[]) : [];
}

export function useModels() {
  const query = useQuery({
    queryKey: ['models'],
    queryFn: fetchModels,
    staleTime: 5 * 60 * 1000,
  });
  return { data: query.data, isLoading: query.isLoading, isError: query.isError };
}

function providerOf(id: string): string {
  const slash = id.indexOf('/');
  if (slash > 0) {
    return id.slice(0, slash);
  }
  const dash = id.indexOf('-');
  if (dash > 0) {
    return id.slice(0, dash);
  }
  return id.length > 0 ? id : 'other';
}

export interface ModelGroup {
  provider: string;
  models: string[];
}

export function groupModelsByProvider(ids: string[]): ModelGroup[] {
  const map = new Map<string, string[]>();
  for (const id of ids) {
    const provider = providerOf(id);
    const bucket = map.get(provider);
    if (bucket) {
      bucket.push(id);
    } else {
      map.set(provider, [id]);
    }
  }
  return [...map.entries()]
    .map(([provider, models]) => ({ provider, models: [...models].sort() }))
    .sort((a, b) => a.provider.localeCompare(b.provider));
}
```

- [ ] **Step 4: Run it, expect PASS.** Run `cd web && npx vitest run test/composer.test.tsx`. Expect the three `groupModelsByProvider` tests to pass.

- [ ] **Step 5: Commit.**
```bash
git add web/lib/use-models.ts web/test/composer.test.tsx
git commit -m "feat(web): add useModels query hook and provider grouping"
```

- [ ] **Step 6: Write the failing test for `Composer`.** Append to `web/test/composer.test.tsx`:

```tsx
import { fireEvent, render, screen } from '@testing-library/react';
import { Composer } from '@/components/chat/composer';

function setup(overrides: Partial<React.ComponentProps<typeof Composer>> = {}) {
  const onSubmit = vi.fn();
  const onStop = vi.fn();
  const onModelChange = vi.fn();
  render(
    <Composer
      model="claude-sonnet-4-6"
      models={['claude-sonnet-4-6', 'gpt-5.4-mini']}
      onModelChange={onModelChange}
      onSubmit={onSubmit}
      onStop={onStop}
      status="ready"
      {...overrides}
    />,
  );
  return { onSubmit, onStop, onModelChange };
}

describe('Composer', () => {
  it('submits trimmed text on Enter (no Shift)', () => {
    const { onSubmit } = setup();
    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: '  Здраво  ' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });
    expect(onSubmit).toHaveBeenCalledWith('Здраво');
  });

  it('does NOT submit on Shift+Enter (newline)', () => {
    const { onSubmit } = setup();
    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: 'ред' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: true });
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('does not submit empty/whitespace-only input', () => {
    const { onSubmit } = setup();
    const textarea = screen.getByRole('textbox');
    fireEvent.change(textarea, { target: { value: '   ' } });
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('calls onStop when the submit button is clicked while streaming', () => {
    const { onStop } = setup({ status: 'streaming' });
    fireEvent.click(screen.getByTestId('composer-submit'));
    expect(onStop).toHaveBeenCalledOnce();
  });

  it('renders every model id as an option', () => {
    setup();
    expect(screen.getByRole('option', { name: 'claude-sonnet-4-6' })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: 'gpt-5.4-mini' })).toBeInTheDocument();
  });

  it('reports model changes', () => {
    const { onModelChange } = setup();
    fireEvent.change(screen.getByTestId('composer-model'), { target: { value: 'gpt-5.4-mini' } });
    expect(onModelChange).toHaveBeenCalledWith('gpt-5.4-mini');
  });
});
```

- [ ] **Step 7: Run it, expect FAIL.** Run `cd web && npx vitest run test/composer.test.tsx`. Expect failure: `Failed to resolve import "@/components/chat/composer"`.

- [ ] **Step 8: Implement `Composer`.** Create `web/components/chat/composer.tsx`. A native `<textarea>` + native `<select>` keep the component testable with jsdom and free of AI-Elements internal portal quirks while still following the PromptInput submit-status contract (`status` drives the submit/stop button):

```tsx
// web/components/chat/composer.tsx
'use client';

import { type KeyboardEvent, useState } from 'react';
import { Loader2, Send, Square } from 'lucide-react';
import { groupModelsByProvider } from '@/lib/use-models';

export interface ComposerProps {
  model: string;
  models: string[];
  onModelChange: (model: string) => void;
  onSubmit: (text: string) => void;
  onStop: () => void;
  status: 'ready' | 'streaming' | 'submitted' | 'error';
  disabled?: boolean;
}

export function Composer({ model, models, onModelChange, onSubmit, onStop, status, disabled }: ComposerProps) {
  const [value, setValue] = useState('');
  const isBusy = status === 'streaming' || status === 'submitted';
  const groups = groupModelsByProvider(models);

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || isBusy || disabled) {
      return;
    }
    onSubmit(trimmed);
    setValue('');
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const onButtonClick = () => {
    if (isBusy) {
      onStop();
    } else {
      submit();
    }
  };

  return (
    <div className="flex flex-col gap-2 border-t border-border bg-background p-3">
      <div className="flex items-center gap-2">
        <label htmlFor="composer-model" className="sr-only">
          Модел
        </label>
        <select
          id="composer-model"
          data-testid="composer-model"
          value={model}
          onChange={(e) => onModelChange(e.target.value)}
          disabled={disabled}
          className="rounded-md border border-border bg-background px-2 py-1 text-sm"
        >
          {groups.map((g) => (
            <optgroup key={g.provider} label={g.provider}>
              {g.models.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      </div>
      <div className="flex items-end gap-2">
        <textarea
          aria-label="Порака"
          data-testid="composer-input"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKeyDown}
          disabled={disabled}
          rows={1}
          placeholder="Напиши порака…"
          className="min-h-[44px] flex-1 resize-none rounded-md border border-border bg-background px-3 py-2 text-sm"
        />
        <button
          type="button"
          data-testid="composer-submit"
          onClick={onButtonClick}
          disabled={disabled || (!isBusy && value.trim().length === 0)}
          aria-label={isBusy ? 'Запри' : 'Испрати'}
          className="inline-flex size-10 items-center justify-center rounded-md border border-border hover:bg-muted disabled:opacity-50"
        >
          {status === 'submitted' ? (
            <Loader2 className="size-4 animate-spin" aria-hidden="true" />
          ) : isBusy ? (
            <Square className="size-4" aria-hidden="true" />
          ) : (
            <Send className="size-4" aria-hidden="true" />
          )}
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 9: Run it, expect PASS.** Run `cd web && npx vitest run test/composer.test.tsx`. Expect all `Composer` tests to pass.

- [ ] **Step 10: Typecheck.** Run `cd web && npx tsc --noEmit`. Expect no errors.

- [ ] **Step 11: Commit.**
```bash
git add web/components/chat/composer.tsx web/test/composer.test.tsx
git commit -m "feat(web): add Composer with Enter/Shift-Enter, stop status, and model picker"
```

### Task 13: App shell + sidebar + persistence wiring

**Files:**
- Create: `web/components/shell/sidebar.tsx`, `web/components/shell/conversation-list.tsx`, `web/lib/ui-store.ts`
- Modify: `web/app/page.tsx`
- Test: `web/test/shell.test.tsx`

**Interfaces:**

Consumes (Task 8, `@/lib/db`):
- `createConversation(input: { title: string; model: string }): Promise<ConversationRow>`
- `listConversations(): Promise<ConversationRow[]>` (newest `updatedAt` first)
- `renameConversation(id: string, title: string): Promise<void>`
- `deleteConversation(id: string): Promise<void>`
- `loadMessages(conversationId: string): Promise<MessageRow[]>`
- `saveMessages(conversationId: string, messages: MyUIMessage[]): Promise<void>`
- types `ConversationRow { id; title; model; createdAt; updatedAt }`, `MessageRow { id; conversationId; role; parts; metadata?; createdAt }`

Consumes (Task 10, `@/lib/transport`): `buildChatTransport(): DefaultChatTransport<MyUIMessage>`.

Consumes (Task 9, `@/lib/messages`): `deriveTitle(firstUserText: string): string`. Consumes (`@/lib/user`): `getAnonUserId(): string`.

Consumes (Task 11): `Thread`. Consumes (Task 12): `Composer`, `useModels`.

Consumes (`@ai-sdk/react`): `useChat`.

Produces (Task 14/16 rely on these):
- `useUiStore` — Zustand store: `export const useUiStore` with state `{ activeConversationId: string | null; model: string; sidebarOpen: boolean; setActiveConversationId(id: string | null): void; setModel(model: string): void; toggleSidebar(): void; setSidebarOpen(open: boolean): void }` (`web/lib/ui-store.ts`)
- `ConversationList` — `export function ConversationList(props: { conversations: ConversationRow[]; activeId: string | null; onSelect(id: string): void; onRename(id: string, title: string): void; onDelete(id: string): void }): JSX.Element` (`web/components/shell/conversation-list.tsx`)
- `Sidebar` — `export function Sidebar(props: { conversations: ConversationRow[]; activeId: string | null; open: boolean; onNewChat(): void; onSelect(id: string): void; onRename(id: string, title: string): void; onDelete(id: string): void }): JSX.Element` (`web/components/shell/sidebar.tsx`)

UIMessage⇄MessageRow mapping helpers live in `page.tsx` (private). The Dexie default model id used for a fresh conversation is `'claude-sonnet-4-6'` (matches the API default `inference_model`).

Steps:

- [ ] **Step 1: Write the failing test for `useUiStore`.** Create `web/test/shell.test.tsx`:

```tsx
// web/test/shell.test.tsx
import { act } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { useUiStore } from '@/lib/ui-store';

describe('useUiStore', () => {
  beforeEach(() => {
    useUiStore.setState({ activeConversationId: null, model: 'claude-sonnet-4-6', sidebarOpen: true });
  });

  it('has sane defaults', () => {
    const s = useUiStore.getState();
    expect(s.model).toBe('claude-sonnet-4-6');
    expect(s.sidebarOpen).toBe(true);
    expect(s.activeConversationId).toBeNull();
  });

  it('updates active conversation and model', () => {
    act(() => {
      useUiStore.getState().setActiveConversationId('c1');
      useUiStore.getState().setModel('gpt-5.4-mini');
    });
    expect(useUiStore.getState().activeConversationId).toBe('c1');
    expect(useUiStore.getState().model).toBe('gpt-5.4-mini');
  });

  it('toggles the sidebar', () => {
    act(() => useUiStore.getState().toggleSidebar());
    expect(useUiStore.getState().sidebarOpen).toBe(false);
    act(() => useUiStore.getState().setSidebarOpen(true));
    expect(useUiStore.getState().sidebarOpen).toBe(true);
  });
});
```

- [ ] **Step 2: Run it, expect FAIL.** Run `cd web && npx vitest run test/shell.test.tsx`. Expect failure: `Failed to resolve import "@/lib/ui-store"`.

- [ ] **Step 3: Implement `useUiStore`.** Create `web/lib/ui-store.ts`:

```ts
// web/lib/ui-store.ts
'use client';

import { create } from 'zustand';

export interface UiState {
  activeConversationId: string | null;
  model: string;
  sidebarOpen: boolean;
  setActiveConversationId: (id: string | null) => void;
  setModel: (model: string) => void;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
}

export const useUiStore = create<UiState>((set) => ({
  activeConversationId: null,
  model: 'claude-sonnet-4-6',
  sidebarOpen: true,
  setActiveConversationId: (id) => set({ activeConversationId: id }),
  setModel: (model) => set({ model }),
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),
}));
```

- [ ] **Step 4: Run it, expect PASS.** Run `cd web && npx vitest run test/shell.test.tsx`. Expect the three `useUiStore` tests to pass.

- [ ] **Step 5: Commit.**
```bash
git add web/lib/ui-store.ts web/test/shell.test.tsx
git commit -m "feat(web): add Zustand ui-store for active conversation, model, and sidebar"
```

- [ ] **Step 6: Write the failing test for `ConversationList`.** Append to `web/test/shell.test.tsx`:

```tsx
import { fireEvent, render, screen, within } from '@testing-library/react';
import { ConversationList } from '@/components/shell/conversation-list';
import type { ConversationRow } from '@/lib/db';

const rows: ConversationRow[] = [
  { id: 'c1', title: 'Прв разговор', model: 'claude-sonnet-4-6', createdAt: 1, updatedAt: 2 },
  { id: 'c2', title: 'Втор разговор', model: 'gpt-5.4-mini', createdAt: 3, updatedAt: 4 },
];

describe('ConversationList', () => {
  it('lists conversations and marks the active one', () => {
    render(
      <ConversationList conversations={rows} activeId="c2" onSelect={vi.fn()} onRename={vi.fn()} onDelete={vi.fn()} />,
    );
    expect(screen.getByText('Прв разговор')).toBeInTheDocument();
    expect(screen.getByTestId('conversation-c2')).toHaveAttribute('aria-current', 'true');
  });

  it('selects a conversation on click', () => {
    const onSelect = vi.fn();
    render(
      <ConversationList conversations={rows} activeId={null} onSelect={onSelect} onRename={vi.fn()} onDelete={vi.fn()} />,
    );
    fireEvent.click(screen.getByText('Прв разговор'));
    expect(onSelect).toHaveBeenCalledWith('c1');
  });

  it('deletes a conversation', () => {
    const onDelete = vi.fn();
    render(
      <ConversationList conversations={rows} activeId={null} onSelect={vi.fn()} onRename={vi.fn()} onDelete={onDelete} />,
    );
    const item = screen.getByTestId('conversation-c1');
    fireEvent.click(within(item).getByRole('button', { name: 'Избриши' }));
    expect(onDelete).toHaveBeenCalledWith('c1');
  });

  it('renames a conversation via prompt', () => {
    const onRename = vi.fn();
    vi.spyOn(window, 'prompt').mockReturnValue('Ново име');
    render(
      <ConversationList conversations={rows} activeId={null} onSelect={vi.fn()} onRename={onRename} onDelete={vi.fn()} />,
    );
    const item = screen.getByTestId('conversation-c1');
    fireEvent.click(within(item).getByRole('button', { name: 'Преименувај' }));
    expect(onRename).toHaveBeenCalledWith('c1', 'Ново име');
  });
});
```

- [ ] **Step 7: Run it, expect FAIL.** Run `cd web && npx vitest run test/shell.test.tsx`. Expect failure: `Failed to resolve import "@/components/shell/conversation-list"`.

- [ ] **Step 8: Implement `ConversationList`.** Create `web/components/shell/conversation-list.tsx`:

```tsx
// web/components/shell/conversation-list.tsx
'use client';

import { Pencil, Trash2 } from 'lucide-react';
import type { ConversationRow } from '@/lib/db';

export interface ConversationListProps {
  conversations: ConversationRow[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
}

export function ConversationList({ conversations, activeId, onSelect, onRename, onDelete }: ConversationListProps) {
  return (
    <ul className="flex flex-col gap-1" role="list">
      {conversations.map((c) => (
        <li
          key={c.id}
          data-testid={`conversation-${c.id}`}
          aria-current={c.id === activeId ? 'true' : undefined}
          className={`group flex items-center justify-between rounded-md px-2 py-1.5 text-sm hover:bg-muted ${
            c.id === activeId ? 'bg-muted font-medium' : ''
          }`}
        >
          <button type="button" onClick={() => onSelect(c.id)} className="flex-1 truncate text-left">
            {c.title}
          </button>
          <span className="flex items-center gap-1 opacity-0 group-hover:opacity-100">
            <button
              type="button"
              aria-label="Преименувај"
              onClick={() => {
                const next = window.prompt('Ново име на разговорот', c.title);
                if (next && next.trim()) {
                  onRename(c.id, next.trim());
                }
              }}
              className="rounded p-1 hover:bg-background"
            >
              <Pencil className="size-3.5" aria-hidden="true" />
            </button>
            <button
              type="button"
              aria-label="Избриши"
              onClick={() => onDelete(c.id)}
              className="rounded p-1 hover:bg-background"
            >
              <Trash2 className="size-3.5" aria-hidden="true" />
            </button>
          </span>
        </li>
      ))}
    </ul>
  );
}
```

- [ ] **Step 9: Run it, expect PASS.** Run `cd web && npx vitest run test/shell.test.tsx`. Expect the four `ConversationList` tests to pass.

- [ ] **Step 10: Commit.**
```bash
git add web/components/shell/conversation-list.tsx web/test/shell.test.tsx
git commit -m "feat(web): add ConversationList with select, rename, and delete"
```

- [ ] **Step 11: Write the failing test for `Sidebar`.** Append to `web/test/shell.test.tsx`:

```tsx
import { Sidebar } from '@/components/shell/sidebar';

describe('Sidebar', () => {
  it('renders the new-chat button and the conversation list when open', () => {
    const onNewChat = vi.fn();
    render(
      <Sidebar
        conversations={rows}
        activeId="c1"
        open
        onNewChat={onNewChat}
        onSelect={vi.fn()}
        onRename={vi.fn()}
        onDelete={vi.fn()}
      />,
    );
    expect(screen.getByText('Прв разговор')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Нов разговор' }));
    expect(onNewChat).toHaveBeenCalledOnce();
  });

  it('hides its content when collapsed', () => {
    render(
      <Sidebar
        conversations={rows}
        activeId={null}
        open={false}
        onNewChat={vi.fn()}
        onSelect={vi.fn()}
        onRename={vi.fn()}
        onDelete={vi.fn()}
      />,
    );
    expect(screen.queryByText('Прв разговор')).not.toBeInTheDocument();
  });
});
```

- [ ] **Step 12: Run it, expect FAIL.** Run `cd web && npx vitest run test/shell.test.tsx`. Expect failure: `Failed to resolve import "@/components/shell/sidebar"`.

- [ ] **Step 13: Implement `Sidebar`.** Create `web/components/shell/sidebar.tsx`:

```tsx
// web/components/shell/sidebar.tsx
'use client';

import { Plus } from 'lucide-react';
import { ConversationList } from '@/components/shell/conversation-list';
import type { ConversationRow } from '@/lib/db';

export interface SidebarProps {
  conversations: ConversationRow[];
  activeId: string | null;
  open: boolean;
  onNewChat: () => void;
  onSelect: (id: string) => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
}

export function Sidebar({ conversations, activeId, open, onNewChat, onSelect, onRename, onDelete }: SidebarProps) {
  if (!open) {
    return <aside aria-label="Странична лента" data-collapsed="true" className="w-0 overflow-hidden" />;
  }
  return (
    <aside aria-label="Странична лента" className="flex w-64 shrink-0 flex-col gap-3 border-r border-border bg-muted/30 p-3">
      <button
        type="button"
        onClick={onNewChat}
        className="inline-flex items-center justify-center gap-2 rounded-md border border-border bg-background px-3 py-2 text-sm font-medium hover:bg-muted"
      >
        <Plus className="size-4" aria-hidden="true" />
        Нов разговор
      </button>
      <nav className="flex-1 overflow-y-auto">
        <ConversationList
          conversations={conversations}
          activeId={activeId}
          onSelect={onSelect}
          onRename={onRename}
          onDelete={onDelete}
        />
      </nav>
    </aside>
  );
}
```

- [ ] **Step 14: Run it, expect PASS.** Run `cd web && npx vitest run test/shell.test.tsx`. Expect the two `Sidebar` tests to pass.

- [ ] **Step 15: Commit.**
```bash
git add web/components/shell/sidebar.tsx web/test/shell.test.tsx
git commit -m "feat(web): add collapsible Sidebar with new-chat and conversation list"
```

- [ ] **Step 16: Write the failing integration test for `page.tsx` (new-chat + persist + hydrate).** Append to `web/test/shell.test.tsx`. This drives the real `useChat` via `buildChatTransport`, but we mock `fetch` so `/api/chat` streams one assistant token then `done`, and assert the answer renders, a Dexie conversation is created, and re-mounting hydrates it:

```tsx
import { render as rtlRender, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach } from 'vitest';
import ChatPage from '@/app/page';
import { db } from '@/lib/db';

function sseChatResponse(): Response {
  // The BFF (Task 5) returns an AI SDK *UI message stream*, NOT raw protocol-v2.
  // useChat/DefaultChatTransport only parses this format, so the mock must emit it:
  // start(messageMetadata) -> text-start -> text-delta{delta} -> text-end -> finish -> [DONE].
  // (responseId reaches the client via the start chunk's messageMetadata, not the header.)
  const chunks = [
    { type: 'start', messageMetadata: { responseId: 'resp-123', inferenceModel: 'claude-sonnet-4-6' } },
    { type: 'text-start', id: 'txt-1' },
    { type: 'text-delta', id: 'txt-1', delta: 'Здраво!' },
    { type: 'text-end', id: 'txt-1' },
    { type: 'finish' },
  ];
  const body = `${chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join('')}data: [DONE]\n\n`;
  return new Response(body, {
    status: 200,
    headers: {
      'content-type': 'text/event-stream',
      'x-vercel-ai-ui-message-stream': 'v1',
      'X-Response-Id': 'resp-123',
    },
  });
}

beforeEach(async () => {
  await db.delete();
  await db.open();
  useUiStore.setState({ activeConversationId: null, model: 'claude-sonnet-4-6', sidebarOpen: true });
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.endsWith('/api/models')) {
        return new Response(JSON.stringify(['claude-sonnet-4-6', 'gpt-5.4-mini']), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        });
      }
      if (url.endsWith('/api/chat')) {
        return sseChatResponse();
      }
      return new Response('{}', { status: 200, headers: { 'content-type': 'application/json' } });
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('ChatPage persistence', () => {
  it('sends a message, renders the streamed answer, and persists to Dexie', async () => {
    const user = userEvent.setup();
    rtlRender(<ChatPage />);

    await user.type(screen.getByRole('textbox'), 'Прашање?');
    await user.keyboard('{Enter}');

    expect(await screen.findByText('Прашање?')).toBeInTheDocument();
    expect(await screen.findByText('Здраво!')).toBeInTheDocument();

    await waitFor(async () => {
      const convos = await db.conversations.toArray();
      expect(convos).toHaveLength(1);
    });
    const convos = await db.conversations.toArray();
    const msgs = await db.messages.where('conversationId').equals(convos[0].id).toArray();
    expect(msgs.some((m) => m.role === 'user')).toBe(true);
    expect(msgs.some((m) => m.role === 'assistant')).toBe(true);
    expect(msgs.find((m) => m.role === 'assistant')?.metadata?.responseId).toBe('resp-123');
  });

  it('hydrates an existing conversation on mount', async () => {
    const now = Date.now();
    await db.conversations.put({ id: 'cX', title: 'Стар разговор', model: 'claude-sonnet-4-6', createdAt: now, updatedAt: now });
    await db.messages.bulkPut([
      { id: 'mU', conversationId: 'cX', role: 'user', parts: [{ type: 'text', text: 'Старо прашање' }], createdAt: now },
      { id: 'mA', conversationId: 'cX', role: 'assistant', parts: [{ type: 'text', text: 'Стар одговор' }], metadata: { responseId: 'r-old' }, createdAt: now + 1 },
    ]);
    useUiStore.setState({ activeConversationId: 'cX', model: 'claude-sonnet-4-6', sidebarOpen: true });

    rtlRender(<ChatPage />);
    expect(await screen.findByText('Стар одговор')).toBeInTheDocument();
  });
});
```

- [ ] **Step 17: Run it, expect FAIL.** Run `cd web && npx vitest run test/shell.test.tsx`. Expect failure resolving `@/app/page` as a usable component (the placeholder from Task 1 has no chat wiring) — the assertions on streamed text / Dexie rows fail.

- [ ] **Step 18: Implement `web/app/page.tsx` (full chat screen).** Replace the placeholder. It wires `useChat` (transport from Task 10), TanStack Query for models, Dexie hydrate/persist, and the shell. Conversation creation is lazy (on first user send); persistence runs on `onFinish` and on user-send:

```tsx
// web/app/page.tsx
'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useChat } from '@ai-sdk/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PanelLeft } from 'lucide-react';
import type { MyUIMessage } from '@/lib/api-types';
import { buildChatTransport } from '@/lib/transport';
import {
  createConversation,
  deleteConversation,
  listConversations,
  loadMessages,
  renameConversation,
  saveMessages,
  type ConversationRow,
  type MessageRow,
} from '@/lib/db';
import { deriveTitle } from '@/lib/messages';
import { useModels } from '@/lib/use-models';
import { useUiStore } from '@/lib/ui-store';
import { Sidebar } from '@/components/shell/sidebar';
import { Thread } from '@/components/chat/thread';
import { Composer } from '@/components/chat/composer';

const queryClient = new QueryClient();

function toRow(message: MyUIMessage, conversationId: string, createdAt: number): MessageRow {
  return {
    id: message.id,
    conversationId,
    role: message.role,
    parts: message.parts,
    metadata: message.metadata,
    createdAt,
  };
}

function fromRow(row: MessageRow): MyUIMessage {
  return { id: row.id, role: row.role, parts: row.parts, metadata: row.metadata } as MyUIMessage;
}

function ChatScreen() {
  const activeId = useUiStore((s) => s.activeConversationId);
  const setActiveId = useUiStore((s) => s.setActiveConversationId);
  const model = useUiStore((s) => s.model);
  const setModel = useUiStore((s) => s.setModel);
  const sidebarOpen = useUiStore((s) => s.sidebarOpen);
  const toggleSidebar = useUiStore((s) => s.toggleSidebar);

  const { data: modelList } = useModels();
  const [conversations, setConversations] = useState<ConversationRow[]>([]);
  const [activeStatus, setActiveStatus] = useState<{ label: string; tool?: string } | undefined>();
  const [activeError, setActiveError] = useState<{ code: string; message: string } | undefined>();
  const convoIdRef = useRef<string | null>(activeId);

  const refreshConversations = useCallback(async () => {
    setConversations(await listConversations());
  }, []);

  const { messages, sendMessage, setMessages, status, stop, regenerate } = useChat<MyUIMessage>({
    transport: buildChatTransport(() => ({ model })),
    onData: (part) => {
      if (part.type === 'data-status') {
        setActiveStatus(part.data);
      }
      if (part.type === 'data-error') {
        setActiveError(part.data);
      }
    },
    onFinish: async ({ message }) => {
      setActiveStatus(undefined);
      const cid = convoIdRef.current;
      if (!cid) {
        return;
      }
      await saveMessages(cid, [message]);
      // Title was already derived from the first user turn at creation (handleSubmit);
      // re-deriving here off a stale `conversations` closure could blank it — so don't rename.
      await refreshConversations();
    },
  });

  useEffect(() => {
    void refreshConversations();
  }, [refreshConversations]);

  // Hydrate when the active conversation changes.
  useEffect(() => {
    convoIdRef.current = activeId;
    setActiveError(undefined);
    setActiveStatus(undefined);
    if (!activeId) {
      setMessages([]);
      return;
    }
    let cancelled = false;
    void loadMessages(activeId).then((rows) => {
      if (!cancelled) {
        setMessages(rows.map(fromRow));
      }
    });
    return () => {
      cancelled = true;
    };
  }, [activeId, setMessages]);

  const handleNewChat = useCallback(() => {
    setActiveId(null);
    setMessages([]);
    setActiveError(undefined);
    convoIdRef.current = null;
  }, [setActiveId, setMessages]);

  const handleSubmit = useCallback(
    async (text: string) => {
      setActiveError(undefined);
      let cid = convoIdRef.current;
      if (!cid) {
        const convo = await createConversation({ title: deriveTitle(text), model });
        cid = convo.id;
        convoIdRef.current = cid;
        setActiveId(cid);
        await refreshConversations();
      }
      const userMessage: MyUIMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        parts: [{ type: 'text', text }],
        metadata: {},
      };
      await saveMessages(cid, [userMessage]);
      sendMessage(userMessage);
    },
    [model, refreshConversations, sendMessage, setActiveId],
  );

  const handleSelect = useCallback((id: string) => setActiveId(id), [setActiveId]);

  const handleDelete = useCallback(
    async (id: string) => {
      await deleteConversation(id);
      if (convoIdRef.current === id) {
        handleNewChat();
      }
      await refreshConversations();
    },
    [handleNewChat, refreshConversations],
  );

  const handleRename = useCallback(
    async (id: string, title: string) => {
      await renameConversation(id, title);
      await refreshConversations();
    },
    [refreshConversations],
  );

  return (
    <div className="flex h-dvh w-full">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        open={sidebarOpen}
        onNewChat={handleNewChat}
        onSelect={handleSelect}
        onRename={handleRename}
        onDelete={handleDelete}
      />
      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-2 border-b border-border p-2">
          <button type="button" aria-label="Прикажи/сокриј странична лента" onClick={toggleSidebar} className="rounded p-1.5 hover:bg-muted">
            <PanelLeft className="size-4" aria-hidden="true" />
          </button>
          <span className="text-sm font-medium">ФИНКИ Хаб</span>
        </header>
        <Thread
          messages={messages}
          status={status}
          activeStatus={activeStatus}
          activeError={activeError}
          onRetry={() => regenerate()}
        />
        <Composer
          model={model}
          models={modelList ?? []}
          onModelChange={setModel}
          onSubmit={(text) => void handleSubmit(text)}
          onStop={stop}
          status={status}
        />
      </main>
    </div>
  );
}

export default function ChatPage() {
  return (
    <QueryClientProvider client={queryClient}>
      <ChatScreen />
    </QueryClientProvider>
  );
}
```

- [ ] **Step 19: Run the full file, expect PASS.** Run `cd web && npx vitest run test/shell.test.tsx`. Expect all `useUiStore`, `ConversationList`, `Sidebar`, and `ChatPage persistence` tests green. If the streamed-answer assertion races, the test already uses `findByText` (async) and `waitFor` to absorb the stream timing.

- [ ] **Step 20: Typecheck.** Run `cd web && npx tsc --noEmit`. Expect no errors.

- [ ] **Step 21: Commit.**
```bash
git add web/app/page.tsx web/test/shell.test.tsx
git commit -m "feat(web): wire app shell, Dexie persistence, and useChat hydration in page"
```

### Task 14: Answer actions (copy / regenerate / like-dislike)

**Files:**
- Create: `web/components/chat/answer-actions.tsx`
- Test: `web/test/answer-actions.test.tsx`

**Interfaces:**

Consumes (Task 7, BFF): `POST /api/feedback` with JSON body `{ responseId, feedbackType, userId, questionText?, answerText?, inferenceModel? }` (shape `FeedbackClientPayload` from `@/lib/api-types`) → returns `FeedbackAck`.

Consumes (Task 9, `@/lib/user`): `getAnonUserId(): string`.

Consumes (Task 3, `@/lib/api-types`): `type MyUIMessage`, `type FeedbackType = 'like' | 'dislike'`.

Produces (Task 13's `renderActions` and Task 16 rely on this):
- `AnswerActions` — `export function AnswerActions(props: { message: MyUIMessage; questionText?: string }): JSX.Element | null` (`web/components/chat/answer-actions.tsx`). Returns `null` when `message.metadata?.responseId` is absent (hides feedback for that turn, spec §9). Renders Copy + Regenerate + Like + Dislike. Regenerate is supplied via the same prop surface: it takes an optional `onRegenerate?: () => void`.

Refined signature used below:
- `AnswerActions(props: { message: MyUIMessage; questionText?: string; onRegenerate?: () => void }): JSX.Element | null`

Steps:

- [ ] **Step 1: Write the failing test for the hidden case + copy.** Create `web/test/answer-actions.test.tsx`:

```tsx
// web/test/answer-actions.test.tsx
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AnswerActions } from '@/components/chat/answer-actions';
import type { MyUIMessage } from '@/lib/api-types';

function assistant(responseId?: string, text = 'Одговор'): MyUIMessage {
  return {
    id: 'a1',
    role: 'assistant',
    parts: [{ type: 'text', text }],
    metadata: responseId ? { responseId, inferenceModel: 'claude-sonnet-4-6' } : {},
  };
}

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(JSON.stringify({ id: 1, response_id: 'r1', feedback_type: 'like' }), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    })),
  );
  Object.assign(navigator, { clipboard: { writeText: vi.fn(async () => undefined) } });
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('AnswerActions', () => {
  it('renders nothing when there is no responseId', () => {
    const { container } = render(<AnswerActions message={assistant(undefined)} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('copies the answer text to the clipboard', async () => {
    render(<AnswerActions message={assistant('r1', 'Текст за копирање')} />);
    fireEvent.click(screen.getByRole('button', { name: 'Копирај' }));
    await waitFor(() => expect(navigator.clipboard.writeText).toHaveBeenCalledWith('Текст за копирање'));
  });
});
```

- [ ] **Step 2: Run it, expect FAIL.** Run `cd web && npx vitest run test/answer-actions.test.tsx`. Expect failure: `Failed to resolve import "@/components/chat/answer-actions"`.

- [ ] **Step 3: Implement `AnswerActions` (copy + hidden case only first).** Create `web/components/chat/answer-actions.tsx`:

```tsx
// web/components/chat/answer-actions.tsx
'use client';

import { useState } from 'react';
import { Check, Copy, RotateCcw, ThumbsDown, ThumbsUp } from 'lucide-react';
import type { FeedbackType, MyUIMessage } from '@/lib/api-types';
import { getAnonUserId } from '@/lib/user';

export interface AnswerActionsProps {
  message: MyUIMessage;
  questionText?: string;
  onRegenerate?: () => void;
}

function answerText(message: MyUIMessage): string {
  const texts = message.parts.filter((p): p is { type: 'text'; text: string } => p.type === 'text');
  return texts.at(-1)?.text ?? '';
}

export function AnswerActions({ message, questionText, onRegenerate }: AnswerActionsProps) {
  const responseId = message.metadata?.responseId;
  const [copied, setCopied] = useState(false);
  const [vote, setVote] = useState<FeedbackType | null>(null);

  if (!responseId) {
    return null;
  }

  const text = answerText(message);

  const copy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  };

  const sendFeedback = async (feedbackType: FeedbackType) => {
    const previous = vote;
    setVote(feedbackType); // optimistic toggle
    try {
      const res = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          responseId,
          feedbackType,
          userId: getAnonUserId(),
          questionText,
          answerText: text,
          inferenceModel: message.metadata?.inferenceModel,
        }),
      });
      if (!res.ok) {
        setVote(previous);
      }
    } catch {
      setVote(previous);
    }
  };

  const btn = 'inline-flex items-center justify-center rounded-md p-1.5 text-muted-foreground hover:bg-muted';

  return (
    <div className="flex items-center gap-1" data-testid="answer-actions">
      <button type="button" aria-label="Копирај" onClick={() => void copy()} className={btn}>
        {copied ? <Check className="size-4" aria-hidden="true" /> : <Copy className="size-4" aria-hidden="true" />}
      </button>
      {onRegenerate ? (
        <button type="button" aria-label="Регенерирај" onClick={onRegenerate} className={btn}>
          <RotateCcw className="size-4" aria-hidden="true" />
        </button>
      ) : null}
      <button
        type="button"
        data-testid="like-button"
        aria-label="Допаѓа"
        aria-pressed={vote === 'like'}
        onClick={() => void sendFeedback('like')}
        className={`${btn} ${vote === 'like' ? 'text-green-600' : ''}`}
      >
        <ThumbsUp className="size-4" aria-hidden="true" />
      </button>
      <button
        type="button"
        aria-label="Не допаѓа"
        aria-pressed={vote === 'dislike'}
        onClick={() => void sendFeedback('dislike')}
        className={`${btn} ${vote === 'dislike' ? 'text-red-600' : ''}`}
      >
        <ThumbsDown className="size-4" aria-hidden="true" />
      </button>
    </div>
  );
}
```

- [ ] **Step 4: Run it, expect PASS.** Run `cd web && npx vitest run test/answer-actions.test.tsx`. Expect the hidden-case + copy tests to pass.

- [ ] **Step 5: Commit.**
```bash
git add web/components/chat/answer-actions.tsx web/test/answer-actions.test.tsx
git commit -m "feat(web): add AnswerActions with copy and hidden-when-no-responseId"
```

- [ ] **Step 6: Write the failing test for feedback + regenerate.** Append to `web/test/answer-actions.test.tsx`:

```tsx
describe('AnswerActions feedback', () => {
  it('posts a like to /api/feedback with the anon user id and metadata', async () => {
    localStorage.setItem('finkiHub.anonUserId', 'user-xyz');
    render(<AnswerActions message={assistant('resp-9', 'Готов одговор')} questionText="Прашање?" />);
    fireEvent.click(screen.getByRole('button', { name: 'Допаѓа' }));

    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));
    const [url, init] = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(url).toBe('/api/feedback');
    const body = JSON.parse((init as RequestInit).body as string);
    expect(body).toMatchObject({
      responseId: 'resp-9',
      feedbackType: 'like',
      userId: 'user-xyz',
      questionText: 'Прашање?',
      answerText: 'Готов одговор',
      inferenceModel: 'claude-sonnet-4-6',
    });
  });

  it('optimistically marks the chosen vote as pressed', async () => {
    render(<AnswerActions message={assistant('resp-9')} />);
    const dislike = screen.getByRole('button', { name: 'Не допаѓа' });
    fireEvent.click(dislike);
    await waitFor(() => expect(dislike).toHaveAttribute('aria-pressed', 'true'));
  });

  it('reverts the optimistic vote when the request fails', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status: 500 })));
    render(<AnswerActions message={assistant('resp-9')} />);
    const like = screen.getByRole('button', { name: 'Допаѓа' });
    fireEvent.click(like);
    await waitFor(() => expect(like).toHaveAttribute('aria-pressed', 'false'));
  });

  it('invokes onRegenerate when Регенерирај is clicked', () => {
    const onRegenerate = vi.fn();
    render(<AnswerActions message={assistant('resp-9')} onRegenerate={onRegenerate} />);
    fireEvent.click(screen.getByRole('button', { name: 'Регенерирај' }));
    expect(onRegenerate).toHaveBeenCalledOnce();
  });
});
```

- [ ] **Step 7: Run it, expect PASS.** Run `cd web && npx vitest run test/answer-actions.test.tsx`. The implementation from Step 3 already covers feedback, optimistic toggle, revert, and regenerate, so these pass without further code. If `getAnonUserId()` reads a different localStorage key than `finkiHub.anonUserId`, align the test's `localStorage.setItem` key with the one defined in Task 9 (`@/lib/user`) — the canonical key is `finkiHub.anonUserId` per spec §7.

- [ ] **Step 8: Typecheck.** Run `cd web && npx tsc --noEmit`. Expect no errors.

- [ ] **Step 9: Wire `AnswerActions` into the thread via `page.tsx` `renderActions`.** Modify `web/app/page.tsx` to pass actions for finished assistant turns. Add the import and the `renderActions` prop on `<Thread>`:

```tsx
// add to the imports in web/app/page.tsx
import { AnswerActions } from '@/components/chat/answer-actions';
```

Then change the `<Thread ... />` usage to include `renderActions` (replace the existing `<Thread .../>` block):

```tsx
        <Thread
          messages={messages}
          status={status}
          activeStatus={activeStatus}
          activeError={activeError}
          onRetry={() => regenerate()}
          renderActions={(m) =>
            m.role === 'assistant' && status !== 'streaming' ? (
              <AnswerActions
                message={m}
                onRegenerate={() => regenerate({ messageId: m.id })}
                questionText={messages
                  .slice(0, messages.indexOf(m))
                  .filter((x) => x.role === 'user')
                  .at(-1)
                  ?.parts.filter((p): p is { type: 'text'; text: string } => p.type === 'text')
                  .map((p) => p.text)
                  .join('')}
              />
            ) : null
          }
        />
```

- [ ] **Step 10: Re-run the shell integration test to confirm no regression.** Run `cd web && npx vitest run test/shell.test.tsx test/answer-actions.test.tsx`. Expect both files green (the shell streamed-answer test still passes; `AnswerActions` appears only once `done` arrives and `status !== 'streaming'`).

- [ ] **Step 11: Typecheck + commit.**
```bash
git add web/app/page.tsx web/components/chat/answer-actions.tsx web/test/answer-actions.test.tsx
git commit -m "feat(web): wire AnswerActions (copy/regenerate/feedback) into the thread"
```

### Task 15: Macedonian chrome strings (i18n)

**Files:**
- Create: `web/lib/i18n.ts`
- Modify: `web/components/chat/thread.tsx`, `web/components/chat/composer.tsx`, `web/components/shell/sidebar.tsx`, `web/components/chat/message.tsx`, `web/components/shell/conversation-list.tsx`
- Test: `web/test/i18n.test.ts`

**Interfaces:**

Produces:
- `messages` — `export const messages` (flat `Record<string, string>` of Macedonian Cyrillic UI strings) (`web/lib/i18n.ts`)
- `t` — `export function t(key: TKey): string` where `TKey = keyof typeof messages` (`web/lib/i18n.ts`)

Consumes: all earlier component literals are swapped to `t('...')` calls so the chrome is centralized (spec §2 / §6 — Macedonian default, structured so EN can be added later). The swaps must preserve the exact same visible strings the Task 11/12/13 tests already assert (e.g. `Започни разговор`, `Нов разговор`, `Обиди се повторно`, `Одговорот е прекинат.`), so those tests stay green.

The canonical key→value map:

| key | value |
|---|---|
| `app.title` | `ФИНКИ Хаб` |
| `sidebar.new` | `Нов разговор` |
| `sidebar.label` | `Странична лента` |
| `sidebar.toggle` | `Прикажи/сокриј странична лента` |
| `conversation.rename` | `Преименувај` |
| `conversation.delete` | `Избриши` |
| `conversation.renamePrompt` | `Ново име на разговорот` |
| `thread.emptyTitle` | `Започни разговор` |
| `thread.emptyDescription` | `Прашај нешто за студиите на ФИНКИ.` |
| `composer.placeholder` | `Напиши порака…` |
| `composer.message` | `Порака` |
| `composer.model` | `Модел` |
| `composer.send` | `Испрати` |
| `composer.stop` | `Запри` |
| `error.retry` | `Обиди се повторно` |
| `error.interrupted` | `Одговорот е прекинат.` |
| `actions.copy` | `Копирај` |
| `actions.regenerate` | `Регенерирај` |
| `actions.like` | `Допаѓа` |
| `actions.dislike` | `Не допаѓа` |

Steps:

- [ ] **Step 1: Write the failing test for `i18n`.** Create `web/test/i18n.test.ts`:

```ts
// web/test/i18n.test.ts
import { describe, expect, it } from 'vitest';
import { messages, t } from '@/lib/i18n';

describe('i18n', () => {
  it('returns the Macedonian string for a known key', () => {
    expect(t('sidebar.new')).toBe('Нов разговор');
    expect(t('thread.emptyTitle')).toBe('Започни разговор');
    expect(t('error.retry')).toBe('Обиди се повторно');
    expect(t('error.interrupted')).toBe('Одговорот е прекинат.');
  });

  it('exposes the flat dictionary', () => {
    expect(messages['actions.copy']).toBe('Копирај');
    expect(messages['app.title']).toBe('ФИНКИ Хаб');
  });

  it('covers every documented key with non-empty Cyrillic-or-symbol values', () => {
    const keys = Object.keys(messages);
    expect(keys.length).toBeGreaterThanOrEqual(20);
    for (const key of keys) {
      expect(messages[key as keyof typeof messages].length).toBeGreaterThan(0);
    }
  });
});
```

- [ ] **Step 2: Run it, expect FAIL.** Run `cd web && npx vitest run test/i18n.test.ts`. Expect failure: `Failed to resolve import "@/lib/i18n"`.

- [ ] **Step 3: Implement `i18n`.** Create `web/lib/i18n.ts`:

```ts
// web/lib/i18n.ts
// Flat Macedonian-Cyrillic chrome dictionary. Default (and only) locale in v1;
// structured as a flat map so an EN locale can be added later without churn.
export const messages = {
  'app.title': 'ФИНКИ Хаб',
  'sidebar.new': 'Нов разговор',
  'sidebar.label': 'Странична лента',
  'sidebar.toggle': 'Прикажи/сокриј странична лента',
  'conversation.rename': 'Преименувај',
  'conversation.delete': 'Избриши',
  'conversation.renamePrompt': 'Ново име на разговорот',
  'thread.emptyTitle': 'Започни разговор',
  'thread.emptyDescription': 'Прашај нешто за студиите на ФИНКИ.',
  'composer.placeholder': 'Напиши порака…',
  'composer.message': 'Порака',
  'composer.model': 'Модел',
  'composer.send': 'Испрати',
  'composer.stop': 'Запри',
  'error.retry': 'Обиди се повторно',
  'error.interrupted': 'Одговорот е прекинат.',
  'actions.copy': 'Копирај',
  'actions.regenerate': 'Регенерирај',
  'actions.like': 'Допаѓа',
  'actions.dislike': 'Не допаѓа',
} as const;

export type TKey = keyof typeof messages;

export function t(key: TKey): string {
  return messages[key];
}
```

- [ ] **Step 4: Run it, expect PASS.** Run `cd web && npx vitest run test/i18n.test.ts`. Expect all three `i18n` tests to pass.

- [ ] **Step 5: Commit.**
```bash
git add web/lib/i18n.ts web/test/i18n.test.ts
git commit -m "feat(web): add Macedonian i18n dictionary and t() helper"
```

- [ ] **Step 6: Wire `t()` into `thread.tsx`.** In `web/components/chat/thread.tsx` add the import and replace the empty-state literals:

```tsx
// add to imports
import { t } from '@/lib/i18n';
```

Replace the `ConversationEmptyState` block:

```tsx
          <ConversationEmptyState
            title={t('thread.emptyTitle')}
            description={t('thread.emptyDescription')}
          />
```

- [ ] **Step 7: Wire `t()` into `message.tsx`.** In `web/components/chat/message.tsx` add `import { t } from '@/lib/i18n';` and replace the two literals: the interrupted notice `<p ...>Одговорот е прекинат.</p>` → `<p className="text-muted-foreground">{t('error.interrupted')}</p>`, and the retry button text `Обиди се повторно` → `{t('error.retry')}`.

- [ ] **Step 8: Wire `t()` into `composer.tsx`.** In `web/components/chat/composer.tsx` add `import { t } from '@/lib/i18n';` and swap: model `<label>` text → `{t('composer.model')}`; textarea `aria-label="Порака"` → `aria-label={t('composer.message')}`; textarea `placeholder="Напиши порака…"` → `placeholder={t('composer.placeholder')}`; the submit button `aria-label={isBusy ? 'Запри' : 'Испрати'}` → `aria-label={isBusy ? t('composer.stop') : t('composer.send')}`.

- [ ] **Step 9: Wire `t()` into `sidebar.tsx` and `conversation-list.tsx`.** In `web/components/shell/sidebar.tsx` add `import { t } from '@/lib/i18n';`, swap `aria-label="Странична лента"` (both occurrences) → `aria-label={t('sidebar.label')}` and the new-chat button text `Нов разговор` → `{t('sidebar.new')}`. In `web/components/shell/conversation-list.tsx` add the import and swap `aria-label="Преименувај"` → `aria-label={t('conversation.rename')}`, `aria-label="Избриши"` → `aria-label={t('conversation.delete')}`, and the prompt string `'Ново име на разговорот'` → `t('conversation.renamePrompt')`.

- [ ] **Step 10: Run the affected component suites, expect PASS (no string drift).** Run `cd web && npx vitest run test/thread.test.tsx test/composer.test.tsx test/shell.test.tsx`. Expect all green — every swapped value equals the literal the tests already assert (`Започни разговор`, `Нов разговор`, `Обиди се повторно`, `Одговорот е прекинат.`, `Преименувај`, `Избриши`, `Порака`, etc.), so behavior is unchanged.

- [ ] **Step 11: Typecheck.** Run `cd web && npx tsc --noEmit`. Expect no errors (the `t()` keys are checked against `TKey`, so a typo is a compile error).

- [ ] **Step 12: Commit.**
```bash
git add web/components/chat/thread.tsx web/components/chat/message.tsx web/components/chat/composer.tsx web/components/shell/sidebar.tsx web/components/shell/conversation-list.tsx
git commit -m "refactor(web): route chrome strings through the i18n t() helper"
```

The `web/` directory doesn't exist yet (this is a greenfield plan). Let me verify the AI Elements/AI SDK v5 route mocking idioms for Playwright by checking the cheat-sheet contract once more — I have everything I need from the spec and the provided contracts. Let me write Tasks 16 and 17.

I have all the information needed. Now I'll write Tasks 16 and 17 as the markdown plan.

```markdown
### Task 16: Playwright e2e — mocked streaming chat (chip, preamble drop, feedback)

**Files:**
- Create: `web/e2e/chat.spec.ts` (the e2e spec)
- Create: `web/e2e/helpers/sse.ts` (route-mock helper that builds protocol-v2 SSE bodies)
- Modify: `web/package.json` (add the `e2e` and `e2e:install` scripts — only if Task 1 did not already add `e2e`)
- Test: this task IS the test (Playwright spec); verified by running `npm run e2e`.

**Interfaces:**

Consumes (from earlier tasks — these are the runtime seams the e2e exercises against the *real* app):
- The app served at `/` renders the composer (`web/components/chat/composer.tsx`, Task 12) and the thread (`web/components/chat/thread.tsx`, Task 11).
- BFF route `POST /api/chat` (Task 5) — **intercepted/replaced** by the Playwright route mock, so the Python API is never contacted.
- BFF route `GET /api/models` (Task 6) — **intercepted** to return a fixed `string[]` so the model picker (Task 12) renders deterministically.
- BFF route `POST /api/feedback` (Task 7) — **intercepted** to assert the like POST body and return a `FeedbackAck`.
- `MyUIMessage` data parts `data-status` / `data-error` and metadata `{ responseId, inferenceModel }` (Task 3, `@/lib/api-types`) — what the mocked stream must emit so the UI behaves identically to production.
- The AI SDK UI message stream wire format (the `/api/chat` response that `useChat` consumes): NDJSON-style `data: <json>\n\n` SSE chunks with chunk types `start`, `text-start`, `text-delta`, `text-end`, `data-status`, `data-error`, `finish`. The mock emits exactly the chunks the real BFF (Task 5) would, so we test the **client** rendering contract.

Produces (used by Task 17's final checklist):
- `npm run e2e` runs green: a streaming tool sequence shows the search chip, drops the preamble, renders the final answer, and like posts to `/api/feedback`.
- `web/e2e/helpers/sse.ts` exporting `aiSdkStream(chunks: object[]): { body: string; contentType: string }` and `aiSdkChunks` factory helpers — reusable by future e2e specs.

---

- [ ] **Step 1: Add the e2e scripts to `web/package.json` (skip if already present).**

Open `web/package.json` and ensure the `scripts` block contains these two entries (add them if Task 1 only added a placeholder). Use `Edit` to insert into the existing `"scripts"` object — do not overwrite other scripts.

```jsonc
// web/package.json — inside "scripts" (merge, don't replace)
{
  "scripts": {
    // ...existing scripts (dev, build, test, typecheck) from Task 1...
    "e2e": "playwright test",
    "e2e:install": "playwright install chromium"
  }
}
```

Run, expected PASS (script is now resolvable):

```bash
cd web && npm run e2e:install
```

Expected: Playwright downloads/installs the Chromium browser (or reports it is already installed). If `playwright.config.ts` from Task 1 sets `webServer` and `testDir: './e2e'`, no further config is needed here.

- [ ] **Step 2: Write the SSE route-mock helper (`web/e2e/helpers/sse.ts`).**

This builds the **AI SDK UI message stream** body — i.e. what the *BFF returns to the browser*, not the raw python protocol-v2. `useChat` consumes SSE frames whose `data:` payload is a JSON chunk. We emit the exact chunk shapes the Task 5 translator produces, so the e2e exercises the client's rendering of a real tool sequence (preamble → status → reset → answer → finish-with-metadata).

```ts
// web/e2e/helpers/sse.ts
// Builds an AI SDK v5 "UI message stream" SSE body for Playwright route mocking.
// This is the BFF->browser wire format that useChat parses: each chunk is a JSON
// object emitted as `data: <json>\n\n`. We model exactly the chunks the real
// /api/chat translator (Task 5) writes for a tool run.

export type UiChunk =
  | { type: 'start'; messageMetadata?: { responseId?: string; inferenceModel?: string } }
  | { type: 'text-start'; id: string }
  | { type: 'text-delta'; id: string; delta: string }
  | { type: 'text-end'; id: string }
  | { type: 'data-status'; data: { label: string; tool?: string }; transient: true }
  | { type: 'data-error'; data: { code: string; message: string }; transient: true }
  | { type: 'finish' };

/** Serialize chunks into an AI SDK UI-message-stream SSE body. */
export function aiSdkStream(chunks: UiChunk[]): { body: string; contentType: string } {
  const body = chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join('');
  return { body: `${body}data: [DONE]\n\n`, contentType: 'text/event-stream' };
}

/**
 * Canonical "tool run" sequence:
 *   start(metadata) -> preamble text part -> status chip -> reset (new text part)
 *   -> answer tokens -> finish.
 * The client must: show the chip, drop the preamble (render-last), render the answer.
 */
export function toolRunChunks(opts: {
  responseId: string;
  inferenceModel: string;
  preamble: string;
  statusLabel: string;
  tool: string;
  answer: string;
}): UiChunk[] {
  const preambleId = 'txt-preamble';
  const answerId = 'txt-answer';
  return [
    { type: 'start', messageMetadata: { responseId: opts.responseId, inferenceModel: opts.inferenceModel } },
    // preamble streamed before the tool ran (must be dropped after reset):
    { type: 'text-start', id: preambleId },
    { type: 'text-delta', id: preambleId, delta: opts.preamble },
    { type: 'text-end', id: preambleId },
    // tool starts -> transient status chip:
    { type: 'data-status', data: { label: opts.statusLabel, tool: opts.tool }, transient: true },
    // reset -> a brand-new answer text part (render-last keeps only this one):
    { type: 'text-start', id: answerId },
    { type: 'text-delta', id: answerId, delta: opts.answer },
    { type: 'text-end', id: answerId },
    { type: 'finish' },
  ];
}
```

This file is a pure helper with no test of its own (it is exercised by the spec in Step 4). Quick sanity-import is implied by the spec compiling.

- [ ] **Step 3: Run the (not-yet-written) spec to confirm Playwright resolves the test dir — expected FAIL (no tests).**

```bash
cd web && npm run e2e
```

Expected: Playwright reports **"No tests found"** (or runs 0 tests) because `web/e2e/chat.spec.ts` does not exist yet. This confirms the runner and config are wired before we author the spec.

- [ ] **Step 4: Write the e2e spec (`web/e2e/chat.spec.ts`).**

The spec mocks all three BFF routes via `page.route`, types a question, submits, and asserts the four spec requirements: (1) the search chip appears, (2) the preamble is dropped, (3) the answer renders, (4) clicking like POSTs to `/api/feedback` with the right body.

> Selector note: this spec relies on test ids / accessible text wired by Tasks 11–14. It asserts against **`data-testid`** hooks (`search-status`, `answer-text`, `like-button`) and Macedonian placeholder/aria text from Task 15. If a selector does not match, fix the component to expose the test id rather than loosening the assertion — the e2e is the contract.

```ts
// web/e2e/chat.spec.ts
import { expect, test } from '@playwright/test';
import { aiSdkStream, toolRunChunks } from './helpers/sse';

const RESPONSE_ID = '11111111-2222-3333-4444-555555555555';
const INFERENCE_MODEL = 'claude-sonnet-4-6';
const PREAMBLE = 'Дозволете да проверам…';
const STATUS_LABEL = '🔍 Пребарувам…';
const TOOL = 'search_documents';
const ANSWER = 'Резултатите од испитите се објавуваат на https://finki.ukim.mk';

test.describe('chat streaming (mocked BFF)', () => {
  test('shows the search chip, drops the preamble, renders the answer, and likes', async ({ page }) => {
    // 1) models picker -> deterministic list
    await page.route('**/api/models', async (route) => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([INFERENCE_MODEL, 'gpt-5.4-mini']),
      });
    });

    // 2) chat -> a full tool-run UI message stream
    await page.route('**/api/chat', async (route) => {
      const { body, contentType } = aiSdkStream(
        toolRunChunks({
          responseId: RESPONSE_ID,
          inferenceModel: INFERENCE_MODEL,
          preamble: PREAMBLE,
          statusLabel: STATUS_LABEL,
          tool: TOOL,
          answer: ANSWER,
        }),
      );
      await route.fulfill({
        status: 200,
        headers: { 'content-type': contentType, 'x-vercel-ai-ui-message-stream': 'v1' },
        body,
      });
    });

    // 3) feedback -> capture the request, return an ack
    let feedbackBody: Record<string, unknown> | null = null;
    await page.route('**/api/feedback', async (route) => {
      feedbackBody = route.request().postDataJSON() as Record<string, unknown>;
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ id: 1, response_id: RESPONSE_ID, feedback_type: 'like' }),
      });
    });

    await page.goto('/');

    // submit a question via the composer (Macedonian placeholder from Task 15 i18n)
    const input = page.getByTestId('composer-input');
    await input.fill('Кога се објавуваат резултатите?');
    await input.press('Enter');

    // (1) the search chip appears for the tool call
    const chip = page.getByTestId('search-status');
    await expect(chip).toBeVisible();
    await expect(chip).toContainText('Пребарувам');

    // (3) the final answer renders (autolinked bare URL via Streamdown)
    const answer = page.getByTestId('answer-text');
    await expect(answer).toContainText('Резултатите од испитите се објавуваат');
    await expect(answer.getByRole('link', { name: /finki\.ukim\.mk/ })).toBeVisible();

    // (2) the preamble text part was dropped (render-last): it must NOT be on screen
    await expect(page.getByText(PREAMBLE)).toHaveCount(0);

    // (4) like posts to /api/feedback with the response id + feedback_type
    await page.getByTestId('like-button').click();
    await expect.poll(() => feedbackBody).not.toBeNull();
    expect(feedbackBody).toMatchObject({
      responseId: RESPONSE_ID,
      feedbackType: 'like',
    });
  });
});
```

> If Task 14's answer-actions client posts the *assembled* `FeedbackSchema` (with `client:"web"`, `user_id`) to `/api/feedback` instead of the thin `{ responseId, feedbackType, userId }` client payload, change the `toMatchObject` to `{ response_id: RESPONSE_ID, feedback_type: 'like', client: 'web' }`. Per the contract (`FeedbackClientPayload` in `@/lib/api-types`, Task 7 BFF assembles the schema), the **browser** sends the camelCase client payload, so the assertion above is correct.

- [ ] **Step 5: Run the e2e — expected PASS.**

```bash
cd web && npm run e2e
```

Expected: 1 passed. If a selector fails (`getByTestId` returns nothing), confirm Tasks 11/12/14 set the matching `data-testid` (`composer-input`, `search-status`, `answer-text`, `like-button`); the e2e is the source of truth for those hooks. If the chip is gone before the assertion (it is transient and disappears once the answer streams), assert it via `expect(chip).toBeVisible()` racing the stream is still safe because Playwright auto-waits on the first matching frame; if it flakes, slow the mock by chunking the stream body across multiple `route.fulfill` is not possible — instead split the answer into more `text-delta` chunks so the chip is observable longer (keep the chip assertion before the answer assertion, as written).

- [ ] **Step 6: Commit.**

```bash
cd web && git add e2e/chat.spec.ts e2e/helpers/sse.ts package.json
git commit -m "test(web): add Playwright e2e for mocked streaming chat (chip, preamble drop, feedback)"
```

---

### Task 17: web/README.md + final verification

**Files:**
- Create: `web/README.md` (setup, env vars, scripts, architecture pointer)
- Modify: none (this task only documents and runs the aggregate verification; it changes no app code)
- Test: no new unit test; the deliverable is the documented commands all passing (`npm run typecheck`, `npm test`, `npm run build`, `npm run e2e`).

**Interfaces:**

Consumes (everything prior — this is the integration gate):
- Scripts from `web/package.json` (Task 1: `dev`, `build`, `typecheck`, `test`; Task 16: `e2e`, `e2e:install`).
- Env vars `API_BASE_URL` and `CHAT_API_KEY` from `@/lib/env` (Task 3, server-only; spec §13).
- The full route/UI surface (Tasks 5–14) so `npm run build` and `npm run e2e` exercise the real app.

Produces:
- `web/README.md` — onboarding doc for an engineer with zero context.
- A green final checklist: typecheck + unit + build + e2e all pass; no `test.skip`/`.only`/TODO placeholders remain.

---

- [ ] **Step 1: Write `web/README.md`.**

```markdown
# FINKI Hub — Web Chat

A claude.ai / ChatGPT-style chat web app for the FINKI Hub chat API. It streams
answers with the full agent UX (live tokens + a "searching…" tool chip), keeps
conversations locally (IndexedDB), lets you pick the model, renders Markdown, and
supports like/dislike feedback.

- **Framework:** Next.js (App Router, React 19) + TypeScript (strict), Tailwind v4.
- **Streaming / chat state:** Vercel AI SDK v5 (`ai`, `@ai-sdk/react`) + AI Elements.
- **Local history:** Dexie (IndexedDB). **Models cache:** TanStack Query. **UI state:** Zustand.
- **BFF:** Next.js Route Handlers under `app/api/*` hold the API key, translate the
  protocol-v2 SSE into an AI SDK UI message stream, and proxy models/feedback.

The browser only ever calls same-origin `/api/*`; the Python API URL and the
`x-api-key` live exclusively in the BFF (server-only). See `docs/superpowers/specs/2026-06-25-web-chat-frontend-design.md` for the full design.

## Prerequisites

- Node.js 20+ and **npm** (ships with Node.js — no extra install). This is a standalone npm app under `web/`.
- A reachable **protocol-v2** FINKI Hub chat API (see env below). The parser also
  tolerates the legacy bare-`data:` stream during the rollout window.

## Setup

```bash
cd web
npm install
npm run e2e:install   # one-time: download the Playwright Chromium browser
```

## Environment variables (server-only — never exposed to the browser)

Create `web/.env.local`:

```bash
# Base URL of the Python chat API (protocol-v2). No /api prefix; /chat/ has a trailing slash.
API_BASE_URL=http://localhost:8880
# Master x-api-key for POST /chat/feedback. Injected by the BFF; never sent to the browser.
CHAT_API_KEY=replace-with-the-master-api-key
```

There are **no** `NEXT_PUBLIC_*` vars in v1. `lib/env.ts` is `import 'server-only'`
and throws on startup if either var is missing — do not import it from a Client
Component.

## Commands

| Command | What it does |
|---|---|
| `npm run dev` | Run the app at http://localhost:3000 (needs the env vars above). |
| `npm run build` | Production build (`next build`). |
| `npm start` | Serve the production build. |
| `npm run typecheck` | `tsc --noEmit` against the strict config. |
| `npm test` | Vitest unit/component suite (jsdom + fake-indexeddb). |
| `npm run e2e` | Playwright e2e (mocked BFF; no live API needed). |
| `npm run e2e:install` | One-time Playwright browser download. |

## Project layout

```
web/
  app/
    layout.tsx, page.tsx          # chat screen (sidebar + thread)
    api/chat/route.ts             # protocol-v2 -> AI SDK UI message stream translator (BFF)
    api/feedback/route.ts         # injects x-api-key, proxies /chat/feedback
    api/models/route.ts           # proxies + caches /chat/models
  components/
    ai-elements/                  # vendored AI Elements (Conversation, Message, PromptInput, …)
    ui/                           # shadcn primitives
    shell/                        # sidebar, conversation-list, app shell
    chat/                         # thread, message, search-status chip, composer, answer-actions
  lib/
    env.ts                        # server-only env access (API_BASE_URL, CHAT_API_KEY)
    api-types.ts                  # ChatSchema/FeedbackSchema/MyUIMessage wire types
    sse.ts                        # protocol-v2 parser (+ legacy bare-data fallback)
    chat-translate.ts             # pure protocol-v2 -> UI-message-stream mapping (unit-tested)
    transport.ts                  # DefaultChatTransport / prepareSendMessagesRequest
    db.ts                         # Dexie schema + conversation/message helpers
    use-models.ts                 # TanStack Query hook for /api/models
    ui-store.ts                   # Zustand (active model/conversation, sidebar)
    user.ts, messages.ts, i18n.ts # anon id, request trimming/titles, Macedonian chrome
  e2e/                            # Playwright specs + SSE mock helper
```

## How the streaming pipeline works (one paragraph)

`useChat` (client) sends `messages[]` + sampling/model fields to `POST /api/chat`.
The BFF converts that to the Python `ChatSchema` (oldest-first, last-is-user, caps
50 msgs / 8000 chars), calls `POST {API_BASE_URL}/chat/`, reads `X-Response-Id`,
then maps protocol-v2 events onto an AI SDK UI message stream: `token`→`text-delta`,
`status`→transient `data-status`, `reset`→end+restart the text part (preamble drop),
`error`→`data-error`, `done`→finish; message metadata carries `{responseId, inferenceModel}`.
The client renders only the **last** text part per assistant message (so the
pre-tool preamble is dropped) and shows the `data-status` chip via `onData`.

## Deferred (not in v1)

Auth/login, rate-limiting, server-side conversation sync, the FAQ-links/recommender
UI, attachments/voice, and the Docker image + `compose.yaml` wiring (the app is a
standalone npm app for now).
```

- [ ] **Step 2: Verify the README's `typecheck` command passes.**

```bash
cd web && npm run typecheck
```

Expected: `tsc --noEmit` exits 0 with no errors. If anything fails, it is a real regression from an earlier task — fix the offending source before proceeding (do not edit the README to hide it).

- [ ] **Step 3: Verify the unit/component suite passes.**

```bash
cd web && npm test
```

Expected: all Vitest suites green (Tasks 3–15). 0 failed, 0 skipped.

- [ ] **Step 4: Guard against fake-completion — assert no skipped/only/TODO markers remain.**

```bash
cd web && grep -rnE "\.(skip|only)\(|\bTODO\b|\bFIXME\b" app components lib e2e test --include="*.ts" --include="*.tsx"
```

Expected: **no output** (grep exits non-zero / prints nothing). Any hit is a blocker: a `test.skip`/`.only`, an unimplemented stub, or a leftover TODO. Implement it or report it — do not ship around it. (`grep` returning exit code 1 with empty output here is the success signal.)

- [ ] **Step 5: Verify the production build succeeds.**

```bash
cd web && npm run build
```

Expected: `next build` completes with no type or build errors and emits the route manifest including `/api/chat`, `/api/models`, `/api/feedback`, and `/`. `API_BASE_URL` / `CHAT_API_KEY` must be set in `web/.env.local` (Step 1 of setup) because `lib/env.ts` validates them — if the build trips the env guard, that confirms the server-only guard is wired correctly; set the vars and re-run.

- [ ] **Step 6: Verify the e2e suite passes (mocked BFF — no live API needed).**

```bash
cd web && npm run e2e
```

Expected: 1 passed (Task 16). This is the end-to-end proof that the search chip appears, the preamble is dropped, the answer renders, and like posts to `/api/feedback`.

- [ ] **Step 7: Commit the README and final verification.**

```bash
cd web && git add README.md
git commit -m "docs(web): add README with setup, env vars, and the final verification checklist"
```
```
