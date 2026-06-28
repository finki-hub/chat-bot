# syntax=docker/dockerfile:1

FROM node:24-slim AS base
ENV NEXT_TELEMETRY_DISABLED=1

FROM base AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# Public PostHog ingest key, inlined at build (NEXT_PUBLIC_* are baked by next build).
# Safe to expose — it is a write-only browser key. Defaults to the public project key so
# images always ship analytics; override via --build-arg NEXT_PUBLIC_POSTHOG_KEY=... if needed.
ARG NEXT_PUBLIC_POSTHOG_KEY=phc_xXEqLMnYeDPuXA6HHwuasQMdSufDGryS8vZZuHmu9Qwd
ARG NEXT_PUBLIC_POSTHOG_HOST=https://eu.i.posthog.com
ENV NEXT_PUBLIC_POSTHOG_KEY=$NEXT_PUBLIC_POSTHOG_KEY
ENV NEXT_PUBLIC_POSTHOG_HOST=$NEXT_PUBLIC_POSTHOG_HOST
RUN API_BASE_URL="http://localhost:8880" CHAT_API_KEY="build-time-placeholder" npm run build

FROM base AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"
RUN groupadd --system --gid 1001 nodejs \
  && useradd --system --uid 1001 --gid nodejs nextjs
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
USER nextjs
EXPOSE 3000
CMD ["node", "server.js"]
