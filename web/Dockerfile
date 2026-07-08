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
# Public PostHog ingest key, baked at build (NEXT_PUBLIC_* are inlined by next build).
# Empty by default = analytics off; CI supplies it via --build-arg NEXT_PUBLIC_POSTHOG_KEY=... to enable.
ARG NEXT_PUBLIC_POSTHOG_KEY=
ARG NEXT_PUBLIC_POSTHOG_HOST=https://eu.i.posthog.com
ENV NEXT_PUBLIC_POSTHOG_KEY=$NEXT_PUBLIC_POSTHOG_KEY
ENV NEXT_PUBLIC_POSTHOG_HOST=$NEXT_PUBLIC_POSTHOG_HOST
RUN API_BASE_URL="http://localhost:8880" \
  AUTH_GOOGLE_ID="build-time-google-client-id" \
  AUTH_GOOGLE_SECRET="build-time-google-client-secret" \
  AUTH_MICROSOFT_ENTRA_ID_ID="" \
  AUTH_MICROSOFT_ENTRA_ID_ISSUER="" \
  AUTH_MICROSOFT_ENTRA_ID_SECRET="" \
  AUTH_SECRET="build-time-auth-secret" \
  AUTH_URL="http://localhost:3000" \
  CHAT_API_KEY="build-time-placeholder" \
  RESUMABLE_STREAM_REDIS_URL="redis://localhost:6379" \
  npm run build

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
