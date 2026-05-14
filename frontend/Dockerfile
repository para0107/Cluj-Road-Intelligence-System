# ── frontend/Dockerfile ───────────────────────────────────────────────────────
# Stage 1 — Build the React/Vite app
# Stage 2 — Serve the static bundle with Nginx
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY frontend/package.json frontend/package-lock.json ./

RUN npm ci

# Copy the rest of the frontend source
COPY frontend/ .

# Production build — output goes to /app/dist
RUN npm run build

# ── Stage 2: serve ────────────────────────────────────────────────────────────
FROM nginx:alpine

# Remove default Nginx static assets
RUN rm -rf /usr/share/nginx/html/*

# Copy built assets from Stage 1
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy our custom Nginx config (handles SPA routing + API proxy)
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
