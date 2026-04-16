# syntax=docker/dockerfile:1.7
# -----------------------------------------------------------------------------
# Webapp image: Vite/React build served by nginx, with /api proxied to backend.
#
# This image is part of a two-service deployment:
#   - webapp   (this image)   : serves static assets + reverse-proxies /api -> backend
#   - backend  (python:3.12)  : uvicorn backend.main:app on :8001
#
# Why two services instead of one image?
#   - A single image would need a process supervisor (supervisord/s6) to run
#     both nginx and uvicorn, or would have to embed Python into an nginx
#     image. Either is more fragile than splitting concerns.
#   - docker-compose already gives us service-level isolation, logs, and
#     restart policies for free.
# -----------------------------------------------------------------------------

# ---- Stage 1: build the Vite bundle ----
FROM node:20-alpine AS webapp-builder

WORKDIR /app/webapp

# Install dependencies first so this layer caches when source changes.
COPY package.json package-lock.json ./
RUN npm ci

# Copy the rest of the webapp source and build.
COPY tsconfig.json tsconfig.app.json tsconfig.node.json vite.config.ts eslint.config.js index.html ./
COPY public ./public
COPY src ./src

# Webapp reads VITE_API_BASE at build time; through nginx we expose /api
# on the same origin, so the default ("/api") is correct.
ENV VITE_API_BASE=/api
RUN npm run build

# ---- Stage 2: nginx runtime ----
FROM nginx:1.27-alpine AS runtime

# Replace default site with one that serves the SPA and proxies /api -> backend.
RUN rm -f /etc/nginx/conf.d/default.conf
COPY --from=webapp-builder /app/webapp/dist /usr/share/nginx/html

# Inline nginx config. Written here (rather than a separate file) so this
# Dockerfile is self-contained for the webapp/ directory.
RUN printf '%s\n' \
    'server {' \
    '  listen 80;' \
    '  server_name _;' \
    '  root /usr/share/nginx/html;' \
    '  index index.html;' \
    '' \
    '  # Reverse-proxy API calls to the backend service on the compose network.' \
    '  location /api/ {' \
    '    proxy_pass http://backend:8001/api/;' \
    '    proxy_http_version 1.1;' \
    '    proxy_set_header Host $host;' \
    '    proxy_set_header X-Real-IP $remote_addr;' \
    '    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;' \
    '    proxy_set_header X-Forwarded-Proto $scheme;' \
    '    proxy_read_timeout 120s;' \
    '  }' \
    '' \
    '  # SPA fallback: unknown routes return index.html so React Router handles them.' \
    '  location / {' \
    '    try_files $uri $uri/ /index.html;' \
    '  }' \
    '}' \
    > /etc/nginx/conf.d/app.conf

EXPOSE 80

# nginx image already sets an appropriate CMD; no override needed.
