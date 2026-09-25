# Production Container for LoanLens (Node.js Express SSR + Python Flask ML Inference Engine)
FROM python:3.11-slim

# Install system dependencies & Node.js 20.x
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python ML dependencies
COPY flask_backend/requirements.txt ./flask_backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r flask_backend/requirements.txt

# Install Node dependencies
COPY package.json package-lock.json* tsconfig.json ./
RUN npm install

# Copy source code and build TypeScript
COPY server.ts ./
COPY src/ ./src/
RUN npm run build

# Copy runtime assets and artifacts
COPY artifacts/ ./artifacts/
COPY flask_backend/ ./flask_backend/
COPY views/ ./views/
COPY public/ ./public/

# Prune development dependencies to keep image lean
RUN npm prune --production

# Default environment variables for production container
ENV NODE_ENV=production \
    HOST=0.0.0.0 \
    PORT=3000 \
    FLASK_HOST=127.0.0.1 \
    FLASK_PORT=5001 \
    FLASK_BACKEND_URL=http://127.0.0.1:5001 \
    USE_REMOTE_BACKEND=true \
    ALLOWED_ORIGINS=*

EXPOSE 3000

CMD ["node", "dist/server.js"]
