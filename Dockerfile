# Stage 1: Build React frontend
FROM node:22-alpine AS frontend-build

WORKDIR /app/frontend

COPY smartlearn-frontend/package.json smartlearn-frontend/package-lock.json ./

RUN npm ci

COPY smartlearn-frontend/ ./

# Empty VITE_API_URL → frontend calls same origin (production).
ARG VITE_API_URL=
ENV VITE_API_URL=${VITE_API_URL}

RUN npm run build

# Stage 2: Python backend + serve static files
FROM python:3.13-slim

WORKDIR /app

# Install backend dependencies
COPY smartlearn-backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY smartlearn-backend/ ./

# Copy built frontend from stage 1
COPY --from=frontend-build /app/frontend/dist ./static

# Serve static files from /app/static
ENV STATIC_DIR=/app/static

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
