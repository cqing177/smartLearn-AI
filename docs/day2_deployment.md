# Day 2 Deployment

## URLs
- Frontend: https://smart-learn-ai-4fmr.vercel.app/
- Backend health: https://smartlearn-ai-production-bd72.up.railway.app/health
- Backend docs: https://smartlearn-ai-production-bd72.up.railway.app/docs

## Source
- Repository: https://github.com/cqing177/smartLearn-AI
- Deployed branch / merge target: main
- Merged commit: 5698e8a (fix: use PORT env var for Railway compatibility)
- Pull Request: N/A (commits pushed directly to main)

## Root Directories
- Railway: smartlearn-backend
- Vercel: smartlearn-frontend

## Environment variable names
- Railway: OPENROUTER_API_KEY, ALLOWED_ORIGINS
- Vercel: VITE_API_URL

## Acceptance results
- /health: pass
- Upload: pass
- Known /chat + citations: pass (citations reference valid pages from uploaded PDF)
- Unknown question: pass (response admits insufficient evidence)
- CORS restart + re-upload recovery: pass

## Known limitations
- Railway restart clears in-memory uploaded/chat state; re-upload is expected.
