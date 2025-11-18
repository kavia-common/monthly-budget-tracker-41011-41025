# Monthly Budget Tracker - Backend API

FastAPI backend providing REST APIs for auth (demo), transactions, budgets, categories, dashboard summaries, and seed data. Uses SQLite via SQLModel and seeds demo data on startup.

## Run (local)
- Install dependencies: `pip install -r requirements.txt`
- Start: `uvicorn src.api.main:app --host 0.0.0.0 --port 3001 --reload`

CORS is permissive for demo. No environment variables required.

## Base URL
`http://localhost:3001`

## API Prefix
`/api`

## Key Endpoints
- POST `/api/auth/login` -> `{ "token": "demo-token-<id>" }`
- GET `/api/categories`
- GET/POST/PUT/DELETE `/api/transactions`
- GET/POST/PUT/DELETE `/api/budgets`
- GET `/api/dashboard/summary?month=YYYY-MM`
- POST `/api/seed`

OpenAPI docs: `/docs` and `/openapi.json`

The app creates and seeds `budget.db` (SQLite) automatically on startup.
