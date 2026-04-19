"""Compatibility shim so `uvicorn main:app` keeps working.

This re-exports the FastAPI `app` from the package entrypoint
`src.api.main` so deployments still calling `main:app` succeed.
"""

from src.api.main import app
