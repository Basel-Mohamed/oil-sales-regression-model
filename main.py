"""Vercel entrypoint. Vercel looks for `app` in a root-level main.py; the real app lives in api/app.py."""
from api.app import app  # noqa: F401
