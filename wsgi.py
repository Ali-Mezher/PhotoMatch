"""Production WSGI entry point used by Gunicorn."""

from src.web import create_app


app = create_app()
