"""
Production entry point: serves the Flask app with waitress (a production WSGI server
that works on both Linux and Windows) instead of Flask's development server.

    python web/serve.py

Configured through environment variables (defaults suit Hugging Face Spaces):
    HOST     interface to bind          (default 0.0.0.0)
    PORT     port to listen on          (default 7860, the Spaces default)
    THREADS  waitress worker threads    (default 4)

The model is loaded in a background thread at startup, so the server accepts
connections immediately; /healthz returns 503 until the model is ready, then 200.
"""

import logging
import os
import threading

from waitress import serve

from app import app, warm_up

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

if __name__ == "__main__":
    threading.Thread(target=warm_up, name="model-warm-up", daemon=True).start()
    serve(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "7860")),
        threads=int(os.environ.get("THREADS", "4")),
    )
