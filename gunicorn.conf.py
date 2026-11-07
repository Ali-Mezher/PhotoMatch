"""Safe defaults for one SQLite database and one background coordinator."""

import os


bind = os.environ.get("PHOTOMATCH_BIND", "0.0.0.0:8000")
workers = 1
threads = int(os.environ.get("PHOTOMATCH_WEB_THREADS", "4"))
worker_class = "gthread"
timeout = int(os.environ.get("PHOTOMATCH_REQUEST_TIMEOUT", "120"))
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
capture_output = True
