"""Keep tests hermetic: never let a developer .env change test behaviour."""

import os

os.environ["SPECCHECK_ADAPTER"] = "mock"
os.environ["META_API_KEY"] = ""
os.environ.pop("MODEL_API_KEY", None)
