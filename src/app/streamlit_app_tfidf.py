from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["MET_ARTIFACTS_DIR"] = "artifacts_tfidf"
os.environ["MET_AUTO_BUILD_ON_STARTUP"] = "false"
os.environ["MET_ENABLE_VISION"] = "false"

# Ensure absolute repo-root import path exists before importing `src`.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import get_settings

get_settings.cache_clear()

from src.app.streamlit_app import *  # noqa: F401,F403
