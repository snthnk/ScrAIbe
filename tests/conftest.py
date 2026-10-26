import os
import sys
from pathlib import Path

os.environ.setdefault("LOG_TO_FILE", "0")
os.environ.setdefault("LOG_LEVEL", "CRITICAL")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
