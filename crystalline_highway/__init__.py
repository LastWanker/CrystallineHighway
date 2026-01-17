"""Crystalline Highway 记忆系统原型包。"""

from __future__ import annotations

import sys
from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from src.crystalline_highway import MemorySystem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
PACKAGE_PATH = SRC_PATH / __name__

if PACKAGE_PATH.exists():
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))
    if str(PACKAGE_PATH) not in __path__:
        __path__.append(str(PACKAGE_PATH))


__all__ = ["MemorySystem"]
