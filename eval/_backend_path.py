"""Path bootstrap for the standalone eval package.

The eval harness was pulled out of ``backend/`` into its own top-level folder,
but it still imports backend modules directly (``config``, ``memory.*``,
``agents.*``, ``core.*``). This shim inserts ``backend/`` onto ``sys.path`` so
those imports resolve regardless of where the eval scripts are launched from.

Import it once at the top of any eval module that touches backend code:

    from eval._backend_path import _  # noqa: F401   # adds backend/ to sys.path
    from config import settings
    from memory.v22 import get_memory_store

``_`` is a throwaway binding so the import line reads as a side-effect.
"""
from __future__ import annotations

import sys
from pathlib import Path

# eval/ is one level up from this file; backend/ is a sibling of eval/.
_BACKEND = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_ = None  # sentinel for `import _backend_path as _` style usage
