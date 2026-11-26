import importlib.util
import os
import sys

REQUIRED_BASE_MODULES = ("yaml", "numpy", "jieba")


def _ensure_usable_python(argv=None, *, find_spec=importlib.util.find_spec,
                          execv=os.execv, executable=None):
    """Re-exec with system Python when invoked under a stripped runtime (e.g. Hermes'
    bundled Python 3.14 in cron jobs, which lacks PyYAML / numpy / jieba)."""
    if not any(find_spec(m) is None for m in REQUIRED_BASE_MODULES):
        return False
    cur = os.path.realpath(executable or sys.executable)
    args = list(sys.argv[1:] if argv is None else argv)
    for alt in (os.environ.get("FAULT_RAG_PYTHON"), "/usr/bin/python3", "/usr/local/bin/python3"):
        if alt and os.path.exists(alt) and os.path.realpath(alt) != cur:
            execv(alt, [alt, "-m", "fault_rag", *args])
            return True
    return False


_ensure_usable_python()

from .cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
