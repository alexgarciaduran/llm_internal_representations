"""Import for side effects: quieter output, and UTF-8 on Windows consoles."""
import logging
import os
import sys
import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# cp1252 cannot encode the byte-BPE markers tokenizers print, so printing a token
# list would otherwise raise UnicodeEncodeError
for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8")
    except Exception:
        pass

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*n_jobs value.*overridden.*")
logging.getLogger("transformers").setLevel(logging.ERROR)
try:
    from transformers.utils import logging as _hf
    _hf.set_verbosity_error()
except Exception:
    pass
