"""The find_the_shadow cache directory ($VISRO_FTS_CACHE, else
~/.cache/visro/find_the_shadow) and how to build it."""
import os
from pathlib import Path

CACHE_DIR = Path(os.environ.get("VISRO_FTS_CACHE")
                 or Path.home() / ".cache" / "visro" / "find_the_shadow").expanduser()
BUILD_HINT = (
    "build it once (about 20 min on 6 processes) with\n"
    "    VISRO_FTS_CACHE=<dir> VISRO_FTS_WORKERS=6 python -m tasks.discrimination_grouping.find_the_shadow.generator.build.build_cache\n"
    "and point VISRO_FTS_CACHE at <dir>; see docs/ASSETS.md")
