"""Install the bundled find_the_shadow law inputs and build the deterministic orientation table."""
import argparse
import os
from pathlib import Path
from visro.paths import ROOT
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True, type=Path)
    a = p.parse_args()
    cache = a.cache.resolve()
    if not (cache / 'strikes_92.json').is_file():
        p.error('build the base cache first (see docs/ASSETS.md)')
    for src in sorted((ROOT / 'assets/find_the_shadow_law').rglob('*')):
        if not src.is_file():
            continue
        target = cache / src.relative_to(ROOT / 'assets/find_the_shadow_law')
        if target.exists() and target.read_bytes() != src.read_bytes():
            raise SystemExit(f'Refusing to replace a different cache input: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
    env = {**os.environ, 'VISRO_FTS_CACHE': str(cache)}
    subprocess.run([sys.executable, '-m', 'tasks.discrimination_grouping.find_the_shadow.generator.build.build_law',
                    'orient'], env=env, cwd=ROOT, check=True)
    print(f"Shadow tables prepared in {cache}")


if __name__ == '__main__':
    main()
