# SPDX-License-Identifier: MIT
"""Fail if the published tree breaks license or portability requirements.

Checks, over `git ls-files` only:
  1. every commentable file declares `SPDX-License-Identifier: MIT` near the top;
  2. every other file is covered by a REUSE.toml annotation (JSON records, images);
  3. no tracked text file contains an absolute Windows, macOS or Linux home path;
  4. committed run records store portable (project-relative) directories;
  5. LICENSE and LICENSES/MIT.txt exist, and pyproject dependencies match requirements.txt.

Usage: python scripts/check_compliance.py
"""
from __future__ import annotations

import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IDENTIFIER = 'SPDX-License-Identifier: MIT'
BINARY_SUFFIXES = {
    '.png', '.jpg', '.jpeg', '.gif', '.webp', '.ico', '.pdf', '.zip', '.npz', '.npy',
    '.pt', '.pth', '.bin', '.exe', '.woff', '.woff2', '.ttf', '.otf', '.mp4', '.wav',
}
# Absolute machine paths that must never appear in a published file.
MACHINE_PATH = re.compile(
    r'(?<![A-Za-z0-9_])[A-Za-z]:[\\/](?:[^\s"\'<>|]*[\\/])?'   # Windows drive
    r'|/Users/[A-Za-z0-9._-]+/'
    r'|/home/[A-Za-z0-9._-]+/'
    r'|\\\\\\\\[A-Za-z0-9._-]+[\\/]'                             # UNC share
)
EXEMPT_PATHS = {'LICENSE', 'LICENSES/MIT.txt', 'REUSE.toml', 'docs/paper.txt'}


def tracked() -> list[str]:
    result = subprocess.run(['git', 'ls-files'], cwd=ROOT, capture_output=True, text=True, check=True)
    return [line for line in result.stdout.splitlines() if line]


def reuse_annotations() -> list[str]:
    """Collect the path patterns annotated in REUSE.toml (minimal TOML subset)."""
    path = ROOT / 'REUSE.toml'
    if not path.is_file():
        return []
    text = path.read_text(encoding='utf-8')
    patterns: list[str] = []
    for block in re.split(r'^\[\[annotations\]\]$', text, flags=re.M)[1:]:
        match = re.search(r'path\s*=\s*\[(.*?)\]', block, flags=re.S)
        licensed = re.search(r'SPDX-License-Identifier\s*=\s*"MIT"', block)
        if match and licensed:
            for item in re.findall(r'"([^"]+)"', match.group(1)):
                patterns.append(item)
    return patterns


def read_text(path: Path) -> str | None:
    if path.suffix.lower() in BINARY_SUFFIXES or not path.is_file():
        return None
    try:
        return path.read_text(encoding='utf-8')
    except (UnicodeDecodeError, OSError):
        return None


def declared(text: str) -> bool:
    return IDENTIFIER in '\n'.join(text.splitlines()[:6])


def check_headers(files: list[str], patterns: list[str]) -> list[str]:
    problems = []
    for relative in files:
        if relative in EXEMPT_PATHS:
            continue
        if any(fnmatch.fnmatch(relative, pattern) for pattern in patterns):
            continue
        text = read_text(ROOT / relative)
        if text is None:
            continue
        if not declared(text):
            problems.append(f'missing {IDENTIFIER}: {relative}')
    return problems


def check_paths(files: list[str]) -> list[str]:
    problems = []
    for relative in files:
        text = read_text(ROOT / relative)
        if text is None:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if MACHINE_PATH.search(line):
                problems.append(f'machine-specific path in {relative}:{number}')
                break
    return problems


def check_run_records(files: list[str]) -> list[str]:
    problems = []
    for relative in files:
        if not relative.startswith('runs/') or not relative.endswith('.json'):
            continue
        payload = json.loads((ROOT / relative).read_text(encoding='utf-8'))
        config = payload.get('config') if isinstance(payload, dict) else None
        if not isinstance(config, dict):
            continue
        for key in ('data_dir', 'run_root'):
            value = config.get(key)
            if isinstance(value, str) and (Path(value).is_absolute() or MACHINE_PATH.search(value)):
                problems.append(f'{relative}: config.{key} is not portable ({value!r})')
    return problems


def normalise(name: str) -> str:
    return re.split(r'[<>=!~\[;]', name, maxsplit=1)[0].strip().lower().replace('_', '-')


def check_dependencies() -> list[str]:
    problems = []
    requirements = ROOT / 'requirements.txt'
    pyproject = ROOT / 'pyproject.toml'
    if not requirements.is_file() or not pyproject.is_file():
        return ['requirements.txt or pyproject.toml is missing']
    wanted = {normalise(line) for line in requirements.read_text(encoding='utf-8').splitlines()
              if line.strip() and not line.strip().startswith('#')}
    text = pyproject.read_text(encoding='utf-8')
    block = re.search(r'^dependencies\s*=\s*\[(.*?)\]', text, flags=re.M | re.S)
    declared_deps = {normalise(item) for item in re.findall(r'"([^"]+)"', block.group(1))} if block else set()
    if wanted != declared_deps:
        problems.append('pyproject dependencies differ from requirements.txt: '
                        f'missing={sorted(wanted - declared_deps)} extra={sorted(declared_deps - wanted)}')
    for license_file in ('LICENSE', 'LICENSES/MIT.txt'):
        if not (ROOT / license_file).is_file():
            problems.append(f'{license_file} is missing')
    return problems


def violations() -> list[str]:
    files = tracked()
    patterns = reuse_annotations()
    return (check_headers(files, patterns) + check_paths(files) +
            check_run_records(files) + check_dependencies())


def main() -> int:
    problems = violations()
    files = tracked()
    declared_files = sum(1 for relative in files
                         if (text := read_text(ROOT / relative)) and declared(text))
    print(f'tracked files: {len(files)}; files carrying an SPDX identifier: {declared_files}')
    print(f'REUSE annotations: {len(reuse_annotations())} patterns')
    if problems:
        print(f'FAIL: {len(problems)} problem(s)')
        for problem in problems:
            print('  -', problem)
        return 1
    print('OK: license identifiers, portable paths and dependency metadata are consistent.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
