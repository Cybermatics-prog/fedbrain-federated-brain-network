# SPDX-License-Identifier: MIT
"""Add an SPDX license identifier to every tracked source file that can hold a comment.

Idempotent: files that already declare an identifier in their first lines are skipped, as
are binary assets and files covered by REUSE.toml annotations.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IDENTIFIER = 'SPDX-License-Identifier: MIT'

# Suffix -> (comment opener, comment closer); opener '' means a whole-line comment.
STYLES = {
    '.py': ('# ', ''),
    '.sh': ('# ', ''),
    '.ps1': ('# ', ''),
    '.toml': ('# ', ''),
    '.txt': ('# ', ''),
    '.cff': ('# ', ''),
    '.cfg': ('# ', ''),
    '.yml': ('# ', ''),
    '.yaml': ('# ', ''),
    '.js': ('// ', ''),
    '.css': ('/* ', ' */'),
    '.md': ('<!-- ', ' -->'),
    '.html': ('<!-- ', ' -->'),
}
EXTRA_FILES = {'.gitignore': ('# ', ''), '.gitattributes': ('# ', ''), 'Makefile': ('# ', '')}
# Covered by REUSE.toml annotations or are the license text itself.
SKIP = {'LICENSE', 'LICENSES/MIT.txt', 'REUSE.toml', 'docs/article.txt'}


def tracked() -> list[str]:
    result = subprocess.run(['git', 'ls-files'], cwd=ROOT, capture_output=True, text=True, check=True)
    return [line for line in result.stdout.splitlines() if line]


def style_for(path: Path) -> tuple[str, str] | None:
    if path.name in EXTRA_FILES:
        return EXTRA_FILES[path.name]
    return STYLES.get(path.suffix.lower())


def insert(text: str, opener: str, closer: str, path: Path) -> str:
    header = f'{opener}{IDENTIFIER}{closer}'
    if path.suffix.lower() == '.html' and text.lstrip().lower().startswith('<!doctype'):
        first, _, rest = text.partition('\n')
        return f'{first}\n{header}\n{rest}'
    return f'{header}\n{text}'


def has_identifier(text: str) -> bool:
    return IDENTIFIER in '\n'.join(text.splitlines()[:6])


def main() -> int:
    changed, skipped = [], []
    for relative in tracked():
        if relative in SKIP:
            continue
        path = ROOT / relative
        if not path.is_file():
            continue
        style = style_for(path)
        if style is None:
            skipped.append(relative)
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            skipped.append(relative)
            continue
        if has_identifier(text):
            continue
        path.write_text(insert(text, *style, path), encoding='utf-8')
        changed.append(relative)
    print(f'headers added: {len(changed)}')
    for relative in changed:
        print('  +', relative)
    if skipped:
        print(f'not applicable: {len(skipped)}')
        for relative in skipped:
            print('  -', relative)
    return 0


if __name__ == '__main__':
    sys.exit(main())
