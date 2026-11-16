"""Fail if any tracked text file contains Chinese, Japanese or Korean characters.

GitHub-visible files must be English-only. Images and other binaries are skipped by
extension, and only git-tracked files are considered so excluded local caches do not
matter.

Usage:
    python scripts/check_no_chinese.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINARY_SUFFIXES = {
    '.png', '.jpg', '.jpeg', '.gif', '.webp', '.ico', '.pdf', '.zip', '.npz', '.npy',
    '.pt', '.pth', '.bin', '.exe', '.woff', '.woff2', '.ttf', '.otf', '.mp4', '.wav',
}
# CJK ideographs, kana, hangul, CJK punctuation and fullwidth forms.
PATTERN = re.compile(r'[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uff00-\uffef]')


def tracked_files() -> list[str]:
    result = subprocess.run(['git', 'ls-files'], cwd=ROOT, capture_output=True, text=True, check=True)
    return [line for line in result.stdout.splitlines() if line]


def main() -> int:
    offenders = []
    for relative in tracked_files():
        path = ROOT / relative
        if path.suffix.lower() in BINARY_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except (UnicodeDecodeError, OSError):
            continue
        matches = PATTERN.findall(text)
        if matches:
            lines = [index for index, line in enumerate(text.splitlines(), 1) if PATTERN.search(line)]
            offenders.append({'file': relative, 'characters': len(matches), 'lines': lines[:8]})
    if offenders:
        print('Non-English text found in ' + str(len(offenders)) + ' file(s):')
        for item in offenders:
            print(f"  {item['file']}: {item['characters']} characters on lines {item['lines']}")
        return 1
    print('OK: no CJK characters in tracked files.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
