# English-only repository

The whole repository is English: four interface views, chart labels, SVG titles, accessibility attributes, form options, loading and empty states, status messages, errors, API notes, documentation, scripts and comments. The page language is `en` and number formatting uses `en-US`.

There is no translation layer. The backend emits English text directly, so `web/i18n.js` only guards presentation: `EnglishUI.message()` replaces a payload that is not English with an English fallback, and records it in `EnglishUI.untranslated` so the substituted string can be inspected. `EnglishUI.translate()` is the identity function kept for call sites that pass structured API payloads.

`scripts/check_no_chinese.py` fails if any git-tracked text file contains CJK characters, so a future edit cannot silently reintroduce Chinese into the published repository. Binary assets are skipped.

Validation:

- 76 backend tests passed.
- No CJK characters in any tracked text file (repository-wide scan).
- Four views plus cohort differences, HC references, invalid-age messages, transfer animation, upload payloads, historical encryption, mobile layout and network errors were inspected in a browser for non-English text, JavaScript errors and horizontal overflow.
- Six actual FC/BOLD previews, 384 uploaded parameter coordinates and 64 aggregate coordinates remain available. A historical run without a trace still shows no invented parameter values, and the recorded demo run keeps its numeric content unchanged.

Recheck:

```
.venv\Scripts\python.exe -m unittest discover -s tests -q
.venv\Scripts\python.exe scripts/check_no_chinese.py
.venv\Scripts\python.exe scripts/check_english_ui.py
```
