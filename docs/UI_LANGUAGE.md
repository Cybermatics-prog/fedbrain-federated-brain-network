# English interface

All four views use English: Health Data, Training, Model Inference, and Study Protocol. Chart labels, SVG titles, accessibility labels, form options, loading/empty states, status messages and errors use English. The page language is `en`, and number formatting uses `en-US`.

Recorded API notes are translated for presentation by `window.EnglishUI`. The translator preserves JSON keys, numbers, arrays, booleans and null values. Source experiment results and research records are not rewritten. Unknown non-English error messages receive an English fallback; unknown notes remain inspectable through the translation inventory rather than being fabricated.

Validation:

- 76 backend tests passed.
- 45 inventoried API notes translated without changing numeric payloads; the inference note is also mapped.
- Four views plus cohort differences, HC references, invalid-age messages, transfer animation, upload payloads, historical encryption, mobile layout and network errors were inspected for Chinese text and accessibility attributes.
- No remaining Chinese interface text, JavaScript errors, unexpected POSTs or horizontal page overflow in those scenarios.
- Six actual FC/BOLD previews, 384 uploaded parameter coordinates and 64 aggregate coordinates remain available. A historical run without a trace still shows no invented parameter values.
- The existing Cloudflare URL was verified in an actual browser. Its read-only restrictions and original expiry were not changed.

Recheck:

```
.venv\Scripts\python.exe scripts/check_english_ui.py
.venv\Scripts\python.exe scripts/check_english_ui.py --public --url https://ladies-interference-stay-statistical.trycloudflare.com
```
