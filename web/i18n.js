// SPDX-License-Identifier: MIT
/* Presentation guard: the recorded API text is English, so nothing is translated here.
   message() still filters a non-English error payload so a mixed-language string can never
   reach the interface; the fallback keeps the UI readable when that happens. */
(() => {
  'use strict';
  const han = /[\u3400-\u9fff\u3000-\u303f\uff00-\uffef]/;
  const untranslated = new Set();
  function translate(value) {
    return value;
  }
  function message(value, fallback = 'Request failed. Please retry.') {
    let text;
    try {
      text = typeof value === 'string' ? value : JSON.stringify(value ?? '');
    } catch (error) {
      text = '';
    }
    text = String(text ?? '').trim();
    if (!text) return fallback;
    if (han.test(text)) {
      untranslated.add(text);
      return fallback;
    }
    return text;
  }
  window.EnglishUI = { translate, message, untranslated };
})();
