/* MathJax: configured here and loaded from the CDN only on pages that contain an equation. */
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

(function () {
  "use strict";
  const SRC = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js";
  let requested = false;
  function typeset() {
    if (!document.querySelector(".arithmatex")) return;
    const mj = window.MathJax;
    if (!mj.startup || !mj.typesetPromise) {
      if (!requested) {
        requested = true;
        const s = document.createElement("script");
        s.src = SRC;
        s.async = true;
        document.head.appendChild(s); // typesets the page by itself once loaded
      }
      return;
    }
    mj.startup.output.clearCache();
    mj.typesetClear();
    mj.texReset();
    mj.typesetPromise();
  }
  if (window.document$ && window.document$.subscribe) window.document$.subscribe(typeset);
  else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", typeset);
  else typeset();
})();
