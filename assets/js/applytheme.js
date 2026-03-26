// Theme toggle: system (auto) → light → dark → system
(function () {
  var STORAGE_KEY = "colorScheme";
  var darkLink = document.querySelector("#dark-css");

  function getStored() {
    try { return localStorage.getItem(STORAGE_KEY); } catch (e) { return null; }
  }

  function setStored(v) {
    try {
      if (v === null) localStorage.removeItem(STORAGE_KEY);
      else localStorage.setItem(STORAGE_KEY, v);
    } catch (e) { /* private browsing */ }
  }

  function apply(mode) {
    if (!darkLink) return;
    if (mode === "dark") darkLink.removeAttribute("media");
    else if (mode === "light") darkLink.setAttribute("media", "not all");
    else darkLink.setAttribute("media", "(prefers-color-scheme: dark)");
  }

  function updateIcon(mode) {
    var toggle = document.getElementById("nav-switch-theme");
    if (!toggle) return;
    var icon = toggle.querySelector(".nav-theme-icon");
    if (!icon) return;
    icon.classList.remove("fa-moon", "fa-sun", "fa-adjust");
    if (mode === "dark") icon.classList.add("fa-sun");
    else if (mode === "light") icon.classList.add("fa-moon");
    else icon.classList.add("fa-adjust");
    toggle.title = { dark: "Switch to System", light: "Switch to Dark", system: "Switch to Light" }[mode] || "Toggle theme";
  }

  function cycle() {
    var stored = getStored();
    var next;
    if (stored === null) next = "light";
    else if (stored === "light") next = "dark";
    else next = null;
    setStored(next);
    apply(next || "system");
    updateIcon(next || "system");
  }

  // Initial apply
  var stored = getStored();
  if (stored === "dark") apply("dark");
  else if (stored === "light") apply("light");

  function init() {
    var toggle = document.getElementById("nav-switch-theme");
    if (toggle) {
      toggle.style.display = "";
      toggle.addEventListener("click", cycle);
    }
    updateIcon(stored || "system");
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();

  // ── Shared utilities (used by main.js, consent-gate.js, utterances.html) ──
  window.isDarkMode = function() {
    var dl = document.getElementById('dark-css');
    if (!dl) return false;
    return dl.media === '' || (dl.media === '(prefers-color-scheme: dark)' && window.matchMedia('(prefers-color-scheme: dark)').matches);
  };
})();
