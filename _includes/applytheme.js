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
    // Update theme-color meta for mobile browser chrome
    var metas = document.querySelectorAll('meta[name="theme-color"]');
    if (metas.length >= 2) {
      if (mode === "dark") { metas[0].content = "#181818"; metas[1].content = "#181818"; }
      else if (mode === "light") { metas[0].content = "#ffffff"; metas[1].content = "#ffffff"; }
      else { metas[0].content = "#ffffff"; metas[1].content = "#181818"; }
    }
  }

  var LABELS = { dark: "Switch to System theme", light: "Switch to Dark theme", system: "Switch to Light theme" };
  var ICONS = { dark: "fa-sun", light: "fa-moon", system: "fa-adjust" };
  var MODE_LABELS = { dark: "Light mode", light: "Dark mode", system: "System theme" };

  function updateToggle(el, iconSel, mode, showLabel) {
    if (!el) return;
    var icon = el.querySelector(iconSel);
    if (icon) {
      icon.classList.remove("fa-moon", "fa-sun", "fa-adjust");
      icon.classList.add(ICONS[mode] || "fa-adjust");
    }
    el.setAttribute("aria-label", LABELS[mode] || "Toggle theme");
    if (!showLabel) el.title = LABELS[mode] || "Toggle theme";
    if (showLabel) {
      var label = el.querySelector(".nav-theme-label-mobile");
      if (label) label.textContent = MODE_LABELS[mode] || "Theme";
    }
  }

  function updateIcon(mode) {
    updateToggle(document.getElementById("nav-switch-theme"), ".nav-theme-icon", mode, false);
    updateToggle(document.getElementById("nav-switch-theme-mobile"), ".nav-theme-icon-mobile", mode, true);
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
    var toggleM = document.getElementById("nav-switch-theme-mobile");
    if (toggleM) {
      toggleM.style.display = "";
      toggleM.addEventListener("click", cycle);
    }
    updateIcon(stored || "system");
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();

  // ── Shared utilities (used by main.js, consent-gate.js, utterances.html) ──
  window.isDarkMode = function() {
    if (!darkLink) return false;
    return darkLink.media === '' || (darkLink.media === '(prefers-color-scheme: dark)' && window.matchMedia('(prefers-color-scheme: dark)').matches);
  };
})();
