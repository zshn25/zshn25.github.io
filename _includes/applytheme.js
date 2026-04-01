// Theme toggle: light ↔ dark (two states only)
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
    else darkLink.setAttribute("media", "not all");
    // Update theme-color meta for mobile browser chrome
    var metas = document.querySelectorAll('meta[name="theme-color"]');
    if (metas.length >= 2) {
      if (mode === "dark") { metas[0].content = "#181818"; metas[1].content = "#181818"; }
      else { metas[0].content = "#ffffff"; metas[1].content = "#ffffff"; }
    }
  }

  // Icon shows what you'll switch TO: moon = "click for dark", sun = "click for light"
  var LABELS = { dark: "Switch to Light mode", light: "Switch to Dark mode" };
  var ICONS = { dark: "fa-sun", light: "fa-moon" };
  var MODE_LABELS = { dark: "Light mode", light: "Dark mode" };

  function updateToggle(el, iconSel, mode, showLabel) {
    if (!el) return;
    var icon = el.querySelector(iconSel);
    if (icon) {
      icon.classList.remove("fa-moon", "fa-sun", "fa-adjust");
      icon.classList.add(ICONS[mode] || "fa-moon");
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

  function resolveMode() {
    var stored = getStored();
    if (stored === "dark" || stored === "light") return stored;
    // No stored preference — use system preference
    return (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) ? "dark" : "light";
  }

  function toggle() {
    var current = resolveMode();
    var next = current === "dark" ? "light" : "dark";
    setStored(next);
    apply(next);
    updateIcon(next);
  }

  // Initial apply
  var mode = resolveMode();
  apply(mode);

  function init() {
    var toggleBtn = document.getElementById("nav-switch-theme");
    if (toggleBtn) {
      toggleBtn.style.display = "";
      toggleBtn.addEventListener("click", toggle);
    }
    var toggleM = document.getElementById("nav-switch-theme-mobile");
    if (toggleM) {
      toggleM.style.display = "";
      toggleM.addEventListener("click", toggle);
    }
    updateIcon(mode);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();

  // ── Shared utilities (used by main.js, consent-gate.js, utterances.html) ──
  window.isDarkMode = function() {
    if (!darkLink) return false;
    return darkLink.media === '' || (darkLink.media === '(prefers-color-scheme: dark)' && window.matchMedia('(prefers-color-scheme: dark)').matches);
  };
})();
