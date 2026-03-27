// Back button after anchor link should return to previous page and not to same page
// From https://github.com/allejo/jekyll-anchor-headings/discussions/31#discussioncomment-564411
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.heading-anchor').forEach(function (el) {
    el.addEventListener('click', function (event) {
      event.preventDefault();
      var hash = el.getAttribute('href');
      window.location.replace(('' + window.location).split('#')[0] + hash);
    });
  });

  // Wrap titled images in figure/figcaption (moved from inline script in custom-head.html)
  document.querySelectorAll('.post img').forEach(function(el) {
    if (!el.getAttribute('loading')) el.setAttribute('loading', 'lazy');
    if (el.getAttribute('title') && !el.classList.contains('emoji')) {
      var caption = document.createElement('figcaption');
      caption.textContent = el.getAttribute('title');
      var wrapper = document.createElement('figure');
      wrapper.className = 'image';
      el.parentNode.insertBefore(wrapper, el);
      wrapper.appendChild(el);
      wrapper.appendChild(caption);
    }
  });

  // Open external links in a new window
  document.querySelectorAll('a[href]').forEach(function(b) {
    if (b.hostname && b.hostname !== location.hostname) {
      b.target = '_blank';
      b.rel = 'noopener noreferrer';
    }
  });

  // ── Close hamburger menu on outside click ──
  var navTrigger = document.getElementById('nav-trigger');
  if (navTrigger) {
    document.addEventListener('click', function(e) {
      if (navTrigger.checked && !e.target.closest('.site-nav')) {
        navTrigger.checked = false;
      }
    });
  }

  // ── Detect blockquote attributions and style them ──
  document.querySelectorAll('.post-content blockquote').forEach(function(bq) {
    var lastP = bq.querySelector('p:last-child');
    if (!lastP) return;
    var text = lastP.textContent.trim();
    if (/^[-–—]/.test(text)) {
      lastP.innerHTML = lastP.innerHTML.replace(/^[-–—]+\s*/, '— ');
      lastP.classList.add('bq-attribution');
    } else if (lastP.children.length === 1 && lastP.children[0].tagName === 'EM' && bq.querySelectorAll('p').length > 1) {
      lastP.innerHTML = '— ' + lastP.innerHTML;
      lastP.classList.add('bq-attribution');
    }
  });

  // ── Reformat internal-link blockquotes as styled callout cards ──
  document.querySelectorAll('.post-content blockquote').forEach(function(bq) {
    if (bq.querySelector('.katex, .katex-display') || bq.textContent.indexOf('$$') !== -1) return;
    var paras = bq.querySelectorAll('p');
    if (paras.length > 2) return;
    var link = bq.querySelector('a[href]');
    if (!link) return;
    var href = link.getAttribute('href');
    if (href.indexOf('http') === 0 && link.hostname !== location.hostname) return;
    var text = link.textContent.trim();
    if (!text || /^\d+$/.test(text)) return;
    var children = bq.children;
    if (children.length > 2) return;
    var isHeading = !!bq.querySelector('h1, h2, h3, h4');
    var prefix = '';
    if (!isHeading) {
      var parent = link.parentNode;
      if (parent && parent.textContent) {
        var fullText = parent.textContent;
        var linkIdx = fullText.indexOf(text);
        if (linkIdx > 0) prefix = fullText.substring(0, linkIdx).trim();
      }
      if (!prefix && !isHeading) return;
    }
    var callout = document.createElement('div');
    callout.className = 'inline-post-callout';
    callout.innerHTML =
      '<a href="' + href + '" class="inline-post-callout-link">' +
        '<span class="inline-post-callout-icon"><i class="fas fa-book-reader"></i></span>' +
        '<span class="inline-post-callout-body">' +
          (prefix ? '<span class="inline-post-callout-prefix">' + prefix + '</span>' : '') +
          '<span class="inline-post-callout-title">' + text + '</span>' +
        '</span>' +
        '<span class="inline-post-callout-arrow"><i class="fas fa-arrow-right"></i></span>' +
      '</a>';
    bq.parentNode.replaceChild(callout, bq);
  });

  // ── Back to top button ──
  var backBtn = document.getElementById('back-to-top');
  if (backBtn) {
    var privacyNotice = document.getElementById('privacy-notice');
    function updateBackBtn() {
      if (window.scrollY > 400) backBtn.classList.add('visible');
      else backBtn.classList.remove('visible');
      // Push above privacy banner if it's visible
      if (privacyNotice && privacyNotice.style.display === 'flex') {
        backBtn.classList.add('above-banner');
      } else {
        backBtn.classList.remove('above-banner');
      }
    }
    window.addEventListener('scroll', updateBackBtn, { passive: true });
    updateBackBtn();
    backBtn.addEventListener('click', function() {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  // ── Dark mode: image handling ──
  // data-transparent-dark="true"  → light card background (preserves text readability, works on GIFs/videos)
  // data-invert-dark="true"       → invert filter
  // data-no-invert                → skip entirely
  // No attribute                  → auto-detect via canvas sampling
  var isDark = typeof window.isDarkMode === 'function' ? window.isDarkMode() : false;
  if (isDark) {
    var INVERT_FILTER = 'invert(0.88) hue-rotate(180deg)';
    var sampleCanvas = document.createElement('canvas');
    sampleCanvas.width = 50;
    sampleCanvas.height = 50;
    var sampleCtx = sampleCanvas.getContext('2d');

    function analyzeAndInvert(img) {
      if (img.hasAttribute('data-no-invert')) return;
      if (img.getAttribute('data-transparent-dark') === 'true') {
        // CSS-only: light card background — works on img, GIF, video, cross-origin
        img.style.background = 'rgba(255, 255, 255, 0.92)';
        img.style.borderRadius = 'var(--radius-md, 6px)';
        img.style.padding = '4px';
        return;
      }
      if (img.getAttribute('data-invert-dark') === 'true') {
        img.style.filter = INVERT_FILTER;
        return;
      }
      try {
        sampleCtx.clearRect(0, 0, 50, 50);
        sampleCtx.drawImage(img, 0, 0, 50, 50);
        var data = sampleCtx.getImageData(0, 0, 50, 50).data;
        var whiteOrTransparent = 0;
        var total = 2500;
        for (var i = 0; i < data.length; i += 4) {
          var r = data[i], g = data[i+1], b = data[i+2], a = data[i+3];
          if (a < 30 || (r > 240 && g > 240 && b > 240)) whiteOrTransparent++;
        }
        if (whiteOrTransparent / total > 0.7) {
          img.style.filter = INVERT_FILTER;
        }
      } catch(e) { /* cross-origin images — skip */ }
    }

    document.querySelectorAll('.post-content img').forEach(function(img) {
      if (img.complete && img.naturalWidth > 0) analyzeAndInvert(img);
      else img.addEventListener('load', function() { analyzeAndInvert(img); });
    });

    document.querySelectorAll('iframe[src*="ourworldindata"]').forEach(function(f) {
      f.classList.add('owid-dark');
    });
  }

  // ── Inline header search (lunr.js) ──
  initHeaderSearch();

  // ── Copy button for all code blocks ──
  var COPY_SVG = '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"/><path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"/></svg>';
  var CHECK_SVG = '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"/></svg>';
  document.querySelectorAll('.post-content pre > code, .highlight pre').forEach(function(codeEl) {
    var pre = codeEl.tagName === 'PRE' ? codeEl : codeEl.parentNode;
    if (!pre || pre.tagName !== 'PRE') return;
    if (pre.querySelector('.copy-code-btn')) return;
    pre.style.position = 'relative';
    var btn = document.createElement('button');
    btn.className = 'copy-code-btn';
    btn.setAttribute('aria-label', 'Copy code');
    btn.innerHTML = COPY_SVG;
    btn.addEventListener('click', function() {
      var text = (codeEl.tagName === 'CODE' ? codeEl : pre.querySelector('code') || pre).textContent;
      navigator.clipboard.writeText(text).then(function() {
        btn.innerHTML = CHECK_SVG;
        setTimeout(function() { btn.innerHTML = COPY_SVG; }, 1500);
      });
    });
    pre.appendChild(btn);
  });

  // ── Table of Contents — scroll-based active section tracking ──
  var tocLinks = document.querySelectorAll('.toc-nav .toc-link');
  if (tocLinks.length > 1) {
    var headingIds = [];
    tocLinks.forEach(function(link) {
      var href = link.getAttribute('href');
      if (href && href.charAt(0) === '#') headingIds.push(href.slice(1));
    });

    var currentActive = null;
    var observer = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          var id = entry.target.id;
          if (currentActive) currentActive.classList.remove('active');
          var link = document.querySelector('.toc-nav .toc-link[href="#' + id + '"]');
          if (link) {
            link.classList.add('active');
            currentActive = link;
          }
        }
      });
    }, { rootMargin: '-80px 0px -70% 0px', threshold: 0 });

    headingIds.forEach(function(id) {
      var el = document.getElementById(id);
      if (el) observer.observe(el);
    });
  }

}, false);

function initHeaderSearch() {
  if (typeof lunr === 'undefined') return;
  var containers = document.querySelectorAll('.header-search');
  if (!containers.length) return;

  var searchData = null;
  var searchIndex = null;

  function loadSearchData(cb) {
    if (searchData) { if (cb) cb(); return; }
    var req = new XMLHttpRequest();
    req.open('GET', '/assets/js/search-data.json', true);
    req.onload = function() {
      if (req.status >= 200 && req.status < 400) {
        try { searchData = JSON.parse(req.responseText); }
        catch (e) { console.error('Search data parse error:', e); return; }
        searchIndex = lunr(function() {
          this.ref('id');
          this.field('title', { boost: 200 });
          this.field('content', { boost: 2 });
          this.metadataWhitelist = ['position'];
          for (var i in searchData) {
            this.add({ id: i, title: searchData[i].title, content: searchData[i].content });
          }
        });
        if (cb) cb();
      }
    };
    req.send();
  }

  function hideAllDropdowns() {
    containers.forEach(function(c) {
      var dd = c.querySelector('.search-results-dropdown');
      if (dd) { dd.innerHTML = ''; dd.classList.remove('active'); }
    });
  }

  function showResults(dropdown, query) {
    dropdown.innerHTML = '';
    dropdown.classList.remove('active');
    if (!query || !searchIndex) return;
    var results = searchIndex.query(function(q) {
      var tokens = lunr.tokenizer(query);
      q.term(tokens, { boost: 10 });
      q.term(tokens, { wildcard: lunr.Query.wildcard.TRAILING });
    });
    if (results.length === 0) return;
    dropdown.classList.add('active');
    var shown = Math.min(results.length, 8);
    for (var i = 0; i < shown; i++) {
      var doc = searchData[results[i].ref];
      if (!doc) continue;
      var a = document.createElement('a');
      a.href = doc.url;
      var title = document.createElement('div');
      title.className = 'search-result-title';
      title.textContent = doc.title;
      a.appendChild(title);
      if (doc.date) {
        var date = document.createElement('span');
        date.className = 'search-result-preview';
        date.textContent = doc.date;
        a.appendChild(date);
      }
      dropdown.appendChild(a);
    }
  }

  containers.forEach(function(container) {
    var input = container.querySelector('.header-search-input');
    var dropdown = container.querySelector('.search-results-dropdown');
    if (!input || !dropdown) return;

    input.addEventListener('focus', function() { loadSearchData(); });

    var debounceTimer;
    input.addEventListener('input', function() {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(function() { showResults(dropdown, input.value); }, 150);
    });

    input.addEventListener('keydown', function(e) {
      var items = dropdown.querySelectorAll('a');
      var active = dropdown.querySelector('a.active');
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (active) { active.classList.remove('active'); var next = active.nextElementSibling; if (next) next.classList.add('active'); }
        else if (items.length) items[0].classList.add('active');
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        if (active) { active.classList.remove('active'); var prev = active.previousElementSibling; if (prev) prev.classList.add('active'); }
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (active) window.location.href = active.getAttribute('href');
        else if (items.length) window.location.href = items[0].getAttribute('href');
      } else if (e.key === 'Escape') {
        hideAllDropdowns();
        input.blur();
      }
    });
  });

  document.addEventListener('click', function(e) {
    if (!e.target.closest('.header-search')) hideAllDropdowns();
  });
}
