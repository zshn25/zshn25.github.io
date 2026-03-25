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

  // Open external links in a new window
  document.querySelectorAll('a[href]').forEach(function(b) {
    if (b.hostname && b.hostname !== location.hostname) {
      b.target = '_blank';
      b.rel = 'noopener noreferrer';
    }
  });

  // ── Detect blockquote attributions and style them ──
  // Matches "-Author", "— Author", "*Author*" as last <p> in a blockquote
  document.querySelectorAll('.post-content blockquote').forEach(function(bq) {
    var lastP = bq.querySelector('p:last-child');
    if (!lastP) return;
    var text = lastP.textContent.trim();
    // Pattern: starts with - or — or – (attribution dash)
    if (/^[-–—]/.test(text)) {
      lastP.innerHTML = lastP.innerHTML.replace(/^[-–—]+\s*/, '— ');
      lastP.classList.add('bq-attribution');
    } else if (lastP.children.length === 1 && lastP.children[0].tagName === 'EM' && bq.querySelectorAll('p').length > 1) {
      // Standalone <em> as last paragraph (e.g., *Nietzsche*)
      lastP.innerHTML = '— ' + lastP.innerHTML;
      lastP.classList.add('bq-attribution');
    }
  });

  // ── Reformat internal-link blockquotes as styled callout cards ──
  // Only converts "checkout my post" or heading-link style blockquotes
  document.querySelectorAll('.post-content blockquote').forEach(function(bq) {
    // Skip blockquotes that contain math
    if (bq.querySelector('.katex, .katex-display') || bq.textContent.indexOf('$$') !== -1) return;
    // Skip blockquotes with multiple paragraphs (real quotes, not ads)
    var paras = bq.querySelectorAll('p');
    if (paras.length > 2) return;
    // Find the primary link inside (either in p or h2/h3)
    var link = bq.querySelector('a[href]');
    if (!link) return;
    var href = link.getAttribute('href');
    // Only convert if it's an internal link (same origin or relative)
    if (href.indexOf('http') === 0 && link.hostname !== location.hostname) return;
    // Skip if link text is just a number (footnote reference)
    var text = link.textContent.trim();
    if (!text || /^\d+$/.test(text)) return;
    // Only convert if the blockquote is short (1-2 child elements, used as an ad)
    var children = bq.children;
    if (children.length > 2) return;
    // Check if it's a "checkout" style or heading style
    var isHeading = !!bq.querySelector('h1, h2, h3, h4');
    var prefix = '';
    if (!isHeading) {
      // Extract any text before the link (e.g., "Checkout my blog post on ")
      var parent = link.parentNode;
      if (parent && parent.textContent) {
        var fullText = parent.textContent;
        var linkIdx = fullText.indexOf(text);
        if (linkIdx > 0) prefix = fullText.substring(0, linkIdx).trim();
      }
      // Skip if no "checkout" prefix and it's not a heading — it's a real quote
      if (!prefix && !isHeading) return;
    }
    // Create the callout card
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

  // ── Inline header search (lunr.js) ──
  initHeaderSearch();

}, false);

function initHeaderSearch() {
  var input = document.querySelector('.header-search-input');
  var dropdown = document.querySelector('.search-results-dropdown');
  if (!input || !dropdown || typeof lunr === 'undefined') return;

  var searchData = null;
  var searchIndex = null;

  // Lazy-load search data on first focus
  input.addEventListener('focus', function() {
    if (searchData) return;
    var req = new XMLHttpRequest();
    req.open('GET', '/assets/js/search-data.json', true);
    req.onload = function() {
      if (req.status >= 200 && req.status < 400) {
        searchData = JSON.parse(req.responseText);
        searchIndex = lunr(function() {
          this.ref('id');
          this.field('title', { boost: 200 });
          this.field('content', { boost: 2 });
          this.metadataWhitelist = ['position'];
          for (var i in searchData) {
            this.add({ id: i, title: searchData[i].title, content: searchData[i].content });
          }
        });
      }
    };
    req.send();
  });

  var debounceTimer;
  input.addEventListener('input', function() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(function() { performSearch(input.value); }, 150);
  });

  // Keyboard navigation
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
      hideDropdown();
      input.blur();
    }
  });

  // Close dropdown on outside click
  document.addEventListener('click', function(e) {
    if (!e.target.closest('.header-search')) hideDropdown();
  });

  function hideDropdown() {
    dropdown.innerHTML = '';
    dropdown.classList.remove('active');
  }

  function performSearch(query) {
    hideDropdown();
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
}
