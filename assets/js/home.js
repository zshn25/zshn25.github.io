document.addEventListener('DOMContentLoaded', function() {
  'use strict';

  var POSTS_PER_SECTION = 6;

  // ── Helper: activate a tag chip, expand its section, and scroll to it ──
  function activateAndScroll(chip) {
    if (!chip.classList.contains('active')) chip.click();
    var section = chip.closest('.topic-section');
    if (!section) return;
    var toggle = section.querySelector('.section-toggle');
    if (toggle && toggle.getAttribute('aria-expanded') !== 'true') toggle.click();
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ── Tag filter: multi-select toggle within each section ──
  document.querySelectorAll('.category-chip[data-tag]').forEach(function(chip) {
    chip.addEventListener('click', function(e) {
      e.preventDefault();
      this.classList.toggle('active');
      applyFilters(this.closest('.topic-section'));
    });
  });

  function applyFilters(section) {
    var activeTags = [];
    section.querySelectorAll('.category-chip.active').forEach(function(c) {
      activeTags.push(c.getAttribute('data-tag'));
    });
    var items = section.querySelectorAll('.post-list > li');
    var toggleBtn = section.querySelector('.section-toggle');
    var expanded = toggleBtn && toggleBtn.getAttribute('aria-expanded') === 'true';
    if (activeTags.length === 0) {
      // No filter — restore default collapsed state
      items.forEach(function(item, i) {
        item.classList.remove('tag-hidden');
        if (!expanded && i >= POSTS_PER_SECTION) item.classList.add('section-hidden');
      });
    } else {
      // Filter active — show matching, hide rest, ignore section-hidden
      items.forEach(function(item) {
        var cats = (item.getAttribute('data-categories') || '').split(',');
        var match = activeTags.some(function(t) { return cats.indexOf(t) !== -1; });
        if (match) {
          item.classList.remove('tag-hidden');
          item.classList.remove('section-hidden');
        } else {
          item.classList.add('tag-hidden');
        }
      });
    }
  }

  // ── Show more / Show less toggle per section ──
  document.querySelectorAll('.section-toggle').forEach(function(btn) {
    btn.addEventListener('click', function() {
      var section = this.closest('.topic-section');
      var allItems = section.querySelectorAll('.post-list > li');
      var activeChips = section.querySelectorAll('.category-chip.active');

      // If filters are active, clear them first
      if (activeChips.length > 0) {
        activeChips.forEach(function(c) { c.classList.remove('active'); });
        allItems.forEach(function(li) { li.classList.remove('tag-hidden'); });
        // Restore collapsed state
        if (this.getAttribute('aria-expanded') !== 'true') {
          allItems.forEach(function(li, i) {
            if (i >= POSTS_PER_SECTION) li.classList.add('section-hidden');
          });
        }
        return;
      }

      if (this.getAttribute('aria-expanded') !== 'true') {
        // Expand
        allItems.forEach(function(li) { li.classList.remove('section-hidden'); });
        this.setAttribute('aria-expanded', 'true');
        this.textContent = 'Show less';
      } else {
        // Collapse back
        allItems.forEach(function(li, i) {
          if (i >= POSTS_PER_SECTION) li.classList.add('section-hidden');
        });
        this.setAttribute('aria-expanded', 'false');
        this.textContent = 'Show all ' + this.getAttribute('data-total') + ' posts';
      }
    });
  });

  // ── Tag links in post cards: act as filters ──
  document.querySelectorAll('.post-list .category-tags-link').forEach(function(tagLink) {
    tagLink.addEventListener('click', function(e) {
      e.preventDefault();
      var tag = this.textContent.replace('#', '').trim().toLowerCase();
      var section = this.closest('.topic-section');
      if (!section) return;
      var chip = section.querySelector('.category-chip[data-tag="' + tag + '"]');
      if (chip) {
        chip.click();
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // ── Word cloud tag clicks: activate section filter + scroll ──
  document.querySelectorAll('.tag-cloud a[data-wc-tag]').forEach(function(wcTag) {
    wcTag.addEventListener('click', function(e) {
      e.preventDefault();
      var tag = this.getAttribute('data-wc-tag');
      document.querySelectorAll('.category-chip[data-tag="' + tag + '"]').forEach(function(chip) {
        activateAndScroll(chip);
      });
    });
  });

  // ── Handle ?tag= URL parameter (from post page tag links) ──
  var params = new URLSearchParams(window.location.search);
  var tagParam = params.get('tag');
  if (tagParam) {
    tagParam = tagParam.toLowerCase();
    var scrolled = false;
    document.querySelectorAll('.category-chip[data-tag="' + tagParam + '"]').forEach(function(chip) {
      if (!scrolled) {
        activateAndScroll(chip);
        scrolled = true;
      } else {
        if (!chip.classList.contains('active')) chip.click();
      }
    });
  }
});
