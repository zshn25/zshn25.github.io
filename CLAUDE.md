# CLAUDE.md — Agent Instructions for zshn25.github.io

## Stack

- **Jekyll** static site with **minima** remote theme, hosted on **GitHub Pages**
- **FontAwesome 5.14.0** — self-hosted in `assets/vendor/fontawesome/`. FA6 icons (e.g. `fa-x-twitter`) do NOT work; use FA5 names (`fa-twitter`)
- **Inter** variable font — self-hosted, Latin subset (~160KB regular + ~176KB italic) in `assets/vendor/fonts/inter/`
- **KaTeX** — self-hosted for math rendering (`kramdown` + `math_engine: katex`)
- **lunr.js** — client-side search, data generated at `assets/js/search-data.json`
- **Ruby/Bundler**: `bundle exec jekyll serve` for local development
- `sass: style: compressed` in `_config.yml` minifies CSS output

## Architecture Overview

### CSS Loading (no critical CSS)
- `style.css` is loaded **synchronously** (not async). A previous critical CSS approach was removed because it caused layout issues. Do not re-add async CSS loading without a proper critical CSS extraction.
- Dark mode: `assets/css/dark.scss` loaded via `<link id="dark-css" media="(prefers-color-scheme: dark)">`. The `media` attribute is toggled by `applytheme.js` — not a body class.

### JavaScript Load Order (matters!)
1. **`applytheme.js`** — inlined in `<head>` via Liquid `{% capture %}`. Runs synchronously before paint to prevent FOUC. Do NOT move to an external file or add `defer`.
2. **`consent-gate.js`** — loaded synchronously (no `defer`) in `<head>`. Must run before the HTML parser reaches `<script async>` embed tags in post body, so its MutationObserver can intercept them. Do NOT add `defer` — it would let embeds load before interception.
3. **`main.js`** — loaded with `defer`. All DOM manipulation runs inside `DOMContentLoaded`.
4. **`home.js`** — loaded with `defer`, only on the homepage. Client-side tag filtering.

### Dark Mode
- Controlled by changing `#dark-css` link's `media` attribute (`"not all"` = light, removed = dark)
- Two-state toggle: light ↔ dark. On first visit with no stored preference, detects system preference.
- `localStorage` key: `"colorScheme"` (values: `"dark"`, `"light"`, or absent for system default)
- `window.isDarkMode()` global — returns true if dark styles are active
- Dark mode overrides are in `assets/css/dark.scss` — all use `!important` to override light theme
- **White line prevention**: All wrapper/container elements (`.wrapper`, `.home`, `.post`, `main`, `article`, `footer`, `.post-layout`, `.relatedPosts`) must have `background-color: $dm-bg !important` in dark.scss to prevent white bleed-through at margin/padding boundaries

### Consent & Privacy
- **Privacy notice** (`_includes/privacy-notice.html`): Informational banner (NOT cookie consent). Dismissed via `localStorage.setItem('privacyOK','1')`. The site itself sets no cookies; `__gh_sess` is set by GitHub Pages infrastructure.
- **Consent gate** (`assets/js/consent-gate.js`): Blocks Twitter/Reddit/YouTube embeds until user opts in per provider. Stores consent in `localStorage` key `"embedConsent"` (JSON object). Exposes `window._consent` API.
- These are separate systems with separate localStorage keys.

### Homepage (`_layouts/home.html` + `assets/js/home.js`)
- Posts grouped into three sections (tech, philosophy, life) based on `_data/categories.yml` (single source of truth for tag-to-section mapping)
- Only first 6 posts shown per section; rest have `class="section-hidden"` (expand via toggle button)
- Filtering is entirely client-side: Liquid renders all posts with `data-categories="cat1,cat2,..."`, JS toggles visibility
- Word cloud at bottom; tags colored by section (`tag-color-tech`, `tag-color-philosophy`, `tag-color-life`)
- URL `?tag=...` activates corresponding filter on load

### Dialogs
- Both search (`#search-dialog`) and mobile nav (`#nav-dialog`) use native `<dialog>` elements with `showModal()`/`close()`
- Shared `closeOnBackdropClick(dialog)` utility in `main.js`
- Search opens via `.search-trigger` buttons or `Ctrl+K`/`Cmd+K`
- `<dialog>` elements need `aria-labelledby` pointing to a visible title `id`

## Key File Paths

| Area | Files |
|------|-------|
| Config | `_config.yml`, `_data/categories.yml` |
| Layouts | `_layouts/default.html`, `_layouts/home.html`, `_layouts/post.html` |
| Head/Header | `_includes/head.html`, `_includes/custom-head.html`, `_includes/header.html` |
| Theme toggle | `_includes/applytheme.js` (inlined, NOT assets/js/) |
| Footer/Social | `_includes/footer.html`, `_includes/social.html` |
| Privacy/Consent | `_includes/privacy-notice.html`, `assets/js/consent-gate.js` |
| Search | `_includes/search-dialog.html`, `assets/js/main.js` (search section) |
| Styles | `_sass/minima/custom-variables.scss` (variables + custom props), `_sass/minima/custom-styles.scss`, `_sass/minima/fastpages-styles.scss` |
| Dark mode CSS | `assets/css/dark.scss` |
| JavaScript | `assets/js/main.js`, `assets/js/home.js`, `assets/js/consent-gate.js` |
| Post cards | `_includes/post_list_image_card.html` |
| Share buttons | `_includes/share-buttons.html` |
| Notebook badges | `_includes/notebook_badges.html`, `notebook_colab_link.html`, `notebook_github_link.html`, `notebook_kaggle_link.html` |
| Scripts | `_scripts/white2transparent.py` (remove white bg from images), `_scripts/nb2post.py` (notebook to post) |

## CSS Specificity Pitfalls

These have caused repeated bugs — be aware:

1. **`a:link` (0,1,1) beats single class (0,1,0)**: `.category-chip { color: var(--fg-secondary) }` is overridden by `a:link { color: #7a51cc }`. Fix: add `!important` to class rules that must beat `a:link`.
2. **Minima defaults**: The remote theme sets `line-height: 54px` and `min-height: 56px` on `.site-header`, `padding: 2px 4px` on `.social-media-list a`. Override with `!important` when needed.
3. **Dark mode `!important` cascade**: Base rules with `!important` (e.g. `background: transparent !important`) override hover rules without `!important`. All hover rules in dark mode need `!important` on background/border-color.
4. **`$on-palm: 800px`**: Breakpoint is wider than minima's default 600px. Must match in both `custom-variables.scss` and `dark.scss`.

## Visual Design

- **Brand purple**: `#9966ff` (primary), `#7a51cc` (dark), `#b399ff` (light/dark-mode links)
- **Accent gold**: `#bf8700` (light, Primer attention), `#d4a72c` (dark, Primer attention-emphasis)
- **Featured posts**: Left border `3px solid $color-featured` (#9a6700, Primer attention-fg). Needs `!important` to override `.post-list .Box` border shorthand.
- **Ghost-style buttons**: Transparent bg + gray border → brand-color fill on hover (used for social icons `.social-icon--*` and share buttons `.share-btn--*`)
- **CSS custom properties** defined in `custom-variables.scss` `:root`, overridden in `dark.scss` `:root`

## Performance Notes

- `style.css` must stay synchronous (no async/preload pattern without critical CSS)
- Inter font regular weight is preloaded in `custom-head.html`
- Do not add `document.body.style.paddingBottom` or similar layout-shifting JS — caused CLS 1.0
- Wrap scroll-reading code in `requestAnimationFrame` to avoid forced reflow
- Images: prefer WebP, ~600px max width for card thumbnails. Use `_scripts/white2transparent.py` for diagrams.
- KaTeX CSS loaded async via `media="print" onload="this.media='all'"` pattern
- Cache TTL 10min is a GitHub Pages limitation — cannot change

## Content Conventions

- Post front matter: `layout: post`, `title`, `description`, `date`, `image` (card thumbnail), `categories` (space-separated), `comments: true` for Utterances
- `featured: true` in front matter pins a post to the top of its section
- `hide: true` or `hidden: true` excludes from homepage
- External link posts: set `link:` in front matter → renders with external link icon
- Notebook posts: set `toc: true`, `badges: true`, and optionally `colab:`, `github:`, `binder:`, `kaggle:` URLs

## Known Issues / Future Work

- FontAwesome `all.min.css` has ~12KB unused CSS (cannot fix without subsetting)
- `leviathan.webp` (71KB) is already well-optimized; Pillow resize produces marginal savings — use `cwebp` if needed
- `vision_transformer.gif` (133KB) — Pillow animated WebP is larger; needs `cwebp` or `ffmpeg` for conversion
- lunr.js search data loaded on all pages; could lazy-load on search open
- dark.scss linter `{ expected` at line 8 is a false positive — VS Code SCSS parser doesn't understand Jekyll YAML front matter (`---`)
