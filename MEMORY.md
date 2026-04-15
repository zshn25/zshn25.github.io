# zshn25.github.io Project Memory

## Stack
- Jekyll + minima remote theme on GitHub Pages
- FontAwesome 5.14.0 (FA6 icons like `fa-x-twitter` won't work — use `fa-twitter`)
- Inter variable font subsetted to Latin (160KB/176KB)
- Dark mode: `assets/css/dark.scss` loaded via `<link media>` + `_includes/applytheme.js`
- `sass: style: compressed` in `_config.yml`

## Architecture
- `style.css` loaded **synchronously** (no async/critical CSS — critical CSS was removed because it caused layout issues)
- `applytheme.js` is an `_include` (inlined via Liquid `{% capture %}`) — NOT `assets/js/`
- Social links config: `_config.yml` → `minima.social_links`

## Visual Design
- Brand purple: `#9966ff` (primary), `#7a51cc` (dark), `#b399ff` (light)
- Accent gold: `#bf8700` (light, Primer attention), `#d4a72c` (dark)
- Featured: `$color-featured: #9a6700` (Primer attention-fg)
- Ghost-style buttons: transparent + gray border/icon, brand-color on hover
- Dark mode links: `#b399ff` (link) / `#9a7ae0` (visited)

## Dark Mode
- Two-state toggle: light ↔ dark (system mode removed)
- On first visit, detects OS preference. Stores in `localStorage("colorScheme")`
- All wrapper/container elements MUST have `background-color: $dm-bg !important` to prevent white horizontal lines at margin/padding boundaries
- Elements covered: `.wrapper`, `.home`, `.post`, `main`, `article`, `footer`, `.post-layout`, `.relatedPosts`, `.topic-section`, `.footer-col-wrapper`

## Patterns & Pitfalls
- CSS specificity: `a:link` (0,1,1) beats single class (0,1,0) — use `!important` on classes that must win
- `overflow:hidden` on `.fixed-top` (not `.header-wrapper`) contains header within 60px
- `.site-title` needs `color: var(--fg-primary) !important` to prevent purple link flash
- Dark mode `!important` cascade: base rules with `!important` override hover rules without it
- All hover rules in dark mode need `!important` on background/border-color
- `$on-palm: 800px` (wider than minima default) — must match in both `custom-variables.scss` and `dark.scss`
- Hardcoded colors should use CSS custom props (`--fg-primary`, `--fg-secondary`, `--fg-muted`, etc.)

## Performance
- Cache TTL 10min is GitHub Pages limitation
- Inter font preload: only regular weight in `<link rel="preload">`
- Do not add `document.body.style.paddingBottom` — caused CLS 1.0
- Wrap scroll-reading in `requestAnimationFrame` to avoid forced reflow
- GIF → WebP conversion with Pillow is worse than original; use cwebp/ffmpeg
