# Self-Hosted Vendor Assets

These assets are self-hosted to eliminate cross-origin requests for user privacy.
No external CDN is contacted on standard page loads.

## Libraries and Licenses

| Library | Version | License | Source |
|---------|---------|---------|--------|
| Primer CSS | 20.x | MIT | https://github.com/primer/css |
| Font Awesome Free | 5.14.0 | Icons: CC BY 4.0, Fonts: SIL OFL 1.1, Code: MIT | https://fontawesome.com/v5/download |
| Academicons | 1.9.x | SIL OFL 1.1 | https://github.com/jpswalsh/academicons |
| KaTeX | 0.16.x | MIT | https://github.com/KaTeX/KaTeX/releases |
| CC License Icons | — | CC BY 4.0 | https://creativecommons.org/about/downloads |
| Inter Font | 4.1 | SIL OFL 1.1 | https://github.com/rsms/inter/releases |
| GLightbox | 3.3.0 | MIT | https://github.com/biati-digital/glightbox |

## How to Update

1. **Primer CSS**: Download from https://unpkg.com/@primer/css/dist/primer.css
   - Replace `assets/vendor/primer/primer.css`

2. **Font Awesome**: Download Web package from https://fontawesome.com/v5/download
   - Replace `assets/vendor/fontawesome/css/all.min.css`
   - Replace woff2 files in `assets/vendor/fontawesome/webfonts/`
   - Update font paths in CSS if they change: should point to `/assets/vendor/fontawesome/webfonts/`

3. **Academicons**: Download from https://github.com/jpswalsh/academicons/releases
   - Replace `assets/vendor/academicons/css/academicons.min.css`
   - Replace `assets/vendor/academicons/fonts/academicons.woff`
   - Update font path in CSS: should point to `/assets/vendor/academicons/fonts/`

4. **KaTeX**: Download from https://github.com/KaTeX/KaTeX/releases
   - Replace `assets/vendor/katex/katex.min.css`, `katex.min.js`, `contrib/auto-render.min.js`
   - Replace woff2 files in `assets/vendor/katex/fonts/`
   - Update font paths in CSS: should point to `/assets/vendor/katex/fonts/`

5. **CC Icons**: Download SVGs from https://creativecommons.org/about/downloads
   - Replace SVGs in `assets/vendor/cc-icons/`

6. **Inter Font**: Download from https://github.com/rsms/inter/releases
   - Extract `InterVariable.woff2` and `InterVariable-Italic.woff2` from the `web/` folder
   - Replace files in `assets/vendor/inter/`

7. **GLightbox**: Download from https://github.com/biati-digital/glightbox/releases
   - Replace `assets/vendor/glightbox/glightbox.min.css` and `glightbox.min.js`

### Important Notes

- After updating any CSS file, verify font paths use absolute paths (`/assets/vendor/...`)
  rather than relative paths, since the CSS is served from `/assets/css/` or `/assets/vendor/`.
- Test locally with `bundle exec jekyll serve` and check the browser Network tab
  to confirm zero cross-origin requests on standard pages.
- All libraries chosen use permissive licenses (MIT, SIL OFL, CC BY) that allow redistribution.
