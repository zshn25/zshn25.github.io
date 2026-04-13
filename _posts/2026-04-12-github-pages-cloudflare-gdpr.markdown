---
layout: post
title:  "Making GitHub Pages GDPR-compliant with Cloudflare"
description: "How to use Cloudflare's free proxy to get GDPR compliance, CDN performance, DDoS protection, and free SSL for your GitHub Pages site."
# image: images/cloudflare-github-pages.png
date:   2026-04-12 17:00:00 -0700
categories: privacy web-hosting devops
author: Zeeshan Khan Suri
published: false
comments: true
---

GitHub Pages is the easiest way to host a static site for free. But if you serve European visitors, there's a problem: [GitHub collects IP addresses](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages#data-collection){:target="_blank"} of all visitors for "security purposes." Under GDPR, an IP address is personal data, and transferring it to US-based GitHub servers without explicit consent is legally questionable[^1].

The fix is straightforward: put Cloudflare's free reverse proxy in front of GitHub Pages. European visitors connect to Cloudflare's EU edge servers — their IP never reaches GitHub.

This post walks through the complete setup and explains what you get beyond GDPR compliance.

## What Cloudflare's proxy does

When you route traffic through Cloudflare (orange cloud / "Proxied" mode), the request flow changes:

```
User → Cloudflare Edge (EU/nearest) → GitHub Pages (US)
```

Cloudflare terminates the TLS connection and creates a *new* connection to GitHub. GitHub sees **Cloudflare's IP**, not the visitor's. The visitor's real IP never leaves Cloudflare's network.

This also means:
- **Caching**: Cloudflare can cache your static files at 300+ edge locations. Subsequent requests are served from the nearest edge — no round-trip to GitHub.
- **DDoS protection**: Cloudflare's free tier includes Layer 3/4 DDoS mitigation and rate limiting.
- **Free SSL**: Cloudflare provides and auto-renews an SSL certificate for your domain.
- **Analytics**: Basic traffic analytics on the Cloudflare dashboard, no cookies or JS required.

## Setup guide

### 1. Add your domain to Cloudflare

Sign up at [dash.cloudflare.com](https://dash.cloudflare.com){:target="_blank"} (free plan). Click **Add a site**, enter your domain. Cloudflare will scan your existing DNS records.

### 2. Update nameservers

Cloudflare assigns you two nameservers. Go to your domain registrar and replace the existing nameservers. Propagation takes 5–60 minutes. You can verify with:

```bash
dig NS yourdomain.xyz +short
```

### 3. Configure DNS records

In Cloudflare's DNS settings, you need records pointing to GitHub Pages. For an **apex domain** (e.g., `wordfor.xyz`), use A records pointing to GitHub's servers:

| Type | Name | Value | Proxy |
|------|------|-------|-------|
| A | `@` | `185.199.108.153` | Proxied |
| A | `@` | `185.199.109.153` | Proxied |
| A | `@` | `185.199.110.153` | Proxied |
| A | `@` | `185.199.111.153` | Proxied |
| CNAME | `www` | `yourusername.github.io` | Proxied |

The critical part is that the **Proxy** column shows the orange cloud (Proxied), not "DNS only". If set to DNS only, visitors connect directly to GitHub and the proxy benefits are lost.

{% include info.html content="You might see advice to use a CNAME for the apex domain instead of A records. While Cloudflare supports CNAME flattening (which makes this work), A records are simpler and what GitHub officially recommends for apex domains." %}

### 4. SSL/TLS settings

In Cloudflare → **SSL/TLS**:

- **Encryption mode: Full** (not "Full (strict)"). GitHub Pages uses its own certificate which doesn't match your custom domain, so strict mode would reject it.
- **Edge Certificates → Always Use HTTPS**: Enable this. All HTTP requests get redirected to HTTPS at the edge.

### 5. Configure your GitHub repository

In your repo → **Settings → Pages**:

1. Set the custom domain to your apex domain (e.g., `wordfor.xyz`)
2. Make sure there's a `CNAME` file in the repo root containing your domain
3. GitHub may show a DNS warning briefly — this resolves once Cloudflare propagation completes

### 6. Cache rules for static assets

Cloudflare caches HTML with a short TTL by default. For large static assets (binary files, model weights) you want aggressive caching.

In **Rules → Cache Rules**, create a rule:

**Expression:**
```
(http.request.uri.path wildcard r"/data/*") or (http.request.uri.path wildcard r"/models/*")
```

**Action:**
- Cache eligibility: **Eligible for cache**
- Edge TTL: **Override origin, 1 month**
- Browser TTL: **Override origin, 1 year**

This ensures your large files are cached at Cloudflare's edge and in the browser. Users who revisit won't re-download them.

### 7. Redirect www → apex (optional)

In **Rules → Redirect Rules**, create a rule:

- **When:** `(http.host eq "www.yourdomain.xyz")`
- **Then:** Dynamic redirect to `concat("https://yourdomain.xyz", http.request.uri.path)`
- **Status code:** 301 (permanent)

This consolidates traffic on a single canonical URL, which is also better for SEO.

## What about GitHub's IP collection?

With Cloudflare proxying, GitHub sees only Cloudflare's IP addresses. From GitHub's perspective, all your traffic comes from Cloudflare's servers in a few data centers. The visitor's actual IP is contained within Cloudflare's network and governed by Cloudflare's privacy policy, which is GDPR-compliant[^2].

You can verify this is working by checking the server response headers:

```bash
curl -sI https://yourdomain.xyz | grep -i server
# Should show: server: cloudflare
```

If it shows `GitHub.com` instead, your DNS record is not proxied (orange cloud is off).

## Can I restrict GitHub Pages to only accept Cloudflare traffic?

No. GitHub Pages is shared hosting — you don't control the server's firewall. Cloudflare recommends [restricting origin access to their IPs](https://developers.cloudflare.com/fundamentals/concepts/cloudflare-ip-addresses/){:target="_blank"}, but this only applies to servers you control (VPS, cloud instances) where you can configure `iptables` or security groups.

For GitHub Pages this doesn't matter in practice: your site is static files with no secrets. Even if someone accesses `yourusername.github.io` directly, they get the same HTML/CSS/JS. There is no admin panel, API, or database to protect at the origin.

## Summary

| Feature | Without Cloudflare | With Cloudflare (free) |
|---------|-------------------|----------------------|
| GDPR | Visitor IPs sent to GitHub (US) | IPs stay at Cloudflare edge |
| CDN | Single origin (US) | 300+ edge locations |
| DDoS | None | L3/L4 mitigation |
| SSL | GitHub-managed | Cloudflare-managed + auto-renew |
| Cache | Browser only | Edge + browser |
| Cost | Free | Free |

The entire setup takes about 15 minutes. For any static site hosted on GitHub Pages that serves international visitors, there's no reason not to do this.

___

© Zeeshan Khan Suri, [<i class="fab fa-creative-commons"></i> <i class="fab fa-creative-commons-by"></i> <i class="fab fa-creative-commons-nc"></i>](http://creativecommons.org/licenses/by-nc/4.0/)

If this article was helpful to you, consider citing

```bibtex
@misc{suri_github_pages_cloudflare_2026,
      title={Making GitHub Pages GDPR-compliant with Cloudflare},
      url={https://zshn25.github.io/github-pages-cloudflare-gdpr},
      journal={Curiosity},
      author={Suri, Zeeshan Khan},
      year={2026},
      month={Apr}}
```

# References

[^1]: The landmark Schrems II ruling (C-311/18) invalidated the EU-US Privacy Shield. While the EU-US Data Privacy Framework (2023) partially addresses this, the legal status of incidental data collection by US-based hosting providers remains debated. Using an EU-based CDN proxy is the safest approach.

[^2]: Cloudflare's [privacy policy](https://www.cloudflare.com/privacypolicy/){:target="_blank"} and [GDPR compliance page](https://www.cloudflare.com/gdpr/introduction/){:target="_blank"} detail their data processing practices. They are registered under the EU-US Data Privacy Framework and offer Data Processing Agreements.

*[GDPR]: General Data Protection Regulation
*[CDN]: Content Delivery Network
*[DDoS]: Distributed Denial of Service
*[TLS]: Transport Layer Security
*[SSL]: Secure Sockets Layer
