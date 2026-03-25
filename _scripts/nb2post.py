#!/usr/bin/env python3
"""Convert fastpages-style notebook markdown to Jekyll posts with front matter.

After `jupyter nbconvert --to markdown`, the first cell of a fastpages notebook
is dumped as plain markdown:

    # Title
    > Description
    - key: value
    - key: value

This script reads such a .md file, extracts the fastpages metadata, and rewrites
the file with proper Jekyll YAML front matter:

    ---
    layout: post
    title: "Title"
    description: "Description"
    key: value
    ---
"""

import re
import sys
from pathlib import Path


def parse_fastpages_header(lines):
    """Parse fastpages-style metadata from the beginning of converted markdown.

    Returns (front_matter_dict, description, remaining_lines).
    """
    meta = {}
    description = ""
    title = ""
    body_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Title: # Some Title
        if i == 0 and stripped.startswith("# "):
            title = stripped[2:].strip()
            continue

        # Description: > Some description
        if stripped.startswith("> ") and not title == "" and not meta:
            description = stripped[2:].strip()
            continue

        # Metadata: - key: value
        m = re.match(r"^-\s+(\w[\w\s]*?):\s*(.*)", stripped)
        if m:
            key = m.group(1).strip()
            value = m.group(2).strip()
            meta[key] = value
            continue

        # Empty line between header items
        if stripped == "" and i < 15:
            continue

        # First non-metadata line → body starts here
        body_start = i
        break

    return title, description, meta, lines[body_start:]


def build_front_matter(title, description, meta):
    """Build Jekyll YAML front matter string."""
    fm_lines = ["---"]
    fm_lines.append("layout: post")

    if title:
        # Escape quotes in title
        safe_title = title.replace('"', '\\"')
        fm_lines.append(f'title: "{safe_title}"')

    if description:
        safe_desc = description.replace('"', '\\"')
        fm_lines.append(f'description: "{safe_desc}"')

    # Map fastpages keys to Jekyll keys
    skip_keys = {"toc", "badges", "hide", "search_exclude", "sticky_rank"}
    for key, value in meta.items():
        if key.lower() in skip_keys:
            continue
        # Handle categories: [a, b, c] format
        if key == "categories" and value.startswith("["):
            # Convert [a, b, c] to space-separated
            cats = value.strip("[]").split(",")
            value = " ".join(c.strip() for c in cats)
        fm_lines.append(f"{key}: {value}")

    # Ensure layout and published are present
    has_published = any(k.lower() == "published" for k in meta)
    if not has_published:
        fm_lines.append("published: true")

    has_comments = any(k.lower() == "comments" for k in meta)
    if not has_comments:
        fm_lines.append("comments: true")

    fm_lines.append("---")
    return "\n".join(fm_lines)


def process_file(filepath):
    """Process a single markdown file: add Jekyll front matter."""
    path = Path(filepath)
    content = path.read_text(encoding="utf-8")
    lines = content.split("\n")

    # Check if file already has front matter
    if lines and lines[0].strip() == "---":
        print(f"  [skip] {path.name} already has front matter")
        return

    title, description, meta, body_lines = parse_fastpages_header(lines)

    if not title:
        print(f"  [skip] {path.name} — no fastpages title found")
        return

    front_matter = build_front_matter(title, description, meta)
    body = "\n".join(body_lines)

    # Write back
    path.write_text(front_matter + "\n\n" + body, encoding="utf-8")
    print(f"  [done] {path.name} — title: {title!r}")


def main():
    if len(sys.argv) < 2:
        print("Usage: nb2post.py <file.md> [file2.md ...]")
        sys.exit(1)

    for filepath in sys.argv[1:]:
        process_file(filepath)


if __name__ == "__main__":
    main()
