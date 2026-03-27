#!/usr/bin/env python3
"""Convert fastpages-style notebook markdown to Jekyll posts with front matter.

After `jupyter nbconvert --to markdown`, the first cell of a fastpages notebook
is dumped as plain markdown:

    # Title
    > Description
    - key: value
    - key: value

This script:
1. Extracts fastpages metadata → proper Jekyll YAML front matter
2. Converts #collapse-hide / #collapse-show directives → <details> HTML
3. Sets layout: notebook for badge support
"""

import re
import sys
from pathlib import Path


def parse_fastpages_header(lines):
    """Parse fastpages-style metadata from the beginning of converted markdown.

    Returns (title, description, meta_dict, remaining_lines).
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
        if stripped.startswith("> ") and title and not meta:
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


def build_front_matter(title, description, meta, nb_filename=None):
    """Build Jekyll YAML front matter string."""
    fm_lines = ["---"]
    fm_lines.append("layout: notebook")

    if title:
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
            cats = value.strip("[]").split(",")
            value = " ".join(c.strip() for c in cats)
        fm_lines.append(f"{key}: {value}")

    # Ensure published and comments
    if not any(k.lower() == "published" for k in meta):
        fm_lines.append("published: true")
    if not any(k.lower() == "comments" for k in meta):
        fm_lines.append("comments: true")

    # Add notebook path for Colab/GitHub/Binder badges
    if nb_filename:
        fm_lines.append(f"nb_path: _notebooks/{nb_filename}")

    fm_lines.append("---")
    return "\n".join(fm_lines)


def process_collapse_directives(content):
    """Convert #collapse-hide / #collapse-show in fenced code blocks to <details>.

    Looks for patterns like:
        ```python
        #collapse-hide
        code...
        ```

    Converts to:
        <details class="cell-collapse">
        <summary>Show code</summary>

        ```python
        code...
        ```

        </details>
    """
    # Match fenced code blocks that start with #collapse-hide or #collapse-show
    pattern = re.compile(
        r'^(```\w*)\n'           # opening fence with optional language
        r'#collapse-(hide|show)\n'  # collapse directive
        r'(.*?)'                 # code content
        r'^(```)\s*$',           # closing fence
        re.MULTILINE | re.DOTALL
    )

    def replacer(m):
        fence_open = m.group(1)
        mode = m.group(2)  # "hide" or "show"
        code = m.group(3)
        fence_close = m.group(4)

        # markdown="1" tells kramdown to parse markdown (fenced code) inside <details>
        if mode == "hide":
            return (
                f'<details class="cell-collapse" markdown="1">\n'
                f'<summary>Show code</summary>\n\n'
                f'{fence_open}\n{code}{fence_close}\n\n'
                f'</details>'
            )
        else:  # show
            return (
                f'<details class="cell-collapse" open markdown="1">\n'
                f'<summary>Hide code</summary>\n\n'
                f'{fence_open}\n{code}{fence_close}\n\n'
                f'</details>'
            )

    return pattern.sub(replacer, content)


def process_file(filepath):
    """Process a single markdown file: add front matter + collapse directives."""
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

    # Derive original notebook filename from post filename
    # e.g. 2021-02-01-ResNet-feature-pyramid-in-Pytorch.md → same.ipynb
    nb_filename = path.stem + ".ipynb"

    front_matter = build_front_matter(title, description, meta, nb_filename)
    body = "\n".join(body_lines)

    # Process collapse directives in the body
    body = process_collapse_directives(body)

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
