# /// script
# dependencies = [
#   "Pillow",
#   "numpy",
# ]
# ///

"""Remove white/near-white background from images, preserving text edges.

Usage:
    python _scripts/white2transparent.py <image> [<image2> ...]
    python _scripts/white2transparent.py images/diagram.png
    python _scripts/white2transparent.py images/chart.jpg --threshold 230
    python _scripts/white2transparent.py images/fig.png --webp        # smaller output
    python _scripts/white2transparent.py images/fig.png --max-width 1000

Outputs a transparent image (PNG by default, WebP with --webp).
Uses alpha blending near edges so text and lines stay sharp.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def white_to_transparent(img_path: str, threshold: int = 240, feather: int = 10,
                         webp: bool = False, max_width: int = 0) -> str:
    """Convert white/near-white pixels to transparent.

    Args:
        img_path: Path to input image.
        threshold: Pixels with R,G,B all above this are considered "white".
        feather: Width of the soft edge (0 = hard cutoff, higher = softer).
        webp: Output WebP instead of PNG (much smaller).
        max_width: Resize if wider than this (0 = no resize).

    Returns:
        Path to the output file.
    """
    im = Image.open(img_path).convert("RGBA")

    if max_width and im.width > max_width:
        im.thumbnail((max_width, max_width), Image.LANCZOS)

    data = np.array(im, dtype=np.float32)

    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

    # Compute "whiteness": min channel value (high = white)
    min_rgb = np.minimum(np.minimum(r, g, ), b)

    if feather > 0:
        # Soft transition: pixels between (threshold - feather) and threshold
        # get partial transparency
        low = float(threshold - feather)
        high = float(threshold)
        # alpha_factor: 1.0 for dark pixels, 0.0 for white pixels
        alpha_factor = np.clip((high - min_rgb) / (high - low), 0.0, 1.0)
    else:
        alpha_factor = np.where(min_rgb >= threshold, 0.0, 1.0)

    # Apply: multiply existing alpha by our factor
    new_alpha = (a * alpha_factor).astype(np.uint8)
    data[:, :, 3] = new_alpha

    out = Image.fromarray(data.astype(np.uint8), "RGBA")

    if webp:
        out_path = Path(img_path).with_suffix(".webp")
        out.save(str(out_path), "WEBP", quality=85, method=6)
    else:
        out_path = Path(img_path).with_suffix(".png")
        out.save(str(out_path), "PNG", optimize=True)

    import os
    size_kb = os.path.getsize(str(out_path)) // 1024
    print(f"  {Path(img_path).name} -> {out_path.name}  ({out.size[0]}x{out.size[1]}, {size_kb}KB)")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Remove white background from images")
    parser.add_argument("images", nargs="+", help="Image file(s) to process")
    parser.add_argument("--threshold", type=int, default=240,
                        help="RGB threshold for 'white' (default: 240)")
    parser.add_argument("--feather", type=int, default=10,
                        help="Edge softness in RGB units (default: 10, 0=hard)")
    parser.add_argument("--webp", action="store_true",
                        help="Output WebP instead of PNG (much smaller)")
    parser.add_argument("--max-width", type=int, default=0,
                        help="Resize if wider than this (default: no resize)")
    args = parser.parse_args()

    for img_path in args.images:
        if not Path(img_path).exists():
            print(f"  [skip] {img_path} not found")
            continue
        white_to_transparent(img_path, threshold=args.threshold,
                             feather=args.feather, webp=args.webp,
                             max_width=args.max_width)


if __name__ == "__main__":
    main()
