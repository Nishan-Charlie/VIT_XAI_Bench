"""Copy web-optimised figure assets into the website directory.

The figures in ``figures/bench/`` are print-resolution RGBA PNGs — the
qualitative grid alone is 5.8 MB, which is far too heavy to put on a page. This
script writes downscaled, flattened copies to ``website/public/figures/`` so the
site stays fast, and so the GitHub Pages deploy (which uploads only ``website/``)
has everything it needs.

Run it after regenerating any figure it lists, then commit the output:

    python scripts/export_web_assets.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SRC_DIR = REPO_ROOT / "figures" / "bench"
OUT_DIR = REPO_ROOT / "website" / "public" / "figures"

#: (source filename, max width in px). Height follows the aspect ratio.
#: Widths are chosen so the figure is still readable at the page's content
#: width on a HiDPI display, without shipping print resolution.
ASSETS: list[tuple[str, int]] = [
    ("qualitative_grid_cat.png", 2000),
    ("cam_upsampling_artifact.png", 1600),
]

#: PNG quantisation palette size. The figures are heatmaps over a photo, so a
#: 256-colour palette is visually lossless at display size and cuts file size
#: by roughly an order of magnitude.
PALETTE_COLORS = 256


def export(src: Path, dst: Path, max_width: int) -> tuple[int, int]:
    """Downscale, flatten onto white, and quantise. Returns (src, dst) bytes."""
    from PIL import Image

    im = Image.open(src)
    src_bytes = src.stat().st_size

    # Flatten alpha onto white: matplotlib writes RGBA, and a transparent
    # background renders as black in some browsers.
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        flat = Image.new("RGB", im.size, (255, 255, 255))
        flat.paste(im, mask=im.split()[-1])
        im = flat
    else:
        im = im.convert("RGB")

    if im.width > max_width:
        height = round(im.height * max_width / im.width)
        im = im.resize((max_width, height), Image.LANCZOS)

    im = im.quantize(colors=PALETTE_COLORS, method=Image.MEDIANCUT)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, format="PNG", optimize=True)
    return src_bytes, dst.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=SRC_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    missing = [name for name, _ in ASSETS if not (args.src / name).exists()]
    if missing:
        print("ERROR: missing source figures:")
        for name in missing:
            print(f"  {args.src / name}")
        return 1

    total_src = total_dst = 0
    for name, max_width in ASSETS:
        src_bytes, dst_bytes = export(args.src / name, args.out / name, max_width)
        total_src += src_bytes
        total_dst += dst_bytes
        print(f"  {name:32s} {src_bytes/1e6:5.2f} MB -> {dst_bytes/1e6:5.2f} MB")

    print(f"\n{len(ASSETS)} asset(s): {total_src/1e6:.2f} MB -> {total_dst/1e6:.2f} MB "
          f"({100 * (1 - total_dst / total_src):.0f}% smaller)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
