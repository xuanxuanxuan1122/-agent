from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


SRC_DIR = Path(r"D:\pychram\RAG2\output\_ch6_render")
OUT_DIR = Path(r"D:\pychram\RAG2\output\_ch6_render_contact")


def page_number(path: Path) -> int:
    return int(path.stem.split("-")[-1])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pages = sorted(SRC_DIR.glob("page-*.png"), key=page_number)
    per_sheet = 12
    thumb_w = 235
    thumb_h = 333
    label_h = 24
    cols = 3
    for sheet_idx in range(0, len(pages), per_sheet):
        subset = pages[sheet_idx : sheet_idx + per_sheet]
        rows = (len(subset) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for local_idx, path in enumerate(subset):
            image = Image.open(path).convert("RGB")
            image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            col = local_idx % cols
            row = local_idx // cols
            x = col * thumb_w + (thumb_w - image.width) // 2
            y = row * (thumb_h + label_h) + label_h
            draw.text((col * thumb_w + 8, row * (thumb_h + label_h) + 4), path.stem, fill=(0, 0, 0))
            sheet.paste(image, (x, y))
        out = OUT_DIR / f"contact-{sheet_idx // per_sheet + 1}.png"
        sheet.save(out)
        print(out)


if __name__ == "__main__":
    main()
