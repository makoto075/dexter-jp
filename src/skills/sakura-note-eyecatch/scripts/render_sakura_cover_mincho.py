#!/usr/bin/env python3
"""Sakura photo cover adaptation of render_luna_cover_v02.py.

The reference script remains unchanged. This derived copy targets 1280 x 670
and retains its JSON-driven crop, horizontal mask, Pillow text composition,
and PNG/JPG/SVG outputs, adding validation, site-specific layout, WebP, and QA
previews. Requires Pillow >= 10. Paths are relative to the JSON file.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont, ImageOps


SIZE = (1280, 670)
LEFT = 136
HEADLINE_RIGHT = 820
REQUIRED = {
    "source_image", "output_basename", "font_bold", "font_regular",
    "font_family", "brand", "label", "headline", "product", "subtitle",
    "label_color", "product_color", "subtitle_color", "mask_rgb",
    "mask_opacity_stops", "crop_center",
}


def resolve(base: Path, name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else base / path


def color(value: str) -> tuple[int, int, int]:
    if not isinstance(value, str) or len(value) != 7 or not value.startswith("#"):
        raise ValueError(f"Invalid #RRGGBB color: {value!r}")
    try:
        return tuple(bytes.fromhex(value[1:]))  # type: ignore[return-value]
    except ValueError as exc:
        raise ValueError(f"Invalid #RRGGBB color: {value!r}") from exc


def validate(data: dict) -> None:
    missing = REQUIRED - data.keys()
    if missing:
        raise ValueError(f"Missing required JSON keys: {sorted(missing)}")
    if not isinstance(data["headline"], list) or len(data["headline"]) != 2 or not all(
        isinstance(line, str) and line.strip() for line in data["headline"]
    ):
        raise ValueError("headline must contain exactly two nonempty strings")
    if not isinstance(data["mask_rgb"], list) or len(data["mask_rgb"]) != 3 or not all(
        type(v) is int and 0 <= v <= 255 for v in data["mask_rgb"]
    ):
        raise ValueError("mask_rgb must contain three integers from 0 to 255")
    stops = data["mask_opacity_stops"]
    if not isinstance(stops, list) or len(stops) < 2 or any(
        not isinstance(p, list) or len(p) != 2 or
        not all(isinstance(v, (float, int)) and math.isfinite(v) and 0 <= v <= 1 for v in p)
        for p in stops
    ) or any(stops[i][0] >= stops[i + 1][0] for i in range(len(stops) - 1)):
        raise ValueError("mask_opacity_stops must be sorted x,opacity pairs in [0,1]")
    if stops[0][0] != 0 or stops[-1][0] != 1:
        raise ValueError("mask_opacity_stops must cover x=0 through x=1")
    center = data["crop_center"]
    if not isinstance(center, list) or len(center) != 2 or not all(
        isinstance(v, (float, int)) and math.isfinite(v) and 0 <= v <= 1 for v in center
    ):
        raise ValueError("crop_center must be two numbers in [0,1]")
    for key in ("label_color", "product_color", "subtitle_color"):
        color(data[key])
    for key in ("brand", "label", "product", "subtitle", "font_family"):
        if not isinstance(data[key], str):
            raise ValueError(f"{key} must be a string")


def crop_cover(image: Image.Image, center: list[float]) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    # Keep the supplied script's ImageOps.fit behavior after EXIF correction.
    return ImageOps.fit(image, SIZE, method=Image.Resampling.LANCZOS,
                        centering=tuple(center))


def opacity_at(x: float, stops: list[list[float]]) -> float:
    for left, right in zip(stops, stops[1:]):
        if x <= right[0]:
            t = (x - left[0]) / (right[0] - left[0])
            return left[1] + (right[1] - left[1]) * t
    return stops[-1][1]


def load_font(path: Path, size: int, weight: int) -> ImageFont.FreeTypeFont:
    if not path.is_file():
        raise FileNotFoundError(f"Font not found: {path}")
    font = ImageFont.truetype(str(path), size)
    try:
        axes = font.get_variation_axes()
        if len(axes) == 1 and b"Weight" in axes[0]["name"]:
            font.set_variation_by_axes([weight])
    except (AttributeError, OSError, ValueError):
        pass  # Static font files have no variation axis.
    return font


def luminance(rgb: tuple[int, int, int]) -> float:
    channels = [v / 255 for v in rgb]
    linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(foreground: tuple[int, int, int], background: tuple[int, int, int]) -> float:
    a, b = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def draw_and_check(base: Image.Image, data: dict, bold: ImageFont.FreeTypeFont,
                   regular: ImageFont.FreeTypeFont, product_font: ImageFont.FreeTypeFont,
                   subtitle_font: ImageFont.FreeTypeFont) -> dict[str, float]:
    items = [
        ("label", data["label"], (LEFT, 102), regular, color(data["label_color"])),
        ("headline1", data["headline"][0], (LEFT, 206), bold, (255, 255, 255)),
        ("headline2", data["headline"][1], (LEFT, 310), bold, (255, 255, 255)),
        ("product", data["product"], (LEFT, 454), product_font, color(data["product_color"])),
        ("subtitle", data["subtitle"], (LEFT, 506), subtitle_font, color(data["subtitle_color"])),
        ("brand", data["brand"], (LEFT, 582), regular, (232, 201, 156)),
    ]
    draw = ImageDraw.Draw(base)
    boxes: dict[str, tuple[int, int, int, int]] = {}
    for name, content, xy, font, ink in items:
        if not content:
            continue
        box = draw.textbbox(xy, content, font=font, anchor="lt")
        if box[0] < 128 or box[1] < 30 or box[2] > SIZE[0] - 58 or box[3] > SIZE[1] - 35:
            raise ValueError(f"{name} is outside the safe area: {box}")
        if name.startswith("headline") and box[2] > HEADLINE_RIGHT:
            raise ValueError(f"{name} exceeds x={HEADLINE_RIGHT}: {box}")
        for other, old in boxes.items():
            if box[0] < old[2] and old[0] < box[2] and box[1] < old[3] and old[1] < box[3]:
                raise ValueError(f"Text overlap: {name} and {other}")
        boxes[name] = box
    # Contrast is measured against the masked image at solid glyph pixels.
    ratios = {}
    for name, content, xy, font, ink in items:
        if not content:
            continue
        mask = Image.new("L", SIZE, 0)
        ImageDraw.Draw(mask).text(xy, content, font=font, fill=255, anchor="lt")
        box = boxes[name]
        low = float("inf")
        glyph = mask.load()
        pixels = base.load()
        for y in range(box[1], box[3]):
            for x in range(box[0], box[2]):
                if glyph[x, y] >= 250:
                    low = min(low, contrast(ink, pixels[x, y]))
        if not math.isfinite(low) or low < 4.5:
            raise ValueError(f"Insufficient text contrast for {name}: {low:.2f}:1")
        ratios[name] = round(low, 2)
        draw.text(xy, content, font=font, fill=ink, anchor="lt")
    return ratios


def write_svg(background: Image.Image, data: dict, destination: Path) -> None:
    """Reference SVG. Fonts remain external, so raster files are canonical."""
    raw = io.BytesIO()
    background.save(raw, format="PNG", optimize=True)
    encoded = base64.b64encode(raw.getvalue()).decode("ascii")
    stops = "".join(
        f'<stop offset="{x * 100:g}%" stop-color="#{data["mask_rgb"][0]:02x}'
        f'{data["mask_rgb"][1]:02x}{data["mask_rgb"][2]:02x}" '
        f'stop-opacity="{opacity:g}"/>'
        for x, opacity in data["mask_opacity_stops"]
    )
    family = escape(data["font_family"], {'"': '&quot;'})
    text_items = [
        ("label", data["label"], 102, 26, data["label_color"], 400),
        ("headline1", data["headline"][0], 206, 82, "#FFFFFF", 400),
        ("headline2", data["headline"][1], 310, 82, "#FFFFFF", 400),
        ("product", data["product"], 454, 32, data["product_color"], 400),
        ("subtitle", data["subtitle"], 506, 25, data["subtitle_color"], 400),
        ("brand", data["brand"], 582, 26, "#E8C99C", 400),
    ]
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="670" '
        'viewBox="0 0 1280 670"><defs><linearGradient id="mask" x1="0" '
        f'y1="0" x2="1" y2="0">{stops}</linearGradient></defs>'
        f'<image width="1280" height="670" href="data:image/png;base64,{encoded}"/>'
        '<rect width="1280" height="670" fill="url(#mask)"/>'
    )
    for _, content, y, size, ink, weight in text_items:
        if content:
            svg += (
                f'<text x="{LEFT}" y="{y}" dominant-baseline="text-before-edge" '
                f'font-family="{family}" font-size="{size}" font-weight="{weight}" '
                f'fill="{ink}">{escape(content)}</text>'
            )
    svg += "</svg>"
    destination.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    validate(data)
    root = config_path.parent
    source = resolve(root, data["source_image"])
    if not source.is_file():
        raise FileNotFoundError(f"Background not found: {source}")
    with Image.open(source) as image:
        if ImageOps.exif_transpose(image).width <= ImageOps.exif_transpose(image).height:
            raise ValueError("Background must be landscape after EXIF orientation is applied")
        background = crop_cover(image, data["crop_center"])
    mask_values = [round(255 * opacity_at(x / (SIZE[0] - 1), data["mask_opacity_stops"]))
                   for x in range(SIZE[0])]
    alpha = Image.new("L", (SIZE[0], 1))
    alpha.putdata(mask_values)
    alpha = alpha.resize(SIZE)
    overlay = Image.new("RGB", SIZE, tuple(data["mask_rgb"]))
    masked = Image.composite(overlay, background, alpha)
    bold = load_font(resolve(root, data["font_bold"]), 82, 400)
    regular_path = resolve(root, data["font_regular"])
    regular = load_font(regular_path, 26, 400)
    # Text sizes are explicit so changing the JSON alone cannot silently change layout.
    label_font = regular
    product_font = load_font(regular_path, 32, 400)
    subtitle_font = load_font(regular_path, 25, 400)
    ratios = draw_and_check(masked, data, bold, label_font, product_font, subtitle_font)
    output = resolve(root, data["output_basename"])
    output.parent.mkdir(parents=True, exist_ok=True)
    expected = [output.with_suffix(ext) for ext in (".png", ".jpg", ".webp", ".svg")]
    expected += [output.with_name(output.name + tail).with_suffix(".png") for tail in ("-320x168", "-160x84", "-1p618-preview")]
    if any(path.exists() for path in expected):
        raise FileExistsError("Output exists; choose a new output_basename to preserve it")
    masked.save(output.with_suffix(".png"), optimize=True)
    masked.save(output.with_suffix(".jpg"), quality=90, subsampling=0, optimize=True)
    masked.save(output.with_suffix(".webp"), quality=88, method=6)
    write_svg(background, data, output.with_suffix(".svg"))
    masked.resize((320, 168), Image.Resampling.LANCZOS).save(
        output.with_name(output.name + "-320x168").with_suffix(".png"))
    masked.resize((160, 84), Image.Resampling.LANCZOS).save(
        output.with_name(output.name + "-160x84").with_suffix(".png"))
    # A narrower reference frame: 670px height at 1.618:1 loses about 98px per side.
    crop_width = round(SIZE[1] * 1.618)
    left = (SIZE[0] - crop_width) // 2
    narrow = masked.crop((left, 0, left + crop_width, SIZE[1]))
    narrow.resize((320, round(320 / 1.618)), Image.Resampling.LANCZOS).save(
        output.with_name(output.name + "-1p618-preview").with_suffix(".png"))
    print(json.dumps({"output": str(output), "size": SIZE, "contrast_min": ratios,
                      "narrow_crop_loss_each_side_px": left}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
