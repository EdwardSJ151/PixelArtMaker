"""Color palette management — extraction from image and system presets."""

from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image


# PICO-8: 16 fixed colors
PICO8_COLORS: dict[str, str] = {
    "black":       "#000000",
    "dark_blue":   "#1D2B53",
    "dark_purple": "#7E2553",
    "dark_green":  "#008751",
    "brown":       "#AB5236",
    "dark_gray":   "#5F574F",
    "light_gray":  "#C2C3C7",
    "white":       "#FFF1E8",
    "red":         "#FF004D",
    "orange":      "#FFA300",
    "yellow":      "#FFEC27",
    "green":       "#00E436",
    "blue":        "#29ADFF",
    "lavender":    "#83769C",
    "pink":        "#FF77A8",
    "peach":       "#FFCCAA",
}

# CGA: 16 fixed colors (mode 3, high intensity)
CGA_COLORS: dict[str, str] = {
    "black":          "#000000",
    "dark_blue":      "#0000AA",
    "dark_green":     "#00AA00",
    "dark_cyan":      "#00AAAA",
    "dark_red":       "#AA0000",
    "dark_magenta":   "#AA00AA",
    "dark_yellow":    "#AA5500",
    "light_gray":     "#AAAAAA",
    "dark_gray":      "#555555",
    "blue":           "#5555FF",
    "green":          "#55FF55",
    "cyan":           "#55FFFF",
    "red":            "#FF5555",
    "magenta":        "#FF55FF",
    "yellow":         "#FFFF55",
    "white":          "#FFFFFF",
}

SYSTEM_PALETTES: dict[str, dict[str, str]] = {
    "pico8": PICO8_COLORS,
    "cga":   CGA_COLORS,
}


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


def best_n_from_palette(
    pixels_rgb: np.ndarray,
    master: "Palette",
    n: int,
    min_distance: float = 30.0,
) -> "Palette":
    """Return a new Palette with the n most-frequent master colors in the image.

    Each pixel is snapped to the nearest master palette entry (L2 RGB distance),
    then candidates are ranked by frequency. Colors too close to an already-selected
    color (L2 < min_distance) are skipped in favor of more distinct ones. If not
    enough distinct colors exist to fill n slots, near-duplicates are admitted as
    a fallback so the count is always honored.
    """
    pixels = pixels_rgb.reshape(-1, 3).astype(np.float32)
    dists = np.sum((master._rgb[None, :, :] - pixels[:, None, :]) ** 2, axis=2)
    assignments = np.argmin(dists, axis=1)
    counts = np.bincount(assignments, minlength=len(master.names))

    # Rank all entries by frequency, most frequent first
    ranked = np.argsort(counts)[::-1]

    selected: list[int] = []
    deferred: list[int] = []  # too-close entries, admitted only if needed

    for idx in ranked:
        if counts[idx] == 0:
            break
        rgb = master._rgb[idx]
        too_close = any(
            float(np.sqrt(np.sum((rgb - master._rgb[s]) ** 2))) < min_distance
            for s in selected
        )
        if too_close:
            deferred.append(idx)
        else:
            selected.append(idx)
        if len(selected) == n:
            break

    # Fill remaining slots from deferred (near-duplicates) if we came up short
    for idx in deferred:
        if len(selected) == n:
            break
        selected.append(idx)

    named = {master.names[i]: master.hex_of(i) for i in selected}
    return Palette(named)


def _auto_name_color(r: int, g: int, b: int, existing_names: set[str]) -> str:
    """Generate a human-readable name from RGB using hue + lightness."""
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    hue_deg = h * 360

    if s < 0.1:
        if l < 0.2:
            base = "black"
        elif l > 0.8:
            base = "white"
        elif l < 0.5:
            base = "dark_gray"
        else:
            base = "light_gray"
    else:
        if hue_deg < 30:
            base = "red"
        elif hue_deg < 60:
            base = "orange"
        elif hue_deg < 90:
            base = "yellow"
        elif hue_deg < 150:
            base = "green"
        elif hue_deg < 195:
            base = "cyan"
        elif hue_deg < 255:
            base = "blue"
        elif hue_deg < 285:
            base = "purple"
        elif hue_deg < 330:
            base = "magenta"
        else:
            base = "red"

        if l < 0.25:
            base = f"dark_{base}"
        elif l > 0.75:
            base = f"light_{base}"
        elif l < 0.45:
            base = f"mid_{base}"

    # Deduplicate
    name = base
    i = 2
    while name in existing_names:
        name = f"{base}_{i}"
        i += 1
    return name


class Palette:
    """A fixed set of named colors used by the pixel grid."""

    def __init__(self, named_colors: dict[str, str]):
        """
        Args:
            named_colors: dict mapping color name → hex string (e.g. {"dark_purple": "#2D1B69"})
        """
        self.named_colors = named_colors  # name → hex
        self._names = list(named_colors.keys())
        self._hex_list = list(named_colors.values())
        self._name_to_idx = {n: i for i, n in enumerate(self._names)}
        self._rgb = np.array([_hex_to_rgb(h) for h in named_colors.values()], dtype=np.float32)

    @property
    def names(self) -> list[str]:
        return self._names

    def index_of(self, name: str) -> int:
        return self._name_to_idx[name]

    def name_of(self, index: int) -> str:
        return self._names[index]

    def hex_of(self, index: int) -> str:
        return self._hex_list[index]

    def nearest_index(self, r: int, g: int, b: int) -> int:
        """Return palette index of the nearest color to (r, g, b)."""
        pixel = np.array([r, g, b], dtype=np.float32)
        dists = np.sum((self._rgb - pixel) ** 2, axis=1)
        return int(np.argmin(dists))

    def format_for_prompt(self) -> str:
        lines = []
        for name, hex_val in self.named_colors.items():
            lines.append(f"  {name:<20} {hex_val}")
        return "\n".join(lines)

    @classmethod
    def from_file(cls, path) -> "Palette":
        """Build a Palette from a plain-text file with one hex color per line."""
        lines = Path(path).read_text().splitlines()
        named: dict[str, str] = {}
        i = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            hex_val = line if line.startswith("#") else f"#{line}"
            named[f"c{i:02d}"] = hex_val
            i += 1
        return cls(named)

    @classmethod
    def from_system(cls, system: str) -> "Palette":
        if system in SYSTEM_PALETTES:
            return cls(SYSTEM_PALETTES[system])
        if system == "nes":
            palette_dir = Path(__file__).parent.parent / "palettes"
            nes_file = palette_dir / "nes_palette.txt"
            if not nes_file.exists():
                nes_file = palette_dir / "nes_wiki_palette.txt"
            return cls.from_file(nes_file)
        available = ", ".join(list(SYSTEM_PALETTES.keys()) + ["nes"])
        raise NotImplementedError(
            f"System palette '{system}' is not implemented. Available: {available}"
        )

    @classmethod
    def extract_from_image(
        cls,
        image: Image.Image,
        n_colors: int = 16,
        kmeans_n_init: int = 10,
        kmeans_seed: int = 42,
        max_pixels: int = 50_000,
    ) -> "Palette":
        """Run k-means on image pixels to extract n_colors representative colors."""
        from sklearn.cluster import KMeans

        img_rgb = image.convert("RGB")
        pixels = np.array(img_rgb).reshape(-1, 3).astype(np.float32)

        if len(pixels) > max_pixels:
            rng = np.random.RandomState(kmeans_seed)
            idx = rng.choice(len(pixels), max_pixels, replace=False)
            pixels = pixels[idx]

        km = KMeans(n_clusters=n_colors, random_state=kmeans_seed, n_init=kmeans_n_init)
        km.fit(pixels)
        centers = km.cluster_centers_.astype(int)

        # Sort by luminance so the palette is predictably ordered
        lums = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in centers]
        order = np.argsort(lums)
        centers = centers[order]

        named: dict[str, str] = {}
        seen_names: set[str] = set()
        for r, g, b in centers:
            name = _auto_name_color(int(r), int(g), int(b), seen_names)
            seen_names.add(name)
            named[name] = _rgb_to_hex(int(r), int(g), int(b))

        return cls(named)
