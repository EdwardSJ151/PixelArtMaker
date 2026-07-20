# NES & SNES Pixel Art Specifications

## NES

### Resolution
- Screen: **256×240** pixels
- Tiles: **8×8** base unit (everything is composed of 8×8 tiles)

### Colors
- Master palette: **54 colors** total (hardware-defined, fixed RGB values)
- Colors on screen at once: **25** (background) + **16** (sprites) = theoretically up to 41 visible simultaneously
- Per palette: **4 entries**, but index 0 is always transparent → **3 usable colors** per palette
- Background: 4 palettes × 3 colors = 12 usable BG colors + 1 shared backdrop color
- Sprites: 4 palettes × 3 colors = 12 usable sprite colors

### Sprite Rules
- Hardware sprite sizes: **8×8** or **8×16** (set globally per frame, not per sprite)
- Larger sprites (16×16, 32×32) are **software composites** — multiple OAM entries stitched together
- Each 8×8 or 8×16 OAM entry uses **1 of 4 sprite palettes** → 3 usable colors per tile
- Color budget by composite size:
  - 8×8 / 8×16: **3 usable**
  - 16×16: **4 usable** (2 tiles can use different palettes)
  - 16×32 / 32×32+: **5 usable**

### Sprite Limits
- OAM (Object Attribute Memory): **64 sprites** max on screen
- **8 sprites per scanline** — sprites beyond 8 on the same horizontal line flicker or disappear
- Sprite flickering is the classic NES trick for working around this limit

### Background (Tiles)
- Background is a tilemap of 8×8 tiles
- **Attribute bytes** assign one palette per **16×16 pixel block** (2×2 tiles share a palette)
- Each 8×8 tile: 3 usable colors, constrained to its attribute block's palette

### Pattern Tables
- 2 pattern tables, each holding **256 tiles** (8×8 each) = 512 tiles total
- One table for BG tiles, one for sprites (configurable)
- **2bpp** (2 bits per pixel) — 4 values per pixel (0=transparent, 1-3=color index)

---

## SNES

### Resolution
- Screen: **256×224** (NTSC) or **256×239** (PAL) — standard mode
- Hi-res mode: **512×448**, but rarely used for sprites

### Colors
- Master palette: **32,768 colors** (15-bit RGB, 5 bits per channel — no fixed hardware colors)
- Colors on screen at once: **256** from the CGRAM (Color Graphics RAM)
- CGRAM: 256 entries, split into **8 palettes of 16 entries each** for sprites (4bpp mode)
- Index 0 of each subpalette is transparent → **15 usable colors per sprite subpalette**

### Sprite Rules
- Hardware sprite sizes (two sizes selectable per frame):
  - Small/large pairs: 8×8 & 16×16, 8×8 & 32×32, 8×8 & 64×64, 16×16 & 32×32, 16×16 & 64×64, 32×32 & 64×64
  - Native **non-square**: **16×32** and **32×64** are also valid hardware sizes
- Color depth: **4bpp** → 16 colors per palette → 15 usable + 1 transparent
- Each sprite references one of 8 sprite subpalettes

### Sprite Limits
- OAM: **128 sprites** max on screen
- **32 sprites per scanline** — exceeding causes the oldest sprites to disappear (no flicker, just drop)
- Each sprite can be size-small or size-large (set per-sprite via OAM bit)

### Background Layers
- Up to **4 background layers** (BG1–BG4), each with independent scroll
- BG layers can use 2bpp (4 colors), 4bpp (16 colors), or 8bpp (256 colors) depending on mode
- Mode 7: single affine-transformed layer — used for pseudo-3D effects (F-Zero, Super Mario Kart)

### Tiles
- Background tiles: **8×8** only
- Tile data stored in VRAM — up to **64KB** of VRAM total (tiles + tilemap + OAM)
- **16bpp** color entries in CGRAM (1 bit priority + 5R + 5G + 5B)

---

## Quick Comparison

| | NES | SNES |
|---|---|---|
| Screen | 256×240 | 256×224 |
| Master palette | 54 fixed colors | 32,768 (free 15-bit RGB) |
| On-screen colors | ~25 | 256 |
| Sprite colors | 3 usable / palette | 15 usable / subpalette |
| Sprite palettes | 4 | 8 |
| Max sprites | 64 | 128 |
| Sprites/scanline | 8 | 32 |
| Tile size | 8×8 | 8×8 (BG), 8×8–64×64 (sprites) |
| Color depth | 2bpp | 4bpp (sprites), up to 8bpp (BG) |
