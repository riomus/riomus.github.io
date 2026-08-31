# Copilot instructions for riomus.github.io

Personal site of Roman Bartusiak at **bartusiak.ai**. Astro, static output,
deployed to GitHub Pages by `.github/workflows/deploy.yml`. It was a Jekyll
site on the Hydeout theme until the Astro rebuild; ignore any Jekyll
conventions you find referenced elsewhere.

## Architecture

- **Content**: `src/content/posts/*.md`, loaded by a content collection defined
  in `src/content.config.ts`. Filenames are `YYYY-MM-DD-slug.md` and the date
  drives both the URL and the volume number.
- **CV facts**: `src/data/record.ts` — experience, publications, software,
  courses, education, certifications. Every entry is a real fact. Do not add,
  inflate, or invent one.
- **Routing**: `src/pages/[...slug].astro` builds posts at `/YYYY/MM/DD/slug/`.
  Tag pages group by slug so two spellings of a tag cannot collide.
- **Styles**: one stylesheet, `src/styles/global.css`, plus scoped `<style>`
  blocks per component. No CSS framework.
- **Markdown**: Astro 7's Sätteri processor. The `dieDefaults` hast plugin in
  `astro.config.mjs` wraps tables for horizontal scroll and lazy-loads images.

## The visual world

Read the direction contract at the top of `<body>` in `src/layouts/Base.astro`
before touching the design, and `DESIGN.md` for the recorded system. Short
version: the page is a chip die floorplan — gutterless blocks tiling the frame,
separated only by the hairline scribe (`--scribe`), navigated from a bond-pad
ring. Textures (`.tex--sram`, `--logic`, `--io`, `--analog`, `--pads`) are
produced live from gradients and magnify on hover via the `--t` tile size.

Two rules that are easy to break:
- `.block > *:not(.tex)` — the texture layer must keep its own absolute
  positioning. Adding a blanket `.block > *` position rule kills every texture.
- Textures are masked away from the reading area. If you raise their opacity,
  re-check text contrast.

## Constraints

- Static output only, GitHub Pages, no server runtime.
- The Toptal badge (`src/components/ToptalBadge.astro`) is Toptal's brand
  asset. Place and scale it; never redraw it. Its single-letter classes are
  scoped under `#r` deliberately.
- Legacy `/YYYY/MM/DD/slug.html` URLs ship as stubs in `public/`. Keep them.
- Polish characters must render: both `latin` and `latin-ext` font subsets are
  self-hosted in `public/fonts/`.

## Workflows

```bash
npm install && npm run dev     # http://localhost:4321
npm run build                  # -> dist/
```
