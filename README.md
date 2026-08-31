# bartusiak.ai

Personal site of Roman Bartusiak — AI researcher and machine learning engineer.
Astro, static output, deployed to GitHub Pages.

## Running it

```bash
npm install
npm run dev      # http://localhost:4321
npm run build    # -> dist/
npm run preview  # serve the built output
```

## Writing a volume

Posts live in `src/content/posts/` as `YYYY-MM-DD-slug.md`. The date in the
filename sets the URL (`/YYYY/MM/DD/slug/`) and the volume number, which is
just the post's position in the run, oldest first.

```yaml
---
title: "What the run was"
date: 2025-11-07
tags: ["DGX Spark", "PyTorch"]
description: "One sentence that has to work as the excerpt and the meta description."
rig: "DGX Spark · GB10"                 # optional
stack: "NGC PyTorch vs. host"           # optional
finding: "+50% TFLOPS · 8192² bf16"     # optional
---
```

`rig`, `stack` and `finding` are the spec strip: the same three slots on every
volume, so two benchmarks can be compared without opening both. Leave them out
for a post that measures nothing and the strip is omitted.

Tags are grouped by slug, so `NVIDIA` and `Nvidia` are one tag. Prefer the
spelling already in use.

## Layout of the repo

| Path | What it holds |
|---|---|
| `src/content/posts/` | The volumes |
| `src/data/record.ts` | CV facts: experience, publications, software, courses |
| `src/styles/global.css` | The visual world (see below) |
| `src/layouts/` | `Base` (pad ring, footer), `Page` (titled pages) |
| `src/components/` | Volume block, scale bar, icons, Toptal badge |
| `public/` | Fonts, images, PGP key, CNAME, legacy URL stubs |
| `PRODUCT.md` | Durable product truth |
| `DESIGN.md` | The visual system, recorded from the built site |

## The design

The site is a die floorplan: gutterless blocks tiling the frame, split by the
hairline scribe, navigated from a bond-pad ring. Block area encodes importance.
Hovering a block racks the stage in — its texture magnifies and its label
widens on Archivo's width axis.

The direction contract sits as an HTML comment at the top of `<body>` in
`src/layouts/Base.astro`. Read it before changing the look.

## URLs

Posts published under Jekyll used `/YYYY/MM/DD/slug.html`. Those exact paths
ship as redirect stubs in `public/`, because a redirect route emits a directory
and a path containing a dot is not safe to serve that way on GitHub Pages.
Don't delete them.

## Fonts

Archivo and Chivo Mono are self-hosted as subsets in `public/fonts/`, built by
`node scripts/subset-fonts.mjs`. Fontsource ships latin and latin-ext as
separate files carrying the whole Unicode block each, which cost 220 KB across
four requests — and because "Wrocław" puts a latin-ext glyph in the hero, the
85 KB latin-ext file sat on the critical path. The subsets are 48 KB in two
files, with the variable axes intact.

The character set in that script is deliberately wider than the glyphs used
today (full Latin-1 plus European accents and typographic marks), so a new post
does not silently lose a character. If you need one outside it, add it there
and re-run.

## Deploying

Pushing to `master` runs `.github/workflows/deploy.yml`, which builds and
publishes `dist/` to GitHub Pages. The repo's Pages source must be set to
**GitHub Actions**, not a branch.
