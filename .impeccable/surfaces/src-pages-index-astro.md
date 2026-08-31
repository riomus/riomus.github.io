---
version: 1
slug: "src-pages-index-astro"
primary_target: "src/pages/index.astro"
related_targets: []
---

## Scope

The home page (`src/pages/index.astro`) — the die floorplan. Visitor mode: **Persuade**.

## Audience and job

Two arrivals, one page. A hiring client or recruiter landing from Toptal, LinkedIn or a referral, deciding in seconds whether Roman is worth a conversation. An ML/infra practitioner who came from a search result on a DGX Spark post and is now looking around. The page has to satisfy the second while converting the first.

## Action

Primary: the gold Hire pad (ring, and the availability block's Toptal CTA). Secondary: open a volume. The page is judged on whether a first-time visitor knows what this is, why it matters, and what to do without scrolling.

## Proof and content

Everything is real and already in the repo: four measured runs with their rig/stack/finding, ten peer-reviewed publications, six open-source libraries, four university courses, the employment record, the Toptal top-3% badge. No testimonials, client logos, traffic numbers or press exist — none may be invented or implied.

## Constraints

Static Astro on GitHub Pages. The Toptal badge is a must-have and is Toptal's own asset: placed and scaled, never redrawn. Legacy `/YYYY/MM/DD/slug.html` post URLs must keep resolving.

## Chosen direction

**Die Shot** — the site is the silicon it measures. Chosen by the user off the IMPECCABLE'S PICK card over the roll's assignment (seed `8a898c39`, mode persuade, code-led). The full contract lives at the top of `<body>` in `src/layouts/Base.astro`.

## Composition

Gutterless blocks tiling the viewport, split only by the hairline scribe, under a bond-pad ring. Block area encodes importance, so the newest measurement is physically the largest region on the die. Reading order: identity, newest volume, the earlier run, then publications / open source / teaching, closing on the gold availability block.

## Memorable moment

Magnification. Hovering a block racks the stage in — the produced texture's tile grows, exposure rises, the title widens on Archivo's width axis, and a `×N` readout appears in the corner. Under it, one ambient 54s illumination sweep, which is what interference colour actually does when a die is tilted under a lamp.

## Unresolved

- Whether a direct contact route (mailto or form) should sit beside the Toptal link. Currently Toptal and LinkedIn are the only paths.
- The availability block states availability outright; if that stops being true it needs editing, and nothing on the page derives it automatically.
- Home shows the lead plus three volumes; past four posts it links to `/volumes/`. That threshold has never been exercised with real overflow.
