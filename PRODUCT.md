# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Astro, producing a fully static build deployed to GitHub Pages at `bartusiak.ai` via `.github/workflows/deploy.yml`. Hard constraints: static output only, no server runtime, GitHub Pages hosting.

The Jekyll + Hydeout site it replaced has been removed. Post URLs it published (`/YYYY/MM/DD/slug.html`) still resolve, as literal redirect stubs in `public/`.

## Users

Two primary audiences, both confirmed, arriving through different doors:

1. **Hiring clients and technical recruiters.** They arrive from Toptal, LinkedIn, or a referral, already holding a shortlist. Their job is to decide quickly whether Roman is credible and worth a conversation. They scan for seniority, domain depth (agentic AI, NLP, semantic search, graph ML, distributed systems, platform engineering), and evidence that the claims are real.
2. **ML and infrastructure practitioners.** They arrive from search or X on a specific technical problem — DGX Spark memory limits, Unsloth builds, NVIDIA PyTorch container performance. Their job is to solve that problem. They did not come for Roman; they came for the answer, and only afterwards may wonder who wrote it.

The two audiences are not served by the same page, but the second is the largest source of the first's respect. A practitioner who finds a post genuinely useful is the highest-quality path into a hire conversation.

## Product Purpose

The personal site of Roman Bartusiak — AI researcher, staff machine learning engineer and AI platform architect. It exists to do three jobs at once:

- **Establish authority.** Anyone who looks Roman up should come away convinced of both depth and range: published NLP research, production ML platform work, distributed systems architecture, and hands-on hardware/performance engineering.
- **Open hire conversations.** Qualified visitors should have an obvious, low-friction path to contact or to the Toptal profile.
- **Serve as a living archive.** A durable, self-owned record of writing, publications, open-source software, and teaching that outlives any individual platform.

Success is a qualified inquiry, or a practitioner who bookmarks the site after a post solved their problem. The site is explicitly required to *stand out* — a generic, off-the-shelf personal-blog template is a failure state regardless of how correct the content is.

## Positioning

The combination is the position, and it is not one a neighboring personal site can truthfully copy: peer-reviewed AI research (Neurocomputing, NeurIPS/LEPISZCZE, IEEE/WIC/ACM) **and** production engineering at scale (Nokia Networks software architect, StashAway staff engineer on investment systems, staff ML engineer and AI platform architect at Panasonic Global Applied AI) **and** founder execution (CTO and co-founder of IamBot, a Techstars company) **and** current, first-hand hardware benchmarking that produces numbers nobody else has published yet.

Roman's own framing, from his CV: he turns research and prototypes into reliable production systems, with explicit trade-offs across quality, latency, throughput, cost and operational risk. Fourteen-plus years, staff-level architecture ownership, nine publications and eight research/R&D projects.

The recent DGX Spark writing is the sharpest expression of this: measured results from a machine most practitioners do not have access to, written by someone who can explain both the CUDA-level cause and the systems-level consequence. Most people writing about LLM tooling have not published research; most people publishing research are not benchmarking their own devbox.

## Operating Context

- Content is authored as Markdown with front matter and committed to git; publishing is a push, not a CMS action.
- Posts are technical, code-and-benchmark-heavy: shell transcripts, Python snippets, environment variables, timing tables, and TFLOPS comparisons. Code blocks and tabular data are primary content, not decoration — they must be first-class in the design, not an afterthought.
- Posts carry a `## TL/DR` section by convention; readers scanning for an answer rely on it.
- Traffic to posts is largely inbound and single-purpose (a search result, an X link). Most sessions start on a post, not the home page.
- Posts reference and continue each other, and cite external discussion (e.g. an exchange with Sebastian Raschka on X).
- The site is one of several `.ml`/`.ai` domains Roman has run (`sparkling.ml`, `lsdp.ml`, `probabilistyczne.ml`).

## Capabilities and Constraints

**Must have (non-negotiable):**
- The Toptal "Top 3% Talent / Vetted by Toptal / Hire me" badge, present and functional, linking to the Toptal résumé. This is the single explicitly named must-keep element.
- Fully static output hosted on GitHub Pages.

**Content handling:** All existing facts — employment history, publications, education, certifications, projects, software list, didactics, and every post — are preserved as true and must remain present and accurate. Structure, sequencing, grouping, and presentation copy around those facts are open to redesign. Do not invent, inflate, or drop a factual claim.

**Surfaces:** home floorplan, post/volume pages, all-volumes index, About, Software, Didactics, tags index and tag pages, client-side search, 404, RSS feed, sitemap, PGP public key (`/pubring.asc`), LinkedIn and GitHub links.

**Retired in the rebuild:** Disqus comments and Google Analytics. Neither was named a must-keep; the site now ships no third-party tracking and no comment system. Re-adding either is a live decision, not an oversight.

**Known defects in current content, to fix during migration (not to reproduce):**
All four defects recorded during init were fixed during the Astro rebuild: the leaked assistant artifact in `about.md`, the `bartusiak.ml` / `bartusiak.ai` mismatch, the empty site description, and three post-title typos. One remains open: didactics links point at `.ml` domains (`lsdp.ml`, `probabilistyczne.ml`) whose liveness is unverified, and the page now says so in a note.

**Resolved since the rebuild (user-confirmed, 2026-08-31):**
- **Availability:** the site no longer claims active availability. The home band reads "Open to interesting work". Revisit if that changes.
- **Contact:** email is published as `roman@bartusiak.ai`, assembled at runtime from base64 so naive harvesters miss it. The phone number on the CV is deliberately **not** published. The personal gmail address is **not** published.
- **CV:** `public/roman-bartusiak-cv.pdf` is linked from the About page. It is `CV___EN (7).pdf` with the phone number stripped from the header — the contact line was re-centred and the following line's `Td` adjusted so the rest of the document is byte-for-byte in place. Regenerate it with `scripts/strip-cv-phone.py` whenever a new revision is dropped in, and re-check that the phone is gone.
- **Toptal:** the CV omits the freelance role entirely; the user asked to keep it on the site alongside the CV record. The badge stays — it is the one hard must-have.
- **Doctorate:** stated as the CV states it — doctoral studies in Computer Science, 2015–2022, research area scalable massive networked data processing, supervisor Prof. Przemysław Kazienko. The previous site's "postponed" framing is retired.

## Brand Commitments

- **Name:** Roman Bartusiak. Domain and identity: `bartusiak.ai`.
- **Assets on hand:** portrait photo (`public/assets/img/me.jpg`), Toptal badge (`src/components/ToptalBadge.astro`, self-contained SVG + CSS, carried over unchanged), favicon, project images (`iambot.png`, `performance.png`, `roosters.png`), PGP public key (`public/pubring.asc`).
- **Voice:** first-person, direct, practical, unpolished in a way that reads as genuine engineering rather than marketing. Posts open with a TL/DR and get to the command line quickly. Preserve this register; do not smooth it into corporate copy.
- **Third-party mark:** the Toptal badge is Toptal's brand asset and its internal styling and geometry are theirs — it may be placed and scaled, not redrawn.

## Evidence on Hand

Everything usable is in the repository or in the CV the user supplied on 2026-08-31.

Source of record for the CV facts: `CV___EN (4).pdf`, read 2026-08-31. Newer revisions exist in the user's Downloads (`CV___EN (5)*`); the user named v4 explicitly.

- **Publications:** 9, with venues, co-authors, years, and Google Scholar links (Neurocomputing 2019, NeurIPS 2022, IEEE/WIC/ACM 2016, ENIC 2015, ACIIDS 2016/2017, CDVE 2015, Encyclopedia of Social Network Analysis and Mining 2017/2018). The tenth Scholar entry, "SparklingGraph Documentation", is documentation rather than a publication and is recorded under software.
- **Grants:** 8 research/R&D projects. R&D leader and author for synthetic data generation, semantic search and dialogue-system projects; participant in RENOIR (H2020-MSCA-RISE) and ENGINE (FP7).
- **Employment:** Independent research (looking for the next thing — vector indexes, JEPA-like models, similarity learning, 09.2026–present), Panasonic Global Applied AI (Staff ML Engineer / AI Platform Architect, 06.2024–08.2026), Causality · FN · Roosters (Founder/Researcher, 10.2020–present, run alongside full-time work and labelled as such), StashAway (Staff Engineer, 10.2020–06.2024), Toptal (freelance, 03.2018–present), IamBot, a Techstars company (CTO, co-founder, 05.2017–09.2020), Wrocław University of Science and Technology (Research Assistant, 08.2015–09.2016), Nokia Networks (Software Engineer/Architect, 07.2013–07.2015), Chop-Chop.org (Web Developer, 05.2012–07.2013).
- **Named production work:** KanpAI, Panasonic's enterprise agentic AI platform — Agent SDK, Agent CLI, multi-tenant deployment and orchestration, LLM-as-a-judge regression workflows; scaled to 20+ agents across 10+ domains, 20+ model families, 30+ training datasets, 70+ concurrent enterprise users in live sessions. StashAway: investment pipelines and portfolio rebalancing each made 2× faster.
- **Languages:** Polish native, English full professional, German elementary.
- **Open source:** SparklingGraph, polish-sentiment, vibe.js, OpticalFlow.js, HMM.js, Data Projector.
- **Teaching:** four university courses lectured at WUST.
- **Measured benchmark data:** real TFLOPS and memory figures from the DGX Spark posts, including the ~50% throughput delta between native PyTorch and the official NVIDIA container. This is original first-party data.
- **Third-party validation:** Toptal top-3% network membership; BSc and MSc both 5.5/A+; Akka Reactive Architecture certifications (2020); IamBot was a Techstars company.

**Do not fabricate:** there are no testimonials, client logos, case studies, press mentions, star counts, traffic numbers, or pricing. None exist and none may be invented or implied.

## Product Principles

1. **The measurement is the credential.** Original, reproducible numbers earn more trust from both audiences than any adjective about seniority. Lead with what was measured, then who measured it.
2. **Every post is a landing page.** Most visitors arrive mid-site on a single post and have never heard of Roman. A post must answer its question first, then make identity discoverable without interrupting the answer.
3. **Research and production are one claim, not two.** The site must never read as an academic CV with a blog bolted on, or as a dev blog with publications appended. The intersection is the differentiator and the structure should make it unmissable.
4. **Technical content is the design's primary material.** Code blocks, terminal output, benchmark tables, and comparison figures carry the substance. They deserve the strongest typographic and structural treatment on the site, not the weakest.
5. **Standing out is a requirement, not a preference.** Recognizable-template design actively undercuts the positioning of someone who builds novel systems. Distinctiveness here is functional.
6. **Nothing is claimed that is not documented.** Every credential on the site traces to a real publication, employer, repository, or measurement.

## Accessibility & Inclusion

No product-specific requirement was established beyond standard practice. Note that the audience reads long technical content with dense code and data tables, often on mobile from a search result — legible code at small widths, horizontal-scroll containment for wide output, and reliable contrast on syntax highlighting are functional needs for this audience, not just compliance items.
