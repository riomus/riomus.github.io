// @ts-check
import { defineConfig } from 'astro/config';
import { satteri } from '@astrojs/markdown-satteri';
import sitemap from '@astrojs/sitemap';

/** Benchmark tables are wide by nature. Each one gets its own scroll
 * container so the page body never scrolls sideways on a phone. In-post
 * rasters decode async but are not lazy: they are the evidence, and a tall
 * lazy chart paints as an empty rectangle. */
const dieDefaults = {
  name: 'die-defaults',
  element: [
    {
      filter: ['table'],
      visit(node, ctx) {
        ctx.wrapNode(node, {
          type: 'element',
          tagName: 'div',
          properties: { className: ['tablewrap'] },
          children: [],
        });
      },
    },
    {
      filter: ['img'],
      visit(node, ctx) {
        // Astro marks markdown images lazy by default. There is exactly one
        // raster in the posts and it is the evidence for the claim, so it
        // loads eagerly rather than painting as an empty rectangle.
        ctx.setProperty(node, 'loading', 'eager');
        ctx.setProperty(node, 'decoding', 'async');
      },
    },
  ],
};

export default defineConfig({
  site: 'https://bartusiak.ai',
  integrations: [sitemap()],
  // Legacy Jekyll URLs ship as literal files in public/ (see the .html
  // stubs there), because a redirect route emits a directory and a path with
  // a dot in it is not safe to serve that way on GitHub Pages.
  build: { inlineStylesheets: 'always' },
  markdown: {
    processor: satteri({ hastPlugins: [dieDefaults] }),
    shikiConfig: { theme: 'css-variables', wrap: false },
  },
});
