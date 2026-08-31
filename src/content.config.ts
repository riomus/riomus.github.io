import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const posts = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/posts' }),
  schema: z.object({
    title: z.string(),
    date: z.date(),
    tags: z.array(z.string()).default([]),
    description: z.string(),
    // The batch log: every volume records its bench in the same three slots,
    // so two measurements can be read against each other without opening both.
    rig: z.string().optional(),
    stack: z.string().optional(),
    finding: z.string().optional(),
  }),
});

export const collections = { posts };
