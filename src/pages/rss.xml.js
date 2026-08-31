import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import { postHref } from '../utils';

export async function GET(context) {
  const posts = (await getCollection('posts')).sort(
    (a, b) => b.data.date.valueOf() - a.data.date.valueOf()
  );
  return rss({
    title: 'Roman Bartusiak',
    description:
      'Measured work on DGX Spark, PyTorch, distributed systems and NLP — by Roman Bartusiak.',
    site: context.site,
    items: posts.map((p) => ({
      title: p.data.title,
      description: p.data.description,
      pubDate: p.data.date,
      link: postHref(p),
      categories: p.data.tags,
    })),
    customData: '<language>en</language>',
  });
}
