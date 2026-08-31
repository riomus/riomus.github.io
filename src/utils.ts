export const tagSlug = (t: string) => t.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

/** Post URLs keep the shape Jekyll shipped for years: /YYYY/MM/DD/slug/ */
export const postHref = (post: { id: string; data: { date: Date } }) => {
  const iso = post.data.date.toISOString().slice(0, 10);
  const name = post.id.replace(/^\d{4}-\d{2}-\d{2}-/, '');
  return `/${iso.slice(0, 4)}/${iso.slice(5, 7)}/${iso.slice(8, 10)}/${name}/`;
};

export const stamp = (d: Date) =>
  d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }).toUpperCase();

/** Rounded to the nearest minute at 200 wpm, floored at 1. Code blocks are
 * skimmed rather than read, so they count at a third weight. */
export const readingTime = (body = '') => {
  const code = (body.match(/```[\s\S]*?```/g) ?? []).join(' ');
  const prose = body.replace(/```[\s\S]*?```/g, ' ');
  const words = prose.split(/\s+/).filter(Boolean).length + code.split(/\s+/).filter(Boolean).length / 3;
  return Math.max(1, Math.round(words / 200));
};
