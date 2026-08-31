/**
 * Subset the self-hosted faces to the characters this site can actually
 * render, keeping the variable axes we use.
 *
 * Why: fontsource ships latin and latin-ext as separate files with the full
 * Unicode block each. That was 200 KB across four requests, and because
 * "Wrocław" puts a latin-ext glyph in the hero, the 85 KB latin-ext file sat
 * on the critical path. One subset file per family removes the split.
 *
 * The character set is the full printable Latin-1 range plus the European
 * accents and typographic marks the content uses — not just the glyphs
 * present today, so a new post does not silently lose a character.
 *
 * Run: node scripts/subset-fonts.mjs
 */
import subsetFont from 'subset-font';
import { readFile, writeFile, readdir } from 'node:fs/promises';
import { statSync } from 'node:fs';

const base = Array.from({ length: 0x7e - 0x20 + 1 }, (_, i) => String.fromCharCode(0x20 + i)).join('');
const latin1 = Array.from({ length: 0xff - 0xa0 + 1 }, (_, i) => String.fromCharCode(0xa0 + i)).join('');
// Polish, plus the wider European set, plus the punctuation and symbols used.
const extended =
  'ĄąĆćĘęŁłŃńÓóŚśŹźŻż' +
  'ĀāĂăĒēĔĕĖėĚěĞğĪīĬĭİıĶķĹĺĻļĽľŅņŇňŌōŎŏŐőŔŕŖŗŘřŞşŠšŢţŤťŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻżŽž' +
  '–—‘’‚“”„†‡•…‰′″‹›€₤₹™©®°±×÷≈≠≤≥←↑→↓⇒∂∆∏∑√∞∫µΩπ' +
  '¹²³½¼¾ˆ˜¸˛ˇ˘˙˚';

const files = (await readdir('public/fonts')).filter((f) => f.endsWith('.woff2') && !f.includes('subset'));
const text = base + latin1 + extended;

const groups = { archivo: [], chivo: [] };
for (const f of files) (f.startsWith('archivo') ? groups.archivo : groups.chivo).push(f);

for (const [family, list] of Object.entries(groups)) {
  // Prefer the widest source (latin-ext carries the accented glyphs).
  const src = list.find((f) => f.includes('latin-ext')) ?? list[0];
  const before = list.reduce((n, f) => n + statSync(`public/fonts/${f}`).size, 0);
  const buf = await readFile(`public/fonts/${src}`);
  const out = await subsetFont(buf, text, { targetFormat: 'woff2' });
  const name = `${family}-subset.woff2`;
  await writeFile(`public/fonts/${name}`, out);
  console.log(
    `${family.padEnd(8)} ${list.length} files ${(before / 1024).toFixed(0)}KB -> ${name} ${(out.length / 1024).toFixed(0)}KB` +
      `  (${Math.round((1 - out.length / before) * 100)}% smaller)`
  );
}
