#!/usr/bin/env python3
"""Publish the CV without the phone number on it.

The CV that gets sent to people directly carries a phone number in its header.
The copy linked from the site must not: a linked PDF is crawled, indexed and
cached, and the phone is the one contact detail deliberately kept private.

This edits the page-1 content stream rather than re-typesetting, so every glyph
of the real document survives untouched. Two operators change:

  1. The phone icon, the number and the separator that follows are dropped, and
     the surviving run starts half the removed width further right — the contact
     line is centred, so half the prefix keeps it centred.
  2. `Td` positions text relative to the current line start, so step 1 would drag
     every later line left with it. The next line's own `Td` absorbs the shift.

Usage:  python3 scripts/strip-cv-phone.py "~/Downloads/CV___EN (7).pdf"

Needs pikepdf (`pip install pikepdf`). Verify the result before committing: the
phone must be absent from the extracted text, and both pages must still render.
"""

import os
import sys

import pikepdf

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'public', 'roman-bartusiak-cv.pdf')

# Phone icon (F16 "H"), the number, and the bullet separator before the email.
DROP = (b'/F16 9.9626 Tf -108.592 -11.955 Td [(H)]TJ'
        b'/F20 9.9626 Tf 11.929 0 Td [(+48)-333(880)-334(002)-333(457)]TJ\n'
        b'0.45 0.45 0.45 rg 0.45 0.45 0.45 RG\n'
        b'0.45 0.45 0.45 rg 0.45 0.45 0.45 RG\n'
        b'/F14 9.9626 Tf 82.469 0 Td [(\\017)]TJ'
        b'/F16 9.9626 Tf 14.944 0 Td [(B)]TJ')
KEEP = b'/F16 9.9626 Tf -53.921 -11.955 Td [(B)]TJ'

# 109.342pt comes off the front of the line; the next line takes back half of it.
SHIFT_OLD = b'/F40 8.9664 Tf -280.423 -30.568 Td [(Distributed)-343(Systems)]TJ'
SHIFT_NEW = b'/F40 8.9664 Tf -225.752 -30.568 Td [(Distributed)-343(Systems)]TJ'


def main(src: str) -> int:
    pdf = pikepdf.open(os.path.expanduser(src))
    page = pikepdf.Page(pdf.pages[0])
    data = page.obj['/Contents'].read_bytes()

    for old, new in ((DROP, KEEP), (SHIFT_OLD, SHIFT_NEW)):
        if data.count(old) != 1:
            # A new revision moved the header. Re-derive the offsets from the
            # content stream rather than guessing — see the module docstring.
            print(f'error: expected exactly one match for {old[:44]!r}, '
                  f'found {data.count(old)}. The CV layout changed.',
                  file=sys.stderr)
            return 1
        data = data.replace(old, new)
    page.obj['/Contents'].write(data)

    # The number can also reach the file through XMP.
    with pdf.open_metadata() as meta:
        for key, value in list(meta.items()):
            if '880' in str(value):
                print(f'cleared metadata key {key}')
                del meta[key]

    pdf.save(OUT, linearize=True)
    print(f'wrote {OUT}')
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
