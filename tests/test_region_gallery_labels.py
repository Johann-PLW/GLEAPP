"""The gallery's Clear cat button and tile label follow the case's region.

Only a US case has Uncategorized at code 5. In a UK case 5 is Extreme
Pornography, and a Clear cat button hard-coded to 5 filed the selection there.
These pin the wiring in index.html and app.js.
"""

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "gleapp/web"
APPJS = WEB / "static/app.js"
INDEX = WEB / "templates/index.html"


def test_no_category_code_is_hard_coded_in_the_page():
    html = INDEX.read_text(encoding="utf-8")
    assert not re.search(r'data-cat="\d+"', html)
    assert 'id="selClear"' in html


def test_clear_cat_sends_the_cases_uncategorized_code():
    js = APPJS.read_text(encoding="utf-8")
    body = js[js.index("async function refreshCats()"):]
    body = body[:body.index("\n}\n")]
    assert re.search(r'\$\("#selClear"\)', body)
    assert re.search(r"\.dataset\.cat = UNCAT;", body)
    assert "catKey(UNCAT)" in body


def test_an_uncategorized_tile_shows_the_regions_own_name():
    js = APPJS.read_text(encoding="utf-8")
    assert '<div class="catbar none">${esc(uncatName())}</div>' in js
    assert '<div class="catbar none">Uncategorized</div>' not in js
