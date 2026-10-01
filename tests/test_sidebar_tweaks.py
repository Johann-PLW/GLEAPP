"""Two sidebar details: the Source storage section and the progress bar's tooltip.

These pin the wiring in app.js; the behavior itself was checked in a browser.
"""

from pathlib import Path

APPJS = Path(__file__).resolve().parents[1] / "gleapp/web/static/app.js"


def _body(js, header):
    start = js.index(header)
    return js[start:js.index("\n}", start)]


def test_source_storage_is_a_section_closed_when_a_case_opens():
    body = _body(APPJS.read_text(encoding="utf-8"), "function renderSourcePanel")
    assert '<details class="fsec" id="srcFold"' in body
    assert "<summary>Source storage (${list.length})</summary>" in body
    # closed on the first draw, kept as the examiner set it on a redraw, and never
    # remembered past the case: no stored setting decides it
    assert 'const open = !!$("#srcFold")?.open;' in body
    assert "localStorage" not in body


def test_the_progress_bar_shows_its_whole_message_on_hover():
    js = APPJS.read_text(encoding="utf-8")
    block = js[js.index("// The sidebar is narrow"):js.index("/* Poll the shared background job")]
    assert '$("#taskProg")' in block
    assert "bar.title =" in block
    # every job writes the bar's text, so the tooltip follows the text rather than
    # being set by one of them
    assert "new MutationObserver(sync)" in block
