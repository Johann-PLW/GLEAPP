"""A pass over an image that cannot open what locks part of it keeps what the case
recorded of that part.

Found by review 2026-10-16: ``pipeline.add_documents`` run in a session that had not
been given an encrypted source's key walked the image again and rewrote the source's
``:volumes`` from what it could see, so the volume inside the locked container was
dropped from the record. Every row read from that volume then had no recorded volume
to be read from, and a keyed ``process --force`` errored on each of them until the
source was ingested again. The fixtures are the ones
``test_image_keys_and_containers.py`` uses and describes.
"""

import gzip
import json
from pathlib import Path

import pytest

from gleapp import archive
from gleapp.case import open_case, parse_source_spec
from gleapp.pipeline import add_documents, ingest_sources, process

FIXTURES = Path(__file__).parent / "fixtures"
LUKS_PASSPHRASE = "correct horse battery staple"


@pytest.fixture(autouse=True)
def _isolate(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("GLEAPP_CONFIG_DIR", str(tmp_path_factory.mktemp("gleapp-cfg")))
    for held in ("_PASSWORDS", "_PRIVATE_KEYS", "_BITLOCKER", "_APFS", "_LUKS"):
        monkeypatch.setattr(archive, held, {})
    from gleapp import hashstore, stash  # pylint: disable=import-outside-toplevel
    hashstore.close()
    stash.close()
    yield
    archive.close_zips()
    hashstore.close()
    stash.close()


def _copy(tmp_path, name) -> Path:
    folder = tmp_path / "ev"
    folder.mkdir(exist_ok=True)
    dest = folder / name[:-3]
    dest.write_bytes(gzip.decompress((FIXTURES / name).read_bytes()))
    return dest


def _ingest(case, image):
    sources, _ = parse_source_spec(str(image))
    for s in sources:
        s.include_other = True
    ingest_sources(case, sources)


def _volumes(case, image) -> list:
    return json.loads(archive.source_record(case, image.name)["volumes"] or "[]")


def _restart(monkeypatch):
    """A new session: nothing that opened an image is held any more."""
    archive.close_zips()
    for held in ("_BITLOCKER", "_LUKS"):
        monkeypatch.setattr(archive, held, {})


@pytest.mark.parametrize("fixture, unlock", [
    ("luks1-cbc.img.gz",
     lambda image: archive.unlock_luks(image, LUKS_PASSPHRASE)),
    ("bitlocker-xts128.img.gz",
     lambda image: archive.unlock_bitlocker(
         image, startup_key=(FIXTURES / "bitlocker-xts128.BEK").read_bytes())),
])
def test_add_documents_without_the_key_keeps_the_locked_volume(tmp_path, monkeypatch,
                                                               fixture, unlock):
    image = _copy(tmp_path, fixture)
    case = open_case(tmp_path / "case", create=True, examiner="t")
    try:
        assert unlock(image)
        _ingest(case, image)
        recorded = _volumes(case, image)
        walked = [f for f in case.db.iter_files("source = ?", (image.name,))
                  if f["volume_base"] is not None]
        assert walked and {v["base"] for v in recorded} >= {
            int(f["volume_base"]) for f in walked}

        _restart(monkeypatch)
        add_documents(case)                     # no key in this session
        assert _volumes(case, image) == recorded

        assert unlock(image)
        stats = process(case, force=True, workers=1, keyframes=0, screen=False,
                        similar=False)
        assert stats.errors == 0, stats
        assert archive.stage_source(case, image.name) == len(walked)
    finally:
        case.close()
