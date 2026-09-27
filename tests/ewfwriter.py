"""Minimal EWF-E01 and AFF writers, for the tests only.

Nothing on PyPI writes EWF and the reference implementation is LGPL, so an E01
to ingest has to be built here. This packs sections and chunk tables from the
format documentation and is a separate program from the reader it feeds.

The reader itself is proven in its own repository, against sets written by
ewfacquire and by FTK Imager. What these fixtures are for is the ingest around
it: that an .E01 is recognised as a source, carved, registered by offset, and
read back later by seeking to that offset.
"""

import hashlib
import os
import struct
import zlib

from gleapp.vendor import ewfprobe


def _section(out, name, payload, last=False):
    start = out.tell()
    size = ewfprobe.SECTION_SIZE + len(payload)
    nxt = start if last else start + size
    head = struct.pack("<16sQQ40s", name.encode("ascii").ljust(16, b"\x00"), nxt, size,
                       b"\x00" * 40)
    out.write(head + struct.pack("<I", zlib.adler32(head) & 0xFFFFFFFF) + payload)


def _volume(chunk_count, sectors_per_chunk, sector_size, sector_count):
    data = bytearray(1052)
    data[0] = 0x01                                        # fixed media
    struct.pack_into("<III", data, 4, chunk_count, sectors_per_chunk, sector_size)
    struct.pack_into("<Q", data, 16, sector_count)
    data[52] = 0x01                                       # compression: good
    return bytes(data)


def _header2():
    text = ("1\nmain\n"
            "c\tn\ta\te\tt\tav\tov\tm\tu\tp\n"
            "CASE-1\tEV-1\ttest image\tExaminer\tnotes here\t1.0\tTest OS\t"
            "2026 1 1 0 0 0\t2026 1 1 0 0 0\t\n")
    return zlib.compress(("﻿" + text).encode("utf-16-le"))


def _pack_chunks(chunks, compress):
    """The chunk data blob plus one table entry per chunk, offsets relative."""
    blob = bytearray()
    entries = []
    for chunk in chunks:
        rel = len(blob)
        if compress:
            packed = zlib.compress(chunk, 6)
            if len(packed) < len(chunk):
                entries.append(rel | 0x80000000)
                blob += packed
                continue
        entries.append(rel)
        blob += chunk + struct.pack("<I", zlib.adler32(chunk) & 0xFFFFFFFF)
    return bytes(blob), entries


def _table_payload(entries, base):
    head = struct.pack("<IIQI", len(entries), 0, base, 0)
    payload = head + struct.pack("<I", zlib.adler32(head) & 0xFFFFFFFF)
    body = b"".join(struct.pack("<I", e) for e in entries)
    return payload + body + struct.pack("<I", zlib.adler32(body) & 0xFFFFFFFF)


def sector_padded(data, sector_size=512):
    """A disk is a whole number of sectors, so the media is padded up to one."""
    return data + b"\x00" * (-len(data) % sector_size)


def write_ewf(folder, stem, data, *, chunk_size=4096, sector_size=512,
              compress=True, chunks_per_segment=None):
    """Write ``data`` as an EWF-E01 set and return the segment paths in order."""
    data = sector_padded(data, sector_size)
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)] or [b""]
    per_segment = chunks_per_segment or len(chunks)
    groups = [chunks[i:i + per_segment] for i in range(0, len(chunks), per_segment)]
    sectors_per_chunk = chunk_size // sector_size
    sector_count = (len(data) + sector_size - 1) // sector_size

    paths = []
    for index, group in enumerate(groups):
        path = os.path.join(str(folder), f"{stem}.E{index + 1:02d}")
        paths.append(path)
        last_segment = index == len(groups) - 1
        with open(path, "wb") as out:
            out.write(struct.pack("<8sBHH", ewfprobe.SIGNATURE, 1, index + 1, 0))
            if index == 0:
                _section(out, "header2", _header2())
                _section(out, "header", zlib.compress(b"1\nmain\nc\n\n"))
                _section(out, "volume",
                         _volume(len(chunks), sectors_per_chunk, sector_size, sector_count))
            blob, entries = _pack_chunks(group, compress)
            base = out.tell() + ewfprobe.SECTION_SIZE
            _section(out, "sectors", blob)
            _section(out, "table", _table_payload(entries, base))
            if last_segment:
                _section(out, "hash", hashlib.md5(data).digest() + b"\x00" * 16)
                _section(out, "done", b"", last=True)
            else:
                _section(out, "next", b"", last=True)
    return paths


# ---- AFF, from AFFLIB's documented layout (as in ewfprobe's own test suite) -----

def _aff_segment(out, name, data=b"", arg=0):
    raw = name.encode("utf-8")
    out.write(struct.pack(">4sIII", b"AFF\x00", len(raw), len(data), arg) + raw + data)
    out.write(struct.pack(">4sI", b"ATT\x00", 16 + len(raw) + len(data) + 8))


def write_aff(path, data, pages=None, page_size=4096, image_size=None):
    """Write the listed pages of data (all of them by default) as an AFF file:
    deflated pages, the sector and page size, and the image size and MD5 when
    image_size is given, which AFFLIB writes into one file of an AFD."""
    count = -(-len(data) // page_size)
    with open(path, "wb") as out:
        out.write(b"AFF10\r\n\x00")
        _aff_segment(out, "sectorsize", b"", 512)
        _aff_segment(out, "pagesize", b"", page_size)
        for n in (range(count) if pages is None else pages):
            page = data[n * page_size:(n + 1) * page_size]
            _aff_segment(out, f"page{n}", zlib.compress(page, 6), 0x01)
        if image_size is not None:
            _aff_segment(out, "imagesize",
                         struct.pack(">II", image_size & 0xFFFFFFFF, image_size >> 32), 2)
            _aff_segment(out, "md5", hashlib.md5(data).digest())
    return str(path)


def write_afd(folder, data, files=3, page_size=4096):
    """Write data as an AFD: a folder named .afd holding ``files`` AFF files that
    share the pages, the image size and hash in the last, as AFFLIB lays one out.
    Returns the paths of the files in order."""
    os.makedirs(folder, exist_ok=True)
    count = -(-len(data) // page_size)
    per = -(-count // files)
    paths = []
    for index in range(files):
        pages = range(index * per, min(count, (index + 1) * per))
        paths.append(write_aff(os.path.join(folder, f"file_{index:03d}.aff"), data,
                               pages, page_size,
                               len(data) if index == files - 1 else None))
    return paths
