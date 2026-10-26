"""Synthetic C2PA test files, copied from c2paprobe's tests/build.py
(github.com/abrignoni/c2paprobe, MIT, commit c51e601) so these tests need nothing
beyond this repository. Nothing here is evidence.

Writers for synthetic test files: CBOR, JUMBF, a C2PA manifest store, a DER
certificate, an RFC 3161 token and each container c2paprobe reads.

Everything here is built from the specifications, so a test that passes shows the
reader and these writers agree. That the reader also agrees with files signed by
real C2PA producers is shown by the corpus comparison recorded in c2paprobe's README, not
by these tests.
"""

import struct
import zlib

C2PA_STORE_UUID = bytes.fromhex('6332706100110010800000aa00389b71')
MANIFEST_UUID = bytes.fromhex('63326d6100110010800000aa00389b71')
UPDATE_UUID = bytes.fromhex('6332756d00110010800000aa00389b71')
ASSERTIONS_UUID = bytes.fromhex('6332617300110010800000aa00389b71')
CLAIM_UUID = bytes.fromhex('6332636c00110010800000aa00389b71')
SIGNATURE_UUID = bytes.fromhex('6332637300110010800000aa00389b71')
CBOR_UUID = bytes.fromhex('63626f7200110010800000aa00389b71')
IPTC = 'http://cv.iptc.org/newscodes/digitalsourcetype/'


# --- CBOR ---------------------------------------------------------------------
def _head(major, n):
    if n < 24:
        return bytes([major << 5 | n])
    for info, size in ((24, 1), (25, 2), (26, 4), (27, 8)):
        if n < 1 << (8 * size):
            return bytes([major << 5 | info]) + n.to_bytes(size, 'big')
    raise ValueError(n)


def cbor(v):
    if v is None:
        return b'\xf6'
    if v is True:
        return b'\xf5'
    if v is False:
        return b'\xf4'
    if isinstance(v, int):
        return _head(0, v) if v >= 0 else _head(1, -1 - v)
    if isinstance(v, float):
        return b'\xfb' + struct.pack('>d', v)
    if isinstance(v, (bytes, bytearray)):
        return _head(2, len(v)) + bytes(v)
    if isinstance(v, str):
        raw = v.encode('utf-8')
        return _head(3, len(raw)) + raw
    if isinstance(v, (list, tuple)):
        return _head(4, len(v)) + b''.join(cbor(x) for x in v)
    if isinstance(v, dict):
        return _head(5, len(v)) + b''.join(cbor(k) + cbor(x) for k, x in v.items())
    raise TypeError(type(v))


def tagged(tag, v):
    return _head(6, tag) + cbor(v)


# --- JUMBF --------------------------------------------------------------------
def box(btype, payload):
    return struct.pack('>I', 8 + len(payload)) + btype + payload


def jumd(uuid, label):
    return box(b'jumd', uuid + b'\x03' + label.encode() + b'\x00')


def superbox(uuid, label, *children):
    return box(b'jumb', jumd(uuid, label) + b''.join(children))


def cbor_box(label, value):
    return superbox(CBOR_UUID, label, box(b'cbor', cbor(value)))


# --- DER ----------------------------------------------------------------------
def der(tag, value):
    n = len(value)
    if n < 0x80:
        return bytes([tag, n]) + value
    raw = n.to_bytes((n.bit_length() + 7) // 8, 'big')
    return bytes([tag, 0x80 | len(raw)]) + raw + value


def seq(*items):
    return der(0x30, b''.join(items))


def oid(raw):
    return der(0x06, raw)


def name(cn, org):
    def atv(o, val):
        return der(0x31, seq(oid(o), der(0x0C, val.encode())))
    return seq(atv(bytes((0x55, 0x04, 0x0A)), org), atv(bytes((0x55, 0x04, 0x03)), cn))


def certificate(subject=('Test Signer', 'Example Org'), issuer=('Test CA', 'Example CA'),
                not_before=b'250101000000Z', not_after=b'20991231235959Z'):
    alg = seq(oid(bytes((0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x04, 0x03, 0x02))))
    nb_tag = 0x17 if len(not_before) == 13 else 0x18
    na_tag = 0x17 if len(not_after) == 13 else 0x18
    tbs = seq(der(0xA0, der(0x02, b'\x02')), der(0x02, b'\x01'), alg, name(*issuer),
              seq(der(nb_tag, not_before), der(na_tag, not_after)), name(*subject),
              seq(seq(oid(bytes((0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x02, 0x01)))), der(0x03, b'\x00\x04')))
    return seq(tbs, alg, der(0x03, b'\x00' + b'\x11' * 8))


TSTINFO_OID = bytes((0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x09, 0x10, 0x01, 0x04))
SIGNED_DATA_OID = bytes((0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x07, 0x02))


def timestamp_token(gentime=b'20250602101112Z'):
    tstinfo = seq(der(0x02, b'\x01'), oid(b'\x2a\x03'), seq(seq(oid(b'\x60\x86\x48')), der(0x04, b'\x00' * 32)),
                  der(0x02, b'\x07'), der(0x18, gentime))
    encap = seq(oid(TSTINFO_OID), der(0xA0, der(0x04, tstinfo)))
    signed = seq(der(0x02, b'\x03'), der(0x31, b''), encap)
    return seq(oid(SIGNED_DATA_OID), der(0xA0, signed))


def cose_sign1(cert=None, tst=True, x5_protected=True, alg=-7):
    cert = certificate() if cert is None else cert
    protected = {1: alg}
    unprotected = {}
    if x5_protected:
        protected[33] = [cert]
    else:
        unprotected[33] = [cert]
    if tst:
        unprotected['sigTst'] = {'tstTokens': [{'val': timestamp_token()}]}
    return [cbor(protected), unprotected, None, b'\x00' * 64]


# --- manifest store -----------------------------------------------------------
def manifest(label, *, generator='Test Generator 1.0', title='image.jpg', fmt='image/jpeg',
             source_types=('trainedAlgorithmicMedia',), actions=('c2pa.created',),
             agent='Test Model', author='A. Author', ingredient=None, update=False,
             cose=None, claim_generator_info=False):
    acts = []
    for i, act in enumerate(actions):
        entry = {'action': act, 'softwareAgent': agent}
        if i < len(source_types) and source_types[i]:
            entry['digitalSourceType'] = (source_types[i] if '/' in source_types[i]
                                          else IPTC + source_types[i])
        acts.append(entry)
    assertions = [cbor_box('c2pa.actions.v2', {'actions': acts})]
    if author:
        assertions.append(cbor_box('stds.schema-org.CreativeWork',
                                   {'author': [{'@type': 'Person', 'name': author}]}))
    if ingredient:
        assertions.append(cbor_box('c2pa.ingredient.v3', ingredient))
    claim = {'dc:title': title, 'dc:format': fmt, 'instanceID': 'xmp:iid:test'}
    if claim_generator_info:
        claim['claim_generator_info'] = [{'name': generator, 'version': '2.0'}]
    else:
        claim['claim_generator'] = generator
    return superbox(UPDATE_UUID if update else MANIFEST_UUID, label,
                    superbox(ASSERTIONS_UUID, 'c2pa.assertions', *assertions),
                    superbox(CLAIM_UUID, 'c2pa.claim.v2', box(b'cbor', cbor(claim))),
                    superbox(SIGNATURE_UUID, 'c2pa.signature',
                             box(b'cbor', tagged(18, cose if cose is not None else cose_sign1()))))


def store(*manifests):
    if not manifests:
        manifests = (manifest('urn:c2pa:test-1'),)
    return superbox(C2PA_STORE_UUID, 'c2pa', *manifests)


def xmp(source_type='trainedAlgorithmicMedia', creator_tool='Test Tool', provenance=None,
        as_attribute=True):
    dst = IPTC + source_type
    attrs = f' xmp:CreatorTool="{creator_tool}"'
    body = ''
    if as_attribute:
        attrs += f' Iptc4xmpExt:DigitalSourceType="{dst}"'
    else:
        body += f'<Iptc4xmpExt:DigitalSourceType>{dst}</Iptc4xmpExt:DigitalSourceType>'
    if provenance:
        attrs += f' dcterms:provenance="{provenance}"'
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" '
        'xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        f'xmlns:dcterms="http://purl.org/dc/terms/"{attrs}>{body}'
        '<dc:creator><rdf:Seq><rdf:li>Jane Doe</rdf:li></rdf:Seq></dc:creator>'
        '</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>').encode('utf-8')


# --- containers ---------------------------------------------------------------
def jpeg(c2pa=None, xmp_packet=None, segment_payload=65000, other_jumbf=False):
    """SOI, APP1 XMP, APP11 JUMBF segments, a quantisation table, SOS, data, EOI."""
    out = b'\xff\xd8'
    if xmp_packet is not None:
        seg = b'http://ns.adobe.com/xap/1.0/\x00' + xmp_packet
        out += b'\xff\xe1' + struct.pack('>H', len(seg) + 2) + seg
    instances = []
    if other_jumbf:
        instances.append(box(b'jumb', jumd(b'\x01' * 16, 'not-c2pa')))
    if c2pa is not None:
        instances.append(c2pa)
    for en, data in enumerate(instances, 1):
        header = data[:8]
        body = data[8:]
        chunks = [body[i:i + segment_payload] for i in range(0, len(body), segment_payload)] or [b'']
        for z, chunk in enumerate(chunks, 1):
            seg = b'JP' + struct.pack('>HI', en, z) + header + chunk
            out += b'\xff\xeb' + struct.pack('>H', len(seg) + 2) + seg
    out += b'\xff\xdb' + struct.pack('>H', 67) + b'\x00' + b'\x01' * 64
    out += b'\xff\xda' + struct.pack('>H', 8) + b'\x01\x01\x00\x00\x3f\x00'
    out += b'\x12\x34\xff\x00\x56' + b'\xff\xd9'
    return out


def png_chunk(ctype, data):
    return struct.pack('>I', len(data)) + ctype + data + struct.pack('>I', zlib.crc32(ctype + data))


def png(c2pa=None, xmp_packet=None, compress_xmp=False, iend=True):
    out = b'\x89PNG\r\n\x1a\n' + png_chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 0, 0, 0, 0))
    if xmp_packet is not None:
        text = zlib.compress(xmp_packet) if compress_xmp else xmp_packet
        flag = b'\x01\x00' if compress_xmp else b'\x00\x00'
        out += png_chunk(b'iTXt', b'XML:com.adobe.xmp\x00' + flag + b'\x00\x00' + text)
    if c2pa is not None:
        out += png_chunk(b'caBX', c2pa)
    out += png_chunk(b'IDAT', zlib.compress(b'\x00\x00'))
    if iend:
        out += png_chunk(b'IEND', b'')
    return out


def bmff_box(btype, payload):
    return struct.pack('>I', 8 + len(payload)) + btype + payload


def full_box(btype, version, flags, payload):
    return bmff_box(btype, bytes([version]) + flags.to_bytes(3, 'big') + payload)


C2PA_BMFF_UUID = bytes.fromhex('d8fec3d61b0e483c92975828877ec481')
XMP_BMFF_UUID = bytes.fromhex('be7acfcb97a942e89c71999491e3afac')


def c2pa_uuid_box(c2pa, purpose=b'manifest'):
    payload = C2PA_BMFF_UUID + b'\x00\x00\x00\x00' + purpose + b'\x00'
    if purpose == b'manifest':
        payload += struct.pack('>Q', 0)
    return bmff_box(b'uuid', payload + c2pa)


def mp4(c2pa=None, xmp_packet=None, udta_xmp=None, brand=b'mp42', trailing=b''):
    out = bmff_box(b'ftyp', brand + b'\x00\x00\x00\x00' + brand + b'isom')
    moov_children = bmff_box(b'mvhd', b'\x00' * 100)
    if udta_xmp is not None:
        moov_children += bmff_box(b'udta', bmff_box(b'XMP_', udta_xmp))
    out += bmff_box(b'moov', moov_children)
    if xmp_packet is not None:
        out += bmff_box(b'uuid', XMP_BMFF_UUID + xmp_packet)
    if c2pa is not None:
        out += c2pa_uuid_box(c2pa)
    out += bmff_box(b'mdat', b'\x00' * 4096)
    return out + trailing


def heif(c2pa=None, xmp_packet=None, method=0):
    """ftyp, meta (hdlr, iinf with a mime XMP item, iloc, optional idat), mdat."""
    ftyp = bmff_box(b'ftyp', b'heic\x00\x00\x00\x00mif1heic')
    hdlr = full_box(b'hdlr', 0, 0, b'\x00' * 4 + b'pict' + b'\x00' * 12 + b'\x00')
    infe_img = full_box(b'infe', 2, 0, struct.pack('>HH', 1, 0) + b'hvc1' + b'\x00')
    items = [infe_img]
    if xmp_packet is not None:
        items.append(full_box(b'infe', 2, 0, struct.pack('>HH', 2, 0) + b'mime' + b'XMP\x00'
                              + b'application/rdf+xml\x00'))
    iinf = full_box(b'iinf', 0, 0, struct.pack('>H', len(items)) + b''.join(items))

    def build(xmp_offset):
        entries = [(1, 0, 0, 100)]
        if xmp_packet is not None:
            entries.append((2, method, xmp_offset, len(xmp_packet)))
        body = bytes([0x44, 0x00])              # offset 4, length 4, base 0, index 0
        body += struct.pack('>H', len(entries))
        for item_id, meth, off, ln in entries:
            body += struct.pack('>HHH', item_id, meth, 0) + struct.pack('>H', 1)
            body += struct.pack('>II', off, ln)
        return full_box(b'iloc', 1, 0, body)

    idat = bmff_box(b'idat', xmp_packet) if (xmp_packet is not None and method == 1) else b''
    meta_len = len(full_box(b'meta', 0, 0, hdlr + iinf + build(0) + idat))
    head = ftyp + (c2pa_uuid_box(c2pa) if c2pa is not None else b'')
    mdat_payload_start = len(head) + meta_len + 8
    xmp_off = 0 if method == 1 else mdat_payload_start + 100
    meta = full_box(b'meta', 0, 0, hdlr + iinf + build(xmp_off) + idat)
    mdat_payload = b'\x00' * 100 + (xmp_packet if (xmp_packet is not None and method == 0) else b'')
    return head + meta + bmff_box(b'mdat', mdat_payload)


def riff(form=b'WEBP', c2pa=None, xmp_packet=None, lie_about_size=False):
    def chunk(cid, data):
        return cid + struct.pack('<I', len(data)) + data + (b'\x00' if len(data) & 1 else b'')
    body = form + chunk(b'VP8L' if form == b'WEBP' else b'fmt ', b'\x2f\x00\x00\x00\x00')
    if xmp_packet is not None:
        body += chunk(b'XMP ' if form != b'WAVE' else b'_PMX', xmp_packet)
    if c2pa is not None:
        body += chunk(b'C2PA', c2pa)
    size = len(body) + (1000 if lie_about_size else 0)
    return b'RIFF' + struct.pack('<I', size) + body


def gif_sub_blocks(data):
    out = b''
    for i in range(0, len(data), 255):
        part = data[i:i + 255]
        out += bytes([len(part)]) + part
    return out + b'\x00'


def gif(c2pa=None, xmp_packet=None, trailer=True):
    out = b'GIF89a' + struct.pack('<HH', 1, 1) + b'\x80\x00\x00' + b'\x00\x00\x00\xff\xff\xff'
    if xmp_packet is not None:
        magic = bytes([1]) + bytes(range(255, -1, -1))
        out += b'\x21\xff\x0bXMP DataXMP' + xmp_packet + magic + b'\x00'
    if c2pa is not None:
        out += b'\x21\xff\x0bC2PA_GIF\x01\x00\x00' + gif_sub_blocks(c2pa)
    out += b'\x2c' + struct.pack('<HHHH', 0, 0, 1, 1) + b'\x00' + b'\x02' + gif_sub_blocks(b'\x44\x01')
    if trailer:
        out += b'\x3b'
    return out


def tiff(c2pa=None, xmp_packet=None, big_endian=False):
    e = '>' if big_endian else '<'
    tags = [(256, 3, 1, struct.pack(e + 'H', 1) + b'\x00\x00')]
    blobs = []
    if xmp_packet is not None:
        blobs.append((700, 1, xmp_packet))
    if c2pa is not None:
        blobs.append((52545, 7, c2pa))
    n = len(tags) + len(blobs)
    ifd_off = 8
    data_off = ifd_off + 2 + 12 * n + 4
    entries = [(t, ft, c, v) for t, ft, c, v in tags]
    data = b''
    for tag, ftype, blob in blobs:
        entries.append((tag, ftype, len(blob), struct.pack(e + 'I', data_off + len(data))))
        data += blob
    entries.sort()
    head = (b'MM' if big_endian else b'II') + struct.pack(e + 'HI', 42, ifd_off)
    ifd = struct.pack(e + 'H', n) + b''.join(struct.pack(e + 'HHI', t, ft, c) + v for t, ft, c, v in entries)
    return head + ifd + struct.pack(e + 'I', 0) + data


def jxl(c2pa=None, xmp_packet=None, brotli_xmp=False):
    out = b'\x00\x00\x00\x0cJXL \r\n\x87\n' + bmff_box(b'ftyp', b'jxl \x00\x00\x00\x00jxl ')
    if xmp_packet is not None:
        out += bmff_box(b'brob', b'xml ' + b'\x00' * 16) if brotli_xmp else bmff_box(b'xml ', xmp_packet)
    if c2pa is not None:
        out += c2pa
    return out + bmff_box(b'jxlc', b'\xff\x0a' + b'\x00' * 16)
