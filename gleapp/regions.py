"""Regional category schemes.

Each region (a country, or an agency such as Europol) has its own locked set of
categories, with its own codes. The examiner picks one at first launch and every
case created afterwards is seeded with it; a case records the region it was
created under (meta ``category_region``) and keeps it, because the codes stored
on its files only mean something under that scheme. A case from before regions
existed is a US case.

A scheme row is ``(code, name, color, illegal, key[, UNCAT])``: ``key`` is the
keyboard shortcut the scheme assigns (a digit, or for Denmark sometimes a
letter), and ``UNCAT`` marks the category a file starts in. ``illegal`` rows are
the notable ones, the ones the hash stash keeps.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_REGION = "US"
UNCAT = "uncat"
_UNCAT_COLOR = "#8b93a3"


@dataclass(frozen=True)
class Preset:
    code: int
    name: str
    color: str
    notable: bool
    key: str
    position: int


@dataclass(frozen=True)
class Scheme:
    id: str
    label: str
    presets: tuple[Preset, ...]
    uncategorized: int
    # where a known-good (NSRL) hit lands; None when the region names no such
    # category, and the file then stays uncategorized
    nonpertinent: int | None

    @property
    def codes(self) -> frozenset[int]:
        return frozenset(p.code for p in self.presets)

    @property
    def first_custom(self) -> int:
        """The lowest code an examiner-added category may take."""
        return max(self.codes) + 1

    @property
    def notable_codes(self) -> tuple[int, ...]:
        return tuple(sorted(p.code for p in self.presets if p.notable))

    def to_json(self) -> dict:
        return {"id": self.id, "label": self.label,
                "uncategorized": self.uncategorized,
                "nonpertinent": self.nonpertinent,
                "first_custom": self.first_custom,
                "keys": {p.key: p.code for p in self.presets if p.key}}


def _region(label: str, rows) -> Scheme:
    presets, uncat = [], None
    for pos, row in enumerate(rows):
        code, name, color, notable, key = row[:5]
        if len(row) > 5 and row[5] == UNCAT:
            uncat, color = code, _UNCAT_COLOR
        presets.append(Preset(code, name, color, notable, key, pos))
    assert uncat is not None, label
    return Scheme("", label, tuple(presets), uncat, None)


# US keeps the names, colors and notable flags GLEAPP has always used: codes
# 1-3 are notable, and the presets are listed in code order.
_SCHEMES = {
    "US": _region("United States (Project VIC)", [
        (0, "Non-pertinent",                        "#8bc34a", False, "0"),
        (1, "CAM (Child Abuse Material)",           "#c0392b", True,  "1"),
        (2, "Child Exploitative / Age Difficult",   "#f1c40f", True,  "2"),
        (3, "CGI / Animation (Child Exploitative)", "#8e44ad", True,  "3"),
        (4, "Comparison Images (Non-pertinent)",    "#2980b9", False, "4"),
        (5, "Uncategorized",                        "#8b93a3", False, "5", UNCAT),
    ]),
    "AU": _region("Australia", [
        (1, '1. CAM - Real Child Pre-Pubescent <13yo', "#ff0000", True, '1'),
        (2, '2. CAM - Other Illegal Content, Child <18yo', "#ff8000", True, '2'),
        (3, '3. Investigative Interest', "#ffff00", False, '3'),
        (4, '4. Ignorable', "#00c000", False, '4'),
        (0, '0. Unchecked', "#ffffff", False, '0', UNCAT),
    ]),
    "AT": _region("Austria", [
        (0, 'Nicht Kategorisiert', "#ffffff", False, '0', UNCAT),
        (1, 'CSEM', "#ff0000", True, '1'),
        (2, 'Präferenz', "#ffff00", False, '2'),
        (3, 'Nicht Strafbar', "#00ff00", False, '3'),
    ]),
    "CA-EN": _region("Canada (English)", [
        (1, 'Child Pornography', "#ff0000", True, '1'),
        (2, 'Investigative Interest', "#ffff00", False, '2'),
        (3, 'Other', "#008000", False, '3'),
        (0, 'Unknown', "#ffffff", False, '0', UNCAT),
    ]),
    "CA-FR": _region("Canada (French)", [
        (1, 'Pornographie Juvénile', "#ff0000", True, '1'),
        (2, 'Images d’intérêt', "#ffff00", False, '2'),
        (3, 'Autre Matériel', "#008000", False, '3'),
        (0, 'Inconnu', "#ffffff", False, '0', UNCAT),
    ]),
    "DK": _region("Denmark", [
        (7, 'NEW', "#00c0c0", False, 'n', UNCAT),
        (24, 'DK KNOWN', "#00ff00", False, 'k'),
        (8, 'REVIEW', "#ffff00", False, '4'),
        (21, 'DK CP CAT1', "#ff8080", True, 'o'),
        (22, 'DK CP CAT2', "#ff0000", True, 'p'),
        (23, 'DK CP CAT3', "#c00000", True, 'q'),
        (10, 'KNOWN LOCAL', "#00ff00", False, '0'),
        (11, 'CAT1 LOCAL', "#ff8080", True, '1'),
        (12, 'CAT2 LOCAL', "#ff0000", True, '2'),
        (13, 'CAT3 LOCAL', "#c00000", True, '3'),
    ]),
    "EE": _region("Estonia", [
        (3, 'Lapsporno', "#ff6347", True, '1'),
        (5, 'Räige lapsporno', "#800000", True, '2'),
        (4, 'OK', "#6b8e23", False, '3'),
        (2, 'Tundmatu', "#4682b4", False, '4', UNCAT),
        (7, '???', "#c0c0c0", False, '5'),
        (8, 'Võimalik uus ohver (LP)', "#ffd700", True, '6'),
    ]),
    "EUROPOL": _region("Europol", [
        (1, 'CSEM', "#ff0000", True, '1'),
        (2, 'INTEL', "#ffff00", False, '2'),
        (3, 'NON PERTINENT', "#7cfc00", False, '3'),
        (0, 'UNCATEGORISED', "#808080", False, '0', UNCAT),
    ]),
    "FR": _region("France", [
        (3, 'Pédopornographique', "#ff0000", True, '1'),
        (4, 'non illicite', "#6b8e23", False, '7'),
        (7, 'indetermine', "#c0c0c0", False, '5'),
        (2, 'Unknown', "#4682b4", False, '4', UNCAT),
        (5, 'Nudite mineur', "#ff8c00", True, '2'),
        (8, 'Mineur non illégal victime dans illegal', "#ffd700", False, '3'),
        (0, 'representation de mineur', "#e0e0e0", True, '6'),
    ]),
    "IE": _region("Ireland", [
        (0, 'Uncategorized', "#e0e0e0", False, '0', UNCAT),
        (1, 'Child Sexual', "#f00000", True, '1'),
        (2, 'Child Exposure', "#0070c0", True, '2'),
        (3, 'Child Indicative', "#00b050", False, '3'),
        (4, 'Child Ánime', "#ffff00", True, '4'),
        (5, 'For Review', "#7030a0", False, '5'),
        (6, 'Adult', "#938953", False, '6'),
        (7, 'Evidential', "#c3d69b", False, '7'),
        (8, 'Known Images', "#494429", False, '8'),
        (9, 'Body Parts', "#494429", False, '9'),
    ]),
    "KR": _region("South Korea", [
        (0, '해당사항없음 (Non-Pertinent)', "#c0ffc0", False, '0'),
        (1, '아동음란물 영상 - 불법 (Child Abuse Material (Illegal)', "#ffc0c0", True, '1'),
        (2, '아동음란물 의심 영상 (Child Exploitive (non-CAM) / Age Difficult)', "#ffffc0", False, '2'),
        (3, '애니메이션 (CGI / Animation - Child Exploitive)', "#e0e0e0", False, '3'),
        (4, '불법 촬영 영상 (Illegal video shooting)', "#f0f8ff", True, '4'),
        (5, '미분류 (Uncategorized)', "#404040", False, '5', UNCAT),
    ]),
    "NL": _region("Netherlands", [
        (1, 'Kinderporno', "#ff0000", True, '1'),
        (2, 'Overig', "#008000", False, '2'),
        (3, 'Dierenporno', "#ff8c00", True, '3'),
        (4, 'Vraagteken', "#ffff00", False, '4'),
        (0, 'Ongeclassificeerd', "#e0e0e0", False, '0', UNCAT),
        (5, 'Onderzoek', "#0070c0", False, '5'),
        (6, 'Bekeken - niet geclassificeerd', "#00ffff", False, '6'),
    ]),
    "NO": _region("Norway", [
        (0, 'IKKE KATEGORISERT', "#e0e0e0", False, '0', UNCAT),
        (1, 'ULOVLIG', "#ff0000", True, '1'),
        (2, 'ANNET', "#00ff00", False, '2'),
        (3, 'Filer av interesse', "#ffff00", False, '3'),
        (4, 'Ulovlig annen kilde', "#ffa07a", False, '4'),
        (5, 'Mulig', "#00bfff", False, '5'),
        (6, 'Annet annen kilde', "#8fbc8b", False, '6'),
    ]),
    "RO": _region("Romania", [
        (0, 'Necunoscut', "#d3d3d3", False, '0', UNCAT),
        (1, 'Materiale Pornografice Minori', "#ff0000", True, '1'),
        (2, 'Minori - Nuditate', "#ffa500", False, '2'),
        (3, 'Alte materiale relevante', "#ffff00", False, '3'),
        (4, 'Irelevant', "#008000", False, '4'),
    ]),
    "SE": _region("Sweden", [
        (6, 'Särskilt hänsynslöst', "#ff0000", True, '2'),
        (3, 'Övergreppsmaterial', "#ffff00", True, '1'),
        (4, 'Ej relevant', "#008000", False, '3'),
        (7, '???', "#c0c0c0", False, '7'),
        (2, 'Ej bedömt', "#c0ffc0", False, '0', UNCAT),
        (5, 'Svårbedömt', "#ff8000", False, '4'),
        (8, 'Stöder identifiering', "#17365d", False, '5'),
        (1, 'Färdigpuberterad <18 år', "#800080", False, '6'),
    ]),
    "CH": _region("Switzerland", [
        (1, 'KiPo', "#8b0000", True, '1'),
        (2, 'VKiPo', "#8b0000", True, '2'),
        (3, 'Zoophilie', "#8b0000", True, '3'),
        (0, 'Unknown', "#4682b4", False, '0', UNCAT),
        (4, 'Sexuelle_Gewalt', "#ff6347", True, '4'),
        (5, 'Verbotene_Gewalt', "#ff6347", True, '5'),
        (6, 'Praeferenzindikatoren', "#ffff80", True, '6'),
        (7, 'Kind', "#80ff80", False, '7'),
        (8, 'Unbedeutend', "#80ff80", False, '8'),
    ]),
    "UK": _region("United Kingdom", [
        (1, 'SC Category A', "#ff6347", True, '1'),
        (2, 'SC Category B', "#6b8e23", True, '2'),
        (3, 'SC Category C', "#ffc0ff", True, '3'),
        (4, 'Prohibited Images of Children', "#4682b4", True, '4'),
        (5, 'Extreme Pornography', "#800000", True, '5'),
        (6, 'Indicative / Borderline', "#ffd700", False, '6'),
        (7, 'Unconfirmed', "#d8d8d8", False, '7'),
        (8, 'Ignorable / Discounted', "#ffffff", False, '8'),
        (9, 'Supports Identification', "#000080", False, '9'),
        (0, 'Uncategorised', "#808080", False, '0', UNCAT),
    ]),
}

# Only where the region's own scheme names a non-pertinent category.
_NONPERTINENT = {"US": 0, "KR": 0, "EUROPOL": 3}

REGIONS: dict[str, Scheme] = {
    rid: Scheme(rid, s.label, s.presets, s.uncategorized, _NONPERTINENT.get(rid))
    for rid, s in _SCHEMES.items()
}


def get(region: str | None) -> Scheme:
    """The scheme for ``region``, US when it is unset or unknown."""
    return REGIONS.get((region or "").upper(), REGIONS[DEFAULT_REGION])


def is_region(region: str | None) -> bool:
    return (region or "").upper() in REGIONS


def choices() -> list[dict]:
    """[{id, label, categories}] for the region picker, US first, then by
    label."""
    rest = sorted((s for s in REGIONS.values() if s.id != DEFAULT_REGION),
                  key=lambda s: s.label)
    return [{"id": s.id, "label": s.label,
             "categories": [{"code": p.code, "name": p.name, "key": p.key}
                            for p in s.presets]}
            for s in [REGIONS[DEFAULT_REGION], *rest]]
