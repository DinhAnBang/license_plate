"""Conservative Vietnam plate prefix data used by the T5 validator.

The set intentionally contains the legacy province prefixes that remain
visible on vehicles in circulation, rather than only the newest
administrative mapping.  It is an allow-list for a *known* prefix; the
validator treats an otherwise well-shaped unknown two-digit prefix as
UNCERTAIN so a future or legacy code is not discarded as a false positive.

Source for the domestic prefix table: Phu luc 02, Thong tu 79/2024/TT-BCA,
Cuc Canh sat giao thong:
https://www.csgt.vn/upload/services/669893851_TT79.2024.TT.BCA.pdf

The data is deliberately isolated from the rule implementation so a later
regulatory update can be reviewed as a data-only change.
"""

from __future__ import annotations


# Legacy/current domestic locality prefixes from the cited Appendix 02,
# including historical prefixes that can still occur on active vehicles.
# 80 is retained for central/state organizations commonly seen in traffic
# imagery.  00 and unlisted values are never treated as known valid prefixes.
KNOWN_VIETNAM_PROVINCE_CODES: frozenset[str] = frozenset(
    {
        "11",
        "12",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "30",
        "31",
        "32",
        "33",
        "34",
        "35",
        "36",
        "37",
        "38",
        "39",
        "40",
        "41",
        "43",
        "47",
        "48",
        "49",
        "50",
        "51",
        "52",
        "53",
        "54",
        "55",
        "56",
        "57",
        "58",
        "59",
        "60",
        "61",
        "62",
        "63",
        "64",
        "65",
        "66",
        "67",
        "68",
        "69",
        "70",
        "71",
        "72",
        "73",
        "74",
        "75",
        "76",
        "77",
        "78",
        "79",
        "80",
        "81",
        "82",
        "83",
        "84",
        "85",
        "86",
        "88",
        "89",
        "90",
        "92",
        "93",
        "94",
        "95",
        "97",
        "98",
        "99",
    }
)


def province_code_status(code: str) -> str:
    """Return ``VALID``, ``UNCERTAIN`` or ``INVALID`` for a two-digit code."""

    if len(code) != 2 or not code.isdigit():
        return "INVALID"
    if code == "00":
        return "INVALID"
    if code in KNOWN_VIETNAM_PROVINCE_CODES:
        return "VALID"
    # Do not reject a plausible historical/future code solely because this
    # data file has not yet been updated.
    return "UNCERTAIN"
