"""Extract lab data from the paper's source report (UPPCB 2013 in the paper's references):
MoWR/NIH Roorkee (2013) "Report of the committee on pollution caused by leather tanning industry
to the water bodies/groundwater in Unnao district of Uttar Pradesh".
PDF: http://117.252.14.250:8080/jspui/handle/123456789/5036

    pdftotext -layout <report>.pdf data/refs/u.txt
    uv run python data/uppcb/extract.py data/refs/u.txt

Writes, next to this file:
  effluent_long.csv  every effluent value as printed (Tables 7.1-7.27), with its standard
  samples.csv        lab sheet for wqml.reshape (one row per station x season)
  groundwater.csv    Tables 7.28a/b, one row per well x season
  wells.csv          Table 5.2 well register (location, type, depth, locality)
  annex3_cgwb_2010.csv   Annex III: metals in groundwater/effluent, CGWB April 2010 (mg/L)
  annex4_arsenic_2011.csv  Annex IV: arsenic, CGWB Dec 2011 (ug/L; geogenic per the report)
  annex5_cgwb_2012.csv   Annex V: pH, EC, Cr(VI), total Cr (ug/L), F, CGWB March 2012
  river_ganga.csv    Annexes VI/VII: CWC monthly river data, Kanpur (u/s) and Bhitaura (d/s), 2001-2011
Cells are placed by character position under their column header, because blank cells and
wrapped parameter names break whitespace splitting.
"""
import re
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent
DATES = {"pre_monsoon": "2012-05-21", "post_monsoon": "2012-11-05"}  # report: May 21-22 / Nov 5-9, 2012
CAMPAIGN = {"pre_monsoon": "2012-pre", "post_monsoon": "2012-post"}
CODES = [  # printed parameter name -> code (first match wins; order matters)
    (r"^pH(?![a-zA-Z])", "pH"), (r"^EC", "EC"), (r"^TDS", "TDS"), (r"^TSS", "TSS"), (r"^Alkalinity", "Alkalinity"),
    (r"^Hardness", "Hardness"), (r"^Chloride", "Cl"), (r"^Sulph", "SO4"), (r"^Nitrate", "NO3"),
    (r"^Phosphate", "PO4"), (r"^Sodium", "Na"), (r"^Potassium", "K"), (r"^Calcium", "Ca"),
    (r"^Magnesium", "Mg"), (r"^Iron", "Fe"), (r"^Manganese", "Mn"), (r"^Copper", "Cu"), (r"^Nickel", "Ni"),
    (r"^Lead", "Pb"), (r"^Cadmium", "Cd"), (r"^Zinc", "Zn"), (r"^Chromium ?\(VI\)", "Cr6"),
    (r"^Total Chromium", "Cr"), (r"^Arsenic", "As"), (r"^Fluoride", "F"), (r"^COD", "COD"), (r"^BOD", "BOD"),
    (r"^Oil", "Oil"), (r"^Sulphide", "S"), (r"^Phenol", "Phenol"), (r"^Ammon", "NH3"),
]
VALUE = re.compile(r"(?<![\w.])(ND|BDL|-|\d+(?:\.\d+)?(?:\s?-\s?\d+(?:\.\d+)?)?)(?![\w.])")


def code(name):
    for pat, c in CODES:
        if re.search(pat, name.strip(), re.I):
            return c
    return None


def blocks(lines, start_pat, stop_pat):
    """Yield (title, lines) for each table: from a title line to the next title/stop."""
    idx = [i for i, l in enumerate(lines) if re.search(start_pat, l)]
    stop = next(i for i, l in enumerate(lines) if re.search(stop_pat, l))
    for a, b in zip(idx, idx[1:] + [stop]):
        yield lines[a].strip(), lines[a:b]


def effluent(lines):
    rows = []
    for title, blk in blocks(lines, r"^\s*Table 7\.(\d+) (Effluent )?Characteristics", r"^7\.2 Ground Water"):
        first = next(i for i, l in enumerate(blk) if re.match(r"^\s*1\.\s", l))
        sidx = max(i for i in range(first) if "(SW-" in blk[i])  # station header nearest above row 1
        found = {m.group(1): m.start() + len(m.group(0)) / 2 for m in re.finditer(r"\((SW-\d+)\)", blk[sidx])}
        head = next((i for i, l in enumerate(blk) if "Pre-" in l and "Post-" in l), None)
        if head is not None:  # layout 1: <station> Pre-/Post- column pairs
            cols = [m.start() + 2 for m in re.finditer(r"Pre-|Post-", blk[head])]
            stations = sorted(found, key=found.get)
            assert len(cols) == 2 * len(stations), (title, stations, cols)
            colmeta = [(stations[k // 2], ["pre_monsoon", "post_monsoon"][k % 2]) for k in range(len(cols))]
            start = head + 1
        else:  # layout 2: Inlet / Outlet columns, one season named under the title
            season = "post_monsoon" if any("Post-monsoon" in l for l in blk[:sidx]) else "pre_monsoon"
            hl = next(l for l in blk[:sidx + 1] if "Outlet" in l)
            cols = [m.start() + len(m.group(0)) / 2 for m in re.finditer(r"Inlet|Outlet", hl)]
            near = {k: min(found, key=lambda st: abs(found[st] - x)) for k, x in enumerate(cols)}
            colmeta = [(near[k] if abs(found[near[k]] - cols[k]) < 8 else None, season) for k in range(len(cols))]
            start = sidx + 1
        stdl = [m.start() + 4 for l in blk[:start] for m in re.finditer(r"Standards?", l)]
        centers = cols + stdl[:1]
        x0 = int(min(cols)) - 8

        # Group lines into table rows. A row may span lines: name on one line, values (and the
        # row number) on the next, unit ('mg/L') on a third.
        grouped, pending = [], []
        for l in blk[start:]:
            if not l.strip():
                if grouped:
                    break  # blank line after the rows = end of table
                continue
            num = re.match(r"^\s*\d+\.\s", l)
            vals = [(m.group(1).replace(" ", ""), m.start() + len(m.group(1)) / 2, m.start()) for m in VALUE.finditer(l, x0)]
            cut = vals[0][2] if vals else len(l)  # name = text before the first value (column may be narrow)
            part = (re.sub(r"^\s*\d+\.\s*", "", l[:cut]).strip(), [(v, x) for v, x, _ in vals])
            if not num:
                pending.append(part)
                continue
            if not part[0] and pending:  # wrapped: name was above the numbered line
                # a unit-only line ('mg/L') left over from the previous wrapped row belongs to it
                while pending and grouped and re.fullmatch(r"[mµ]?[gS]?/?[Lc]?m?|mg/L|µS/cm", pending[0][0]) \
                        and not pending[0][1]:
                    grouped[-1].append(pending.pop(0))
                grouped.append(pending + [part])
            else:
                if pending and grouped:
                    grouped[-1] += pending  # trailing unit line of the previous row
                grouped.append([part])
            pending = []
        if pending and grouped:
            grouped[-1] += pending

        for parts in grouped:
            name = " ".join(n for n, _ in parts if n)
            if not code(name):
                continue
            cells = {}
            for v, x in (t for _, vals in parts for t in vals):
                k = min(range(len(centers)), key=lambda i: abs(centers[i] - x))
                assert k not in cells, (title, name, parts)  # two values in one column = misparse
                cells[k] = v
            for k, v in cells.items():
                if k == len(cols):
                    continue
                st, season = colmeta[k]
                if st is None:  # column without a sampling station (e.g. no inlet sampled)
                    continue
                rows.append({"table": title.split(" Characteristics")[0].replace("Effluent", "").strip(),
                             "station_id": st, "season": season, "parameter": name, "pollutant": code(name),
                             "value": v, "standard": cells.get(len(cols))})
    return pd.DataFrame(rows)


def groundwater(lines):
    rows, season = [], None
    start = next(i for i, l in enumerate(lines) if "Table 7.28(a)" in l)
    stop = next(i for i, l in enumerate(lines) if i > start and re.match(r"^8\.\s+Concluding", l))
    for l in lines[start:stop]:
        if "Pre-monsoon" in l:
            season = "pre_monsoon"
        elif "Post-monsoon" in l:
            season = "post_monsoon"
        if "Parameters" in l and "G-" in l:
            wells = [(m.group(0), m.start() + len(m.group(0)) / 2) for m in re.finditer(r"G-\d+", l)]
            continue
        m = re.match(r"^\s*\d+\.\s+(.+?)\s{2,}(.*)$", l)
        if not m or not code(m.group(1)):
            continue
        vals = [(v.group(1), v.start() + len(v.group(1)) / 2) for v in VALUE.finditer(l, m.start(2))]
        assert len(vals) == len(wells), (season, l)
        for (w, _), (v, _) in zip(wells, vals):
            rows.append({"station_id": w, "season": season, "pollutant": code(m.group(1)), "value": v})
    return pd.DataFrame(rows)


def wells(lines):
    a = next(i for i, l in enumerate(lines) if "Table 5.2 Details of Ground Water" in l)
    b = next(i for i, l in enumerate(lines) if i > a and re.match(r"^6\.\s+Analytical", l))
    pat = r"^\s*\d+\.?\s+(G-\d+)\s+(.+?)\s{2,}(TW|IM II|Pvt\.?\s?H?P?|Sub\.?\s?P?u?m?p?|OW)\s+(\d+)\s+(.+?)\s*$"
    rows = [re.match(pat, l).groups() for l in lines[a:b] if re.match(pat, l)]
    w = pd.DataFrame(rows, columns=["well_id", "location", "type", "depth_ft", "locality"])
    assert len(w) == 55 and w["well_id"].is_unique
    return w


TYPES = r"(Effluent|Eff\.|IM ?II|IM-II|Pvt\. ?HP|Port HP|HP|DW|DTW|SP|TW|Drain|River|SW)"
CELL = r"(ND|BDL|-|\d+(?:\.\d+)?)"


def annex_rows(lines, title, pattern, cols):
    a = next(i for i, l in enumerate(lines) if title in l and i > 3000)
    b = next((i for i, l in enumerate(lines) if i > a + 5 and ("Annex" in l or "Water Quality Data" in l)), len(lines))
    rows = [m.groups() for l in lines[a:b] for m in [re.match(pattern, l)] if m]
    return pd.DataFrame(rows, columns=cols)


def annexes(lines):
    head = r"^\s*(\d+)\.+\s+(.+?)\s{2,}" + TYPES + r"\s+"
    a3 = annex_rows(lines, "Heavy Metals in Ground Water and Effluent Samples of Unnao",
                    head + r"\s+".join([CELL] * 8) + r"\s*$",
                    ["no", "location", "type", "Cu", "Cr", "Mn", "Zn", "Ni", "Pb", "Fe", "Cd"])
    a4 = annex_rows(lines, "Arsenic Content in Ground Water of Unnao",
                    head + r"(\d\d\.\d\d\.\d\d)\s+" + CELL + r"\s*$",
                    ["no", "location", "type", "date", "As_ugL"])
    a5 = annex_rows(lines, "Ground Water Quality in Unnao District, Uttar Pradesh",
                    head + r"\s+".join([CELL] * 5) + r"\s*$",
                    ["no", "location", "type", "pH", "EC", "Cr6_ugL", "Cr_ugL", "F"])
    return a3.drop_duplicates(), a4.drop_duplicates(), a5.drop_duplicates()


RIVER_LABELS = {"Temp": "temp_c", "Colour": "colour", "Odour": "odour", "EC": "EC", "pH": "pH", "NO3-N": "NO3N",
                "NO2-N": "NO2N", "BOD3-27": "BOD", "CO3": "CO3", "Na": "Na", "Ca": "Ca", "Cl": "Cl", "SO4": "SO4",
                "Mg": "Mg", "K": "K", "HCO3(mg/L)": "HCO3", "F": "F", "Fe": "Fe"}


def river(pdf):
    """CWC monthly river data (Annexes VI/VII). Blank cells vary by year, so cells are placed by
    their x position in the PDF under the (vertical) header label of each annex's first page."""
    import html
    import subprocess
    rows = []
    for station, first, last in [("Kanpur", 79, 82), ("Bhitaura", 83, 86)]:
        xml = subprocess.run(["pdftotext", "-f", str(first), "-l", str(last), "-bbox", pdf, "-"],
                             capture_output=True, text=True, check=True).stdout
        pages = xml.split("<page ")[1:]
        words = [[(float(a), float(b), float(c), html.unescape(t)) for a, b, c, t in
                  re.findall(r'xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)"[^>]*>([^<]+)</word>', pg)] for pg in pages]
        centers = {}
        for x0, y0, x1, t in words[0]:  # header labels on the first page
            if t in RIVER_LABELS and y0 < 212 and RIVER_LABELS[t] not in centers:  # header band above row 1
                centers[RIVER_LABELS[t]] = (x0 + x1) / 2
        assert len(centers) == len(RIVER_LABELS), (station, sorted(centers))
        for pg in words:
            lines = {}
            for x0, y0, x1, t in pg:
                lines.setdefault(round(y0), []).append((x0, x1, t))
            for y, ws in sorted(lines.items()):
                ws.sort()
                if len(ws) < 6 or not ws[0][2].isdigit() or not re.match(r"[A-Za-z.]+/\d\d$", ws[1][2]):
                    continue
                # month from the row number (row 1 = June 2001, monthly); labels have typos ('Ma./03')
                row = {"station": station, "month": pd.Timestamp("2001-06-01") + pd.DateOffset(months=int(ws[0][2]) - 1),
                       "label": ws[1][2]}
                assert ws[1][2].endswith("/" + row["month"].strftime("%y")), (station, ws[0][2], ws[1][2])
                for x0, x1, t in ws[2:]:
                    k = min(centers, key=lambda c: abs(centers[c] - (x0 + x1) / 2))
                    if k == "odour":
                        row[k] = (row.get(k, "") + " " + t).strip()
                        continue
                    if k == "colour":  # text column, left-aligned; "Light Brown" spans lines
                        row[k] = (row.get(k, "") + " " + t).strip()
                        continue
                    assert abs(centers[k] - (x0 + x1) / 2) < 15 and k not in row, (station, ws[1][2], k, t)
                    row[k] = t
                rows.append(row)
    return pd.DataFrame(rows)


def lab_sheet(long):
    wide = long.pivot_table(index=["station_id", "season"], columns="pollutant", values="value", aggfunc="first")
    wide = wide.reset_index()
    wide.insert(1, "campaign", wide["season"].map(CAMPAIGN))
    wide.insert(2, "sample_date", wide["season"].map(DATES))
    order = wide["station_id"].str.extract(r"(\d+)")[0].astype(int)
    return wide.assign(_o=order).sort_values(["_o", "season"], ascending=[True, False]).drop(columns="_o")


if __name__ == "__main__":
    # report typo fix: '0,82' -> '0.82' (a leading 0 before a comma is never a thousands separator)
    lines = [re.sub(r"\b0,(\d)", r"0.\1", l) for l in Path(sys.argv[1]).read_text().splitlines()]
    eff = effluent(lines)
    eff.to_csv(OUT / "effluent_long.csv", index=False)
    lab_sheet(eff).to_csv(OUT / "samples.csv", index=False)
    lab_sheet(groundwater(lines)).drop(columns="campaign").to_csv(OUT / "groundwater.csv", index=False)
    wells(lines).to_csv(OUT / "wells.csv", index=False)
    a3, a4, a5 = annexes(lines)
    a3.to_csv(OUT / "annex3_cgwb_2010.csv", index=False)
    a4.to_csv(OUT / "annex4_arsenic_2011.csv", index=False)
    a5.to_csv(OUT / "annex5_cgwb_2012.csv", index=False)
    river(str(Path(sys.argv[1]).with_name("uppcb_2013_unnao_tanning.pdf"))).to_csv(OUT / "river_ganga.csv", index=False)
    print(len(eff), "effluent values,", eff.station_id.nunique(), "stations")
