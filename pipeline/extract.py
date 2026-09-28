import re

from pipeline.db import process_documents
from pipeline.lines import group_lines

EXTRACTOR_VERSION = "rules-v1"

# "16,500" "23.000" "Rp 45.000" "@28,000" "-5,000" "(5,000)" "20.00" "0". Both "," and "." are
# thousands separators on these (Indonesian) receipts; a 2-digit tail is read as cents.
AMOUNT = re.compile(r"^(?P<neg>[-(])?(?:rp\.?\s*)?@?(?P<int>\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,]\d{2})?\)?-?$", re.I)
QTY = re.compile(r"^(?:(\d{1,2})[xX]|[xX](\d{1,2}))$")  # "2x" / "x2"

# Checked in order: the first match decides the line's kind ("SUB TOTAL" before "TOTAL",
# "Discount BCA 15%" is a discount, "NON TUNAI" before "TUNAI", "Pay Cash Change" is change).
KINDS = [
    ("subtotal", r"\bSUB\s*-?\s*(TOTAL|TTL)\b|\bSUBTOTAL\b|\bSUBTTL\b"),
    ("total", r"\bTOTAL\s+BAYAR\b"),  # "amount to pay", not the cash handed over
    ("change", r"\bCHANGED?\b|\bKEMBALI(AN)?\b"),
    ("discount", r"\bDISC(OUNT)?\b|\bDISKON\b|\bPROMO\b"),
    ("card", r"\bCARD\b|\bCREDIT\b|\bDEBIT\b|\bVISA\b|\bMASTER|\bBCA\b|\bEDC\b|\bQRIS\b|\bGOPAY\b|\bOVO\b|\bKARTU\b|\bNON\s*TUNAI\b"),
    ("cash", r"\bCASH\b|\bTUNAI\b|\bTENDERED\b|\bBAYAR\b|\bPAY\b"),
    ("tax", r"\bTAX\b|\bPAJAK\b|\bPB\s*1\b|\bPPN\b"),
    ("service", r"\bSERVICE\b|\bSVC\b|\bSRV\b|\bCHG\b|\bCHRG\b"),
    ("rounding", r"\bROUNDING\b|\bPEMBULATAN\b"),
    ("count", r"ITEMS?\b|\bQTY\b"),  # "5.00 xITEMS"
    ("total", r"TOTAL\b|\bTTL\b|\bDUE\b|\bJUMLAH\b"),  # no leading \b: "***TOTAL"
]
KINDS = [(kind, re.compile(pattern, re.I)) for kind, pattern in KINDS]


def clean(text):
    return text.strip(":;*,")


def parse_amount(text):
    m = AMOUNT.match(clean(text))
    if not m:
        return None
    value = int(re.sub(r"[.,]", "", m["int"]))
    return -value if m["neg"] else value


NOT_WORDS = {"X", "@", "RP"}  # "2 x @ 12.000", "Rp 36.000"


def is_word(text):
    t = clean(text)
    return (any(c.isalpha() for c in t) and t.upper() not in NOT_WORDS
            and parse_amount(t) is None and not QTY.match(t))


def classify(line_text):
    for kind, pattern in KINDS:
        if pattern.search(line_text):
            return kind
    return None


def parse_line(texts):
    """Kind, name, qty, unit price and amount of one line, from its token texts left to right."""
    line_text = " ".join(texts)
    numbers = [(i, parse_amount(t)) for i, t in enumerate(texts) if parse_amount(t) is not None]
    words = [t for t in texts if is_word(t)]
    amount = numbers[-1][1] if numbers else None

    kind = classify(line_text)
    if kind is None:
        if amount is None:
            kind = "note"  # "Less Ice 70%"
        elif not words:
            kind = "unknown"
        else:
            kind = "item"

    row = {"kind": kind, "name": None, "qty": None, "unit_price": None, "amount": amount, "line_text": line_text}
    if kind != "item":
        return row

    row["name"] = " ".join(clean(w) for w in words)
    # Quantity: "2x" anywhere, else a small bare integer at the start ("1 EGG TART 13,000")
    # or just before the price ("Kopi Susu Kolonel 1 23.000").
    qty_idx = None
    for i, t in enumerate(texts):
        m = QTY.match(clean(t))
        if m:
            row["qty"], qty_idx = int(m[1] or m[2]), i
            break
    if row["qty"] is None:
        for i, value in numbers[:-1]:
            if clean(texts[i]).isdigit() and 0 < value < 100 and (i == 0 or i == numbers[-1][0] - 1):
                row["qty"], qty_idx = value, i
                break
    # Unit price: another amount besides the qty and the line amount ("@28,000 28,000").
    others = [value for i, value in numbers[:-1] if i != qty_idx]
    if others:
        row["unit_price"] = others[-1]
    return row


def extract_lines(tokens):
    """
    Parses a receipt into rows for raw.extracted_lines. A line without an amount followed
    by a line with no words (label on one row, amount on the next) is merged into one.
    Items only come before the total: an unlabelled amount below it ("Other Rp 39.600")
    is a payment line, so it's kept as `unknown` rather than counted as an item.
    """
    lines = group_lines(tokens)
    merged, i = [], 0
    while i < len(lines):
        line = lines[i]
        texts = [tokens[j]["text"] for j in line]
        if i + 1 < len(lines) and all(parse_amount(t) is None for t in texts):
            nxt = [tokens[j]["text"] for j in lines[i + 1]]
            if nxt and not any(is_word(t) for t in nxt) and any(parse_amount(t) is not None for t in nxt):
                line = line + lines[i + 1]
                i += 1
        merged.append(line)
        i += 1

    rows, seen_total = [], False
    for line_idx, line in enumerate(merged):
        row = parse_line([tokens[j]["text"] for j in line])
        if seen_total and row["kind"] == "item":
            row.update(kind="unknown", name=None, qty=None, unit_price=None)
        seen_total = seen_total or row["kind"] == "total"
        row.update(line_idx=line_idx, token_idxs=line)
        rows.append(row)
    return rows


def run_extract(conn, run_id):
    def extract_document(conn, doc):
        rows = conn.execute(
            "SELECT text, polygon FROM raw.ocr_tokens WHERE doc_id = %s ORDER BY token_idx", (doc["doc_id"],)
        ).fetchall()
        tokens = [{"text": text, "polygon": polygon} for text, polygon in rows]
        lines = extract_lines(tokens)
        if not any(r["kind"] == "total" and r["amount"] is not None for r in lines):
            raise ValueError("no TOTAL amount found")

        conn.execute("DELETE FROM raw.extracted_lines WHERE doc_id = %s", (doc["doc_id"],))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO raw.extracted_lines (doc_id, line_idx, kind, name, qty, unit_price, amount,"
                " line_text, token_idxs, extractor_version) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    (doc["doc_id"], r["line_idx"], r["kind"], r["name"], r["qty"], r["unit_price"], r["amount"],
                     r["line_text"], r["token_idxs"], EXTRACTOR_VERSION)
                    for r in lines
                ],
            )

    return process_documents(
        conn, run_id, "redacted", "extracted", "extract_failed", extract_document,
        outputs=["raw.extracted_lines"],
    )
