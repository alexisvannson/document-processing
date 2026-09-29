import re

from data_platform.pipeline.db import process_documents
from data_platform.pipeline.lines import group_lines

# Token-level patterns. Amounts use thousands separators ("16,500"), so long unbroken digit
# runs are what look like identifiers.
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"^(?:\+?62|0)?8\d{8,11}$|^(?:\+?62|0)[2-9]\d{7,10}$")  # Indonesian mobile / landline
MASKED_CARD = re.compile(r"[*Xx#]{4,}\d{3,4}")
DIGIT_RUN = re.compile(r"\d{6,}")

# Line-level context: on these lines, identifiers follow the keyword.
PHONE_CONTEXT = re.compile(r"\b(TELP?|PHONE|HP|WA|FAX)\b", re.I)
CARD_CONTEXT = re.compile(r"\b(CARD|CREDIT|DEBIT|VISA|MASTER\w*|BCA|MANDIRI|BNI|BRI|EDC|APPR\w*)\b", re.I)
NAME_CONTEXT = re.compile(r"\b(CASHIER|KASIR|SERVER|WAITER|WAITRESS|NAMA|NAME|GUEST|TAMU)\b", re.I)


def luhn_valid(digits):
    total = 0
    for i, d in enumerate(int(c) for c in reversed(digits)):
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def token_pii(text):
    stripped = re.sub(r"[\s().-]", "", text)
    if EMAIL.search(text):
        return "EMAIL"
    if MASKED_CARD.search(text):
        return "CARD"
    # 13-19 digits passing Luhn is a card number; EAN-13 product barcodes mostly fail it
    if 13 <= len(stripped) <= 19 and stripped.isdigit() and luhn_valid(stripped):
        return "CARD"
    if PHONE.match(stripped):
        return "PHONE"
    return None


def redact_tokens(tokens):
    """Returns a pii_type (or None) for every token. Rules first; a NER model can slot in here."""
    pii = [token_pii(t["text"]) for t in tokens]
    for line in group_lines(tokens):
        text = " ".join(tokens[i]["text"] for i in line)
        for pos, i in enumerate(line):
            if pii[i]:
                continue
            word = tokens[i]["text"]
            if PHONE_CONTEXT.search(text) and DIGIT_RUN.search(word):
                pii[i] = "PHONE"
            elif CARD_CONTEXT.search(text) and DIGIT_RUN.search(word):
                pii[i] = "CARD"
            elif NAME_CONTEXT.search(text) and word.isalpha() and not NAME_CONTEXT.fullmatch(word):
                # words after the keyword on its line: "Kasir: Budi Santoso"
                if any(NAME_CONTEXT.search(tokens[j]["text"]) for j in line[:pos]):
                    pii[i] = "PERSON"
    return pii


def run_redact(conn, run_id):
    def redact_document(conn, doc):
        rows = conn.execute(
            "SELECT token_idx, text, polygon, confidence FROM raw_restricted.ocr_tokens"
            " WHERE doc_id = %s ORDER BY token_idx",
            (doc["doc_id"],),
        ).fetchall()
        tokens = [{"text": text, "polygon": polygon} for _, text, polygon, _ in rows]
        pii = redact_tokens(tokens)

        conn.execute("DELETE FROM raw.ocr_tokens WHERE doc_id = %s", (doc["doc_id"],))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO raw.ocr_tokens (doc_id, token_idx, text, pii_type, polygon, confidence)"
                " SELECT doc_id, token_idx, %s, %s, polygon, confidence FROM raw_restricted.ocr_tokens"
                " WHERE doc_id = %s AND token_idx = %s",
                [(f"[{p}]" if p else text, p, doc["doc_id"], idx) for (idx, text, _, _), p in zip(rows, pii)],
            )

    return process_documents(
        conn, run_id, "ocr_done", "redacted", "redact_failed", redact_document,
        outputs=["raw.ocr_tokens"],
    )
