import random

import pandas as pd
from psycopg.types.json import Jsonb

from pipeline.db import process_documents


class GroundTruthOCR:
    """
    Stands in for DBNet + the recognizer: returns the annotated words and boxes from
    metadata.pkl, so the rest of the pipeline can be built before the models are trained.

    corrupt > 0 swaps one digit in that fraction of the numeric tokens (seeded per file),
    to simulate recognition errors and exercise the downstream validation.
    """

    name = "ground_truth"

    def __init__(self, metadata_path, corrupt=0.0, seed=42):
        metadata = pd.read_pickle(metadata_path)
        self.records = {row.file_name: row for row in metadata.itertuples()}
        self.corrupt = corrupt
        self.seed = seed

    def __call__(self, doc):
        row = self.records.get(doc["file_name"])
        if row is None:
            raise KeyError(f"no ground truth for {doc['file_name']}")
        rng = random.Random(f"{self.seed}:{doc['file_name']}")
        tokens = []
        for text, box in zip(row.words, row.bboxes):
            if self.corrupt and any(c.isdigit() for c in text) and rng.random() < self.corrupt:
                text = swap_digit(text, rng)
            polygon = [[box[f"x{i}"], box[f"y{i}"]] for i in range(1, 5)]
            tokens.append({"text": text, "polygon": polygon, "confidence": 1.0})
        return tokens


class ModelOCR:
    """The trained models: DBNet boxes, then the ViT -> BERT recognizer on each box (ocr/inference.py)."""

    name = "model"

    def __init__(self, dbnet_path, recognizer_path):
        from ocr.inference import OCRModel  # torch & co. are only needed for this engine

        self.model = OCRModel(dbnet_path, recognizer_path)

    def __call__(self, doc):
        import cv2

        image = cv2.imread(doc["image_path"])
        if image is None:
            raise FileNotFoundError(doc["image_path"])
        return self.model.read(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def swap_digit(text, rng):
    positions = [i for i, c in enumerate(text) if c.isdigit()]
    i = rng.choice(positions)
    new = rng.choice([d for d in "0123456789" if d != text[i]])
    return text[:i] + new + text[i + 1:]


def run_ocr(conn, run_id, engine):
    def ocr_document(conn, doc):
        tokens = engine(doc)
        if not tokens:
            raise ValueError("no text found")
        conn.execute("DELETE FROM raw_restricted.ocr_tokens WHERE doc_id = %s", (doc["doc_id"],))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO raw_restricted.ocr_tokens (doc_id, token_idx, text, polygon, confidence)"
                " VALUES (%s, %s, %s, %s, %s)",
                [(doc["doc_id"], i, t["text"], Jsonb(t["polygon"]), t["confidence"]) for i, t in enumerate(tokens)],
            )
        conn.execute("UPDATE ops.documents SET ocr_source = %s WHERE doc_id = %s", (engine.name, doc["doc_id"]))

    return process_documents(
        conn, run_id, "pending", "ocr_done", "ocr_failed", ocr_document,
        outputs=["raw_restricted.ocr_tokens"],
    )
