import hashlib
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ingest(conn, img_dir):
    """Registers every image in img_dir as a `pending` document. Already-seen images are skipped."""
    paths = sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    new = 0
    with conn.transaction():
        for path in paths:
            cur = conn.execute(
                "INSERT INTO ops.documents (file_name, file_hash, image_path) VALUES (%s, %s, %s)"
                " ON CONFLICT (file_hash) DO NOTHING",
                (path.name, file_hash(path), str(path)),
            )
            new += cur.rowcount
    return {"found": len(paths), "new": new}
