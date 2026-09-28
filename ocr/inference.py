"""
Two-step OCR with the trained models: DBNet finds word boxes, the ViT -> BERT recognizer
reads each box. Preprocessing is the same as at validation time (ocr/dataset.py), so the
models see what they were trained on.
"""
import os

import numpy as np
import torch

from ocr.dataset import crop_quad, get_recognition_transforms, transform as detection_transform
from ocr.getDBbboxes import extract_bounding_boxes
from ocr.models.BERT import CharTokenizer
from ocr.models.DBNET import DBNet

DETECTION_SIZE = 1280  # detection_transform resizes the longest side to this


def get_device():
    """CUDA if available, else CPU. Not MPS by default: on a 16 GB Mac the recognizer ran ~3.5x
    slower on MPS than on CPU (0.58 s vs 2.1 s per word). OCR_DEVICE=mps overrides."""
    if os.environ.get("OCR_DEVICE"):
        return torch.device(os.environ["OCR_DEVICE"])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def order_corners(box):
    """Corners as top-left, top-right, bottom-right, bottom-left (what crop_quad expects)."""
    box = np.asarray(box, dtype=np.float32)
    s, d = box.sum(axis=1), np.diff(box, axis=1).ravel()  # x + y, y - x
    return np.stack([box[s.argmin()], box[d.argmin()], box[s.argmax()], box[d.argmax()]])


class OCRModel:
    def __init__(self, dbnet_path="checkpoints/dbnet_best.pt", recognizer_path="checkpoints/trocr/best",
                 device=None, batch_size=8, box_thresh=0.3):
        from transformers import VisionEncoderDecoderModel

        self.device = device or get_device()
        self.batch_size = batch_size
        self.box_thresh = box_thresh

        self.detector = DBNet()
        self.detector.load_state_dict(torch.load(dbnet_path, map_location="cpu"))
        self.detector.to(self.device).eval()

        self.tokenizer = CharTokenizer(recognizer_path)
        self.recognizer = VisionEncoderDecoderModel.from_pretrained(recognizer_path).to(self.device).eval()
        _, self.crop_transform = get_recognition_transforms(self.recognizer.config.encoder.image_size)

    @torch.no_grad()
    def detect(self, image):
        """Word boxes, 4 corners each, in original image pixels."""
        h, w = image.shape[:2]
        x = detection_transform(image=image)["image"].unsqueeze(0).to(self.device)
        prob_map = self.detector(x)[0, 0].cpu().numpy()
        scale = max(h, w) / DETECTION_SIZE  # the transform only resizes and pads bottom/right
        return [order_corners(box) * scale for box in extract_bounding_boxes(prob_map, thresh=self.box_thresh)]

    @torch.no_grad()
    def recognize(self, image, boxes):
        """Text of each box, and a confidence: the geometric mean of the per-character probabilities."""
        texts, confidences = [], []
        crops = [self.crop_transform(image=crop_quad(image, box))["image"] for box in boxes]
        for i in range(0, len(crops), self.batch_size):
            batch = torch.stack(crops[i:i + self.batch_size]).to(self.device)
            out = self.recognizer.generate(pixel_values=batch, output_scores=True, return_dict_in_generate=True)
            # log-probability of each generated token, (batch, steps)
            logp = torch.stack([s.log_softmax(-1) for s in out.scores], dim=1)
            tokens = out.sequences[:, 1:]  # drop the decoder start token
            token_logp = logp.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
            # keep tokens up to and including the first [SEP]; what follows is padding
            seps = (tokens == self.tokenizer.sep_id).cumsum(dim=1)
            keep = (seps == 0) | ((seps == 1) & (tokens == self.tokenizer.sep_id))
            for seq, lp, k in zip(out.sequences.cpu(), token_logp.cpu(), keep.cpu()):
                texts.append(self.tokenizer.decode(seq))
                confidences.append(float(lp[k].mean().exp()) if k.any() else 0.0)
        return texts, confidences

    def read(self, image):
        """Words of an RGB image: [{"text", "polygon", "confidence"}], skipping boxes read as empty."""
        boxes = self.detect(image)
        texts, confidences = self.recognize(image, boxes) if boxes else ([], [])
        return [
            {"text": text.strip(), "polygon": np.round(box).astype(int).tolist(), "confidence": round(conf, 4)}
            for box, text, conf in zip(boxes, texts, confidences)
            if text.strip()
        ]
