# document-processing

OCR for photographed receipts (the [CORD](https://github.com/clovaai/cord) dataset, 200 Indonesian receipts):
a **DBNet** detector finds the words, and a **TrOCR** recognizer fine-tuned on the receipts reads them.

## Results

All numbers are on the **test split**: 30 receipts (629 words) that were never used for training or for choosing
checkpoints. The receipts are split 70 / 15 / 15 into train / val / test, by receipt, with a fixed seed.

| Stage | Result |
| --- | --- |
| Detection (DBNet), IoU ≥ 0.5 | recall **0.949**, precision **0.913** (F1 0.93) |
| Recognition on ground-truth boxes (TrOCR fine-tuned, beam 4) | CER **0.031**, **90.5%** of words exactly right |
| End to end (DBNet boxes → TrOCR) | **84.1%** of words found *and* read exactly; CER 0.041 on the words found |
| Speed | 0.9 s per receipt on a Colab GPU, ~18 s on a laptop CPU |

The recognizers compared (`notebooks/eval_recognizers.ipynb`, all run through the same inference code):

| Recognizer | CER | Exact words | CER on numbers |
| --- | --- | --- | --- |
| My first recognizer, ViT → BERT trained from pretrained parts | 0.702 | 12.1% | 0.481 |
| `microsoft/trocr-base-printed`, zero-shot | 0.258 | 66.6% | 0.017 |
| `trocr-base-printed` fine-tuned on the receipts (greedy) | 0.032 | 90.3% | 0.008 |
| `trocr-base-printed` fine-tuned on the receipts (beam 4) | **0.031** | **90.5%** | **0.008** |

**What the OCR error costs downstream.** I also ran the OCR through a small extraction pipeline, which checks
that each receipt's items add up to its total. Test receipts that pass: **52%** with my OCR vs **60%** with
perfect (ground-truth) OCR. On receipts that pass in both runs, the extracted total is the same **99%** of the time
(`notebooks/eval_system.ipynb`, see [Beyond the brief](#beyond-the-brief-a-data-platform-around-the-ocr)).

## Approach

### Detection: DBNet (`ocr/models/DBNET.py`)

A ResNet-18 backbone with a feature pyramid predicts, for every pixel, the probability that it is text, plus a
threshold map. "Differentiable binarization" learns where to cut between neighbouring lines, which suits dense
receipt text. `ocr/getDBbboxes.py` turns the probability map back into boxes. I chose DBNet over:

| Approach | Why not (for receipts) |
| --- | --- |
| Faster R-CNN | slow and memory-hungry |
| YOLO | struggles with long, thin text lines and dense packing |
| UNet segmentation | merges close lines; heavy post-processing |
| CRAFT (`ocr/models/CRAFT.py`) | character-level; kept as an alternative, no training script |

### Recognition: from a failed first model to fine-tuned TrOCR

**First attempt: ViT → BERT** (`ocr/models/VIT.py`, `BERT.py`, `TrOCR.py`, `training/train_trocr.py`). A pretrained
ViT encoder and a pretrained BERT decoder, joined by cross-attention, with a character-level tokenizer. It scored a
CER of 0.70 on test. `notebooks/eda_words.ipynb` explains why:

- **The model didn't read the image.** Given a white or random-noise image instead of a word crop, it produced the
  same outputs (`Tol`, `1`). The cross-attention layers, the only link from image to text, are new in BERT and start
  random; their weights stayed at the random-initialization scale.
- **The data rewards not reading.** `1` is 8% of all words and `0` is 17% of all characters, so predicting the
  frequent patterns lowers the loss without looking at the crop.

**Second attempt: fine-tune `microsoft/trocr-base-printed`** (`notebooks/train_trocr_printed.ipynb`). Its
cross-attention is already trained to read printed text. On top of that:

| Problem | Fix |
| --- | --- |
| Frequent words dominate each epoch | weighted sampling: weight `1 / sqrt(frequency)`, up to 3× for words with rare characters (`1` drops from 8.0% to 0.7% of samples) |
| Few examples of rare characters and prices | synthetic words rendered on the fly (prices with `,` or `.`, quantities, rare characters), 30% on top of the real words |
| Loss can drop without reading | checkpoint chosen on validation **CER**, plus a "blind test" each epoch: CER on white images vs real crops |
| One learning rate for everything | separate learning rates for the encoder and the decoder, warmup on 10% of the steps |

Zero-shot, the model already scored 0.26 CER, and most of its errors were receipt conventions (`Subtotal` →
`SUBTOTAL`, `13,636` → `13.636`). One epoch of fine-tuning brought validation CER to 0.032.

### Robustness

Detector training uses augmentations for the physical defects of real photos: shadows and lighting gradients,
creases (elastic and grid distortion, rotations), and contrast changes. The recognizer uses lighter affine ones
(rotation, shear, scale) on each word crop.

## Error analysis

The fine-tuned recognizer gets 60 of the 629 test words wrong (`notebooks/eval_recognizers.ipynb`):

| Kind of error | Share | Examples (truth → prediction) |
| --- | --- | --- |
| a letter wrong, missing or extra | 47% | `COFFEE` → `COFFE`, `Bandeng` → `Bandang` |
| a digit wrong | 20% | `80,500` → `60,500`, `12000` → `1200` |
| case only | 18% | `Cash` → `cash`, `MILK` → `MILk` |
| other punctuation or spaces | 8% | `SUB_TOTAL` → `SUB TOTAL` |
| `,` vs `.` separator only | 7% | `20,000` → `20.000` |

Case and separator errors (a quarter of the total) are harmless once amounts and names are normalized.
Digit errors are the dangerous ones, because they change amounts.

**Can confidence flag the errors?** Confidence is the geometric mean of the token probabilities. Sending every
word below a threshold to human review:

| Flag if confidence < | Words flagged | Errors caught |
| --- | --- | --- |
| 0.80 | 7% | 37% |
| 0.90 | 15% | 58% |
| 0.95 | 24% | 85% |

At 0.95, a reviewer checks a quarter of the words and catches 85% of the errors. Some errors are confidently
wrong: `20,000` → `20.000` at 0.999. That's why the downstream check that amounts add up still matters.

## Repository layout

```
ocr/                  models and data code
  models/             DBNET.py (ResNet-18 + FPN + DB head); VIT.py + BERT.py + TrOCR.py (first recognizer); CRAFT.py
  dataset.py          detection and recognition datasets, augmentations, 70/15/15 split by receipt
  DBLoss.py           DBNet loss
  getDBprobMap.py     ground-truth probability and threshold maps from word boxes
  getDBbboxes.py      probability map -> word boxes
  inference.py        OCRModel: detect + recognize, used by the evaluation notebooks and the pipeline
training/             train_dbnet.py, train_trocr.py (W&B logging)
notebooks/
  train_model.ipynb           train DBNet (Colab)
  eval_dbnet.ipynb            evaluate DBNet, try it on your own image
  train_trocr.ipynb           train the first recognizer, ViT -> BERT (Colab)
  eda_words.ipynb             word / character imbalance, and why the first recognizer failed
  train_trocr_printed.ipynb   fine-tune trocr-base-printed (Colab)
  eval_recognizers.ipynb      compare the recognizers on test, error analysis, end to end with DBNet
  eval_system.ipynb           the OCR inside the extraction pipeline vs perfect OCR
results/              per-receipt outputs of the pipeline runs, read by eval_system.ipynb
data_platform/        optional extra: pipeline, dbt, Airflow, Metabase, agent (see below)
```

Not in git: `dataset_receipt/` (the data) and `checkpoints/` (trained weights).

## Quick start

**Prerequisites**
- Python 3.12. `make setup` creates `.venv` from `requirements.txt`.
- The data in `dataset_receipt/`: `images/*.png` plus `metadata.pkl`, a DataFrame with one row per receipt:
  `file_name`, `split_origin`, `words` (list of strings) and `bboxes` (list of 4-point boxes `x1..y4`).
- A GPU for training. The Colab notebooks copy the data from Google Drive, train, and save checkpoints back to
  Drive (`MyDrive/document-processing/checkpoints/`).

**Train**

```bash
make traindbnet EPOCHS=50 BATCH_SIZE=4     # -> checkpoints/dbnet_{best,last}.pt
```

The fine-tuned recognizer is trained in `notebooks/train_trocr_printed.ipynb` (Colab, GPU), which saves
`checkpoints/trocr-printed/best`. `make traintrocr` trains the first ViT → BERT recognizer.

**Run the OCR on an image**

```python
import cv2
from ocr.inference import OCRModel

ocr = OCRModel("checkpoints/dbnet_best.pt", "checkpoints/trocr-printed/best")
image = cv2.cvtColor(cv2.imread("dataset_receipt/images/test_receipt_00000.png"), cv2.COLOR_BGR2RGB)
boxes = ocr.detect(image)
texts, confidences = ocr.recognize(image, boxes, num_beams=4)
```

**Evaluate:** `notebooks/eval_recognizers.ipynb` runs in a few minutes on a Colab GPU, or about an hour on a CPU.

## Limitations and next steps

- **Small test set.** 30 receipts and 629 words, so a difference of one or two points between models may be noise.
- **Detection misses 5% of the words**, and they can't be recovered later. End-to-end accuracy (84%) is lower
  than recognition on ground-truth boxes (90%) mostly because of these misses.
- **Crops are stretched to a square** (384×384) although the median word is 2.5× wider than tall. Padding to keep
  the aspect ratio is worth trying.
- **Digit errors can be confidently wrong.** A digit-specific check (e.g. against the other amounts on the receipt)
  would catch more of them than a confidence threshold.
- **CPU inference is slow** (~18 s per receipt, mostly the 334M-parameter recognizer). Greedy decoding loses almost
  nothing (0.032 vs 0.031 CER) and is 2.5× faster than beam 4.

## Beyond the brief: a data platform around the OCR

After the OCR, I built an optional extra in [`data_platform/`](data_platform/README.md). It isn't part of the
brief. It shows how the OCR would be used on real documents, such as medical reports, which are also photographed
in bad conditions and contain personal data:

- a **pipeline** that runs the OCR on each receipt, redacts personal data, and extracts the items and totals into
  Postgres;
- **dbt** checks that publish a receipt only if its items add up to its total, and put the rest in quarantine;
- **Airflow** to run it daily, a **Metabase** dashboard, and a **LangGraph** agent that answers questions in plain
  English.

This is where the downstream numbers in [Results](#results) come from. Running the extraction on perfect OCR and
on my OCR, with everything else identical, isolates the cost of OCR errors (`notebooks/eval_system.ipynb`).
