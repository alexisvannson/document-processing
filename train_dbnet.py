import argparse
import os
import random
import time

import cv2
import numpy as np
import pandas as pd
import torch
from shapely.geometry import Polygon

from dataset import get_dataloaders
from DBLoss import DBLoss
from getDBbboxes import extract_bounding_boxes
from models.DBNET import DBNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train DBNet on dataset_receipt")
    parser.add_argument("--metadata", default="dataset_receipt/metadata.pkl")
    parser.add_argument("--img-dir", default="dataset_receipt/images")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=10, help="Evaluate every N epochs (and on the last one)")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Seeds the data split, shuffling, augmentations and init")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default="document-processing")
    parser.add_argument("--run-name", default=None, help="W&B run name (default: auto-generated)")
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_wandb(args, job_type):
    """W&B run, or None when --wandb is off (so the scripts run without an account)."""
    if not args.wandb:
        return None
    import wandb
    return wandb.init(project=args.wandb_project, name=args.run_name, job_type=job_type, config=vars(args))


def polygon_iou(a, b):
    pa, pb = Polygon(a), Polygon(b)
    if not pa.is_valid or not pb.is_valid:
        return 0.0
    inter = pa.intersection(pb).area
    return inter / (pa.area + pb.area - inter + 1e-6)


def match_boxes(pred_boxes, gt_boxes, iou_thresh=0.5):
    """Greedy one-to-one matching. Returns the number of true positives."""
    matched_gt = set()
    tp = 0
    for pred in pred_boxes:
        best_iou, best_j = 0.0, None
        for j, gt in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            iou = polygon_iou(pred, gt)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh:
            matched_gt.add(best_j)
            tp += 1
    return tp


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    """DB loss, and detection precision / recall / F1 at IoU >= 0.5, on a val or test loader."""
    model.eval()
    criterion = criterion or DBLoss()
    df = loader.dataset.df
    total_loss = 0.0
    tp = n_pred = n_gt = 0

    for idx, (image, gt_prob, prob_mask, gt_thresh, thresh_mask) in enumerate(loader):
        image, gt_prob, prob_mask, gt_thresh, thresh_mask = (
            t.to(device) for t in (image, gt_prob, prob_mask, gt_thresh, thresh_mask)
        )
        preds = model(image, return_all=True)
        total_loss += criterion(preds, gt_prob, gt_thresh, thresh_mask, prob_mask)[0].item()
        pred_boxes = extract_bounding_boxes(preds[0][0, 0].cpu().numpy())

        # The val transform resizes the longest side to 1280 and pads bottom/right only,
        # so ground-truth corners just need the same scale factor.
        row = df.iloc[idx]
        h, w = cv2.imread(os.path.join(loader.dataset.img_dir, row.file_name)).shape[:2]
        scale = 1280 / max(h, w)
        gt_boxes = [[[b[f"x{k}"] * scale, b[f"y{k}"] * scale] for k in range(1, 5)] for b in row.bboxes]

        tp += match_boxes(pred_boxes, gt_boxes)
        n_pred += len(pred_boxes)
        n_gt += len(gt_boxes)

    precision = tp / max(n_pred, 1)
    recall = tp / max(n_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    return total_loss / max(len(loader), 1), precision, recall, f1


def main():
    args = parse_args()
    device = get_device()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device} | seed {args.seed}")
    run = init_wandb(args, "detection")

    train_loader, valid_loader, test_loader = get_dataloaders(
        args.metadata, args.img_dir, batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed
    )
    print(
        f"Train: {len(train_loader.dataset)} | Val: {len(valid_loader.dataset)}"
        f" | Test: {len(test_loader.dataset)} receipts"
    )

    model = DBNet().to(device)
    criterion = DBLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.epochs, power=0.9)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        totals = {"loss": 0.0, "loss_prob": 0.0, "loss_binary": 0.0, "loss_thresh": 0.0}

        for image, gt_prob, prob_mask, gt_thresh, thresh_mask in train_loader:
            image, gt_prob, prob_mask, gt_thresh, thresh_mask = (
                t.to(device) for t in (image, gt_prob, prob_mask, gt_thresh, thresh_mask)
            )
            preds = model(image)
            loss, parts = criterion(preds, gt_prob, gt_thresh, thresh_mask, prob_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            totals["loss"] += loss.item()
            for k, v in parts.items():
                totals[k] += v.item()

        scheduler.step()
        n = len(train_loader)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            + " | ".join(f"{k} {v / n:.4f}" for k, v in totals.items())
            + f" | lr {scheduler.get_last_lr()[0]:.2e} | {time.time() - start:.0f}s"
        )
        if run:
            run.log({f"train/{k}": v / n for k, v in totals.items()} | {"lr": scheduler.get_last_lr()[0]}, step=epoch)

        torch.save(model.state_dict(), os.path.join(args.out_dir, "dbnet_last.pt"))

        if epoch % args.val_every == 0 or epoch == args.epochs:
            val_loss, precision, recall, f1 = evaluate(model, valid_loader, device, criterion)
            print(f"  Val | loss {val_loss:.4f} | precision {precision:.3f} | recall {recall:.3f} | F1 {f1:.3f}")
            if run:
                run.log({"val/loss": val_loss, "val/precision": precision, "val/recall": recall, "val/f1": f1},
                        step=epoch)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), os.path.join(args.out_dir, "dbnet_best.pt"))
                print(f"  New best val loss {val_loss:.4f} -> {args.out_dir}/dbnet_best.pt")

    # Test receipts are only touched once, with the checkpoint picked on val.
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "dbnet_best.pt"), map_location=device))
    test_loss, precision, recall, f1 = evaluate(model, test_loader, device, criterion)
    print(f"Test | loss {test_loss:.4f} | precision {precision:.3f} | recall {recall:.3f} | F1 {f1:.3f}")
    if run:
        run.summary.update({"test/loss": test_loss, "test/precision": precision, "test/recall": recall,
                            "test/f1": f1, "best_val_loss": best_loss})
        run.finish()


if __name__ == "__main__":
    main()
