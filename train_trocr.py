import argparse
import os
import time
from contextlib import nullcontext

import torch
from transformers import VisionEncoderDecoderModel, get_cosine_schedule_with_warmup

from dataset import get_recognition_dataloaders
from models.BERT import CharTokenizer
from models.TrOCR import build_trocr
from train_dbnet import get_device, init_wandb, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ViT -> BERT text recognizer on dataset_receipt word crops")
    parser.add_argument("--metadata", default="dataset_receipt/metadata.pkl")
    parser.add_argument("--img-dir", default="dataset_receipt/images")
    parser.add_argument("--encoder", default="google/vit-base-patch16-384")
    parser.add_argument("--decoder", default="bert-base-cased")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--max-length", type=int, default=40, help="Max target tokens (characters + [SEP])")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=1, help="Evaluate every N epochs (and on the last one)")
    parser.add_argument("--out-dir", default="checkpoints/trocr")
    parser.add_argument("--seed", type=int, default=42, help="Seeds the data split, shuffling, augmentations and init")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default="document-processing")
    parser.add_argument("--run-name", default=None, help="W&B run name (default: auto-generated)")
    return parser.parse_args()


def get_autocast(device):
    """bf16 where supported, else fp16 (needs a GradScaler). No autocast on CPU / MPS."""
    if device.type != "cuda":
        return nullcontext, None
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return (lambda: torch.autocast("cuda", dtype=dtype)), dtype


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, autocast):
    """Teacher-forced loss, plus greedy-decoded character error rate and exact word accuracy (val or test)."""
    model.eval()
    texts = loader.dataset.texts
    preds, total_loss = [], 0.0

    for image, labels in loader:
        image, labels = image.to(device), labels.to(device)
        with autocast():
            total_loss += model(pixel_values=image, labels=labels).loss.item()
            generated = model.generate(pixel_values=image)
        preds += [tokenizer.decode(ids) for ids in generated.cpu()]

    errors = sum(edit_distance(p, t) for p, t in zip(preds, texts))
    cer = errors / max(sum(len(t) for t in texts), 1)
    word_acc = sum(p == t for p, t in zip(preds, texts)) / len(texts)
    return total_loss / len(loader), cer, word_acc, list(zip(preds, texts))


def samples_table(samples, n=50):
    import wandb
    return wandb.Table(columns=["target", "prediction", "correct"], data=[[t, p, p == t] for p, t in samples[:n]])


def main():
    args = parse_args()
    device = get_device()
    autocast, amp_dtype = get_autocast(device)
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device} | AMP: {amp_dtype} | seed {args.seed}")
    run = init_wandb(args, "recognition")

    tokenizer = CharTokenizer(args.decoder)
    model = build_trocr(tokenizer, args.encoder, args.decoder, max_length=args.max_length).to(device)
    image_size = model.config.encoder.image_size
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M | input {image_size}x{image_size}")

    train_loader, valid_loader, test_loader = get_recognition_dataloaders(
        tokenizer, args.metadata, args.img_dir, image_size=image_size, batch_size=args.batch_size,
        num_workers=args.num_workers, max_length=args.max_length, seed=args.seed,
    )
    print(
        f"Train: {len(train_loader.dataset)} | Val: {len(valid_loader.dataset)}"
        f" | Test: {len(test_loader.dataset)} words"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.epochs * len(train_loader))
    scaler = torch.cuda.amp.GradScaler() if amp_dtype == torch.float16 else None

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        total_loss = 0.0

        for image, labels in train_loader:
            image, labels = image.to(device), labels.to(device)
            with autocast():
                loss = model(pixel_values=image, labels=labels).loss

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        print(
            f"Epoch {epoch}/{args.epochs} | loss {total_loss / len(train_loader):.4f}"
            f" | lr {scheduler.get_last_lr()[0]:.2e} | {time.time() - start:.0f}s"
        )
        if run:
            run.log({"train/loss": total_loss / len(train_loader), "lr": scheduler.get_last_lr()[0]}, step=epoch)

        if epoch % args.val_every == 0 or epoch == args.epochs:
            val_loss, cer, word_acc, samples = evaluate(model, valid_loader, tokenizer, device, autocast)
            print(f"  Val | loss {val_loss:.4f} | CER {cer:.4f} | word acc {word_acc:.3f}")
            for pred, text in samples[:5]:
                print(f"    {text!r:>20} -> {pred!r}")
            if run:
                run.log({"val/loss": val_loss, "val/cer": cer, "val/word_acc": word_acc,
                         "val/samples": samples_table(samples)}, step=epoch)

            model.save_pretrained(os.path.join(args.out_dir, "last"))
            tokenizer.save_pretrained(os.path.join(args.out_dir, "last"))
            if val_loss < best_loss:
                best_loss = val_loss
                model.save_pretrained(os.path.join(args.out_dir, "best"))
                tokenizer.save_pretrained(os.path.join(args.out_dir, "best"))
                print(f"  New best val loss {val_loss:.4f} -> {args.out_dir}/best")

    # Test words are only touched once, with the checkpoint picked on val.
    model = VisionEncoderDecoderModel.from_pretrained(os.path.join(args.out_dir, "best")).to(device)
    test_loss, cer, word_acc, samples = evaluate(model, test_loader, tokenizer, device, autocast)
    print(f"Test | loss {test_loss:.4f} | CER {cer:.4f} | word acc {word_acc:.3f}")
    if run:
        run.log({"test/samples": samples_table(samples)})
        run.summary.update({"test/loss": test_loss, "test/cer": cer, "test/word_acc": word_acc,
                            "best_val_loss": best_loss})
        run.finish()


if __name__ == "__main__":
    main()
