"""
Author: Reynaldo Gomez

Description:
    Lightweight experiment tracker for the WaferCNN project (EXP-15).
    Appends one row to results.csv after each training run so every experiment
    is permanently recorded and comparable.

Usage (at the end of any training script's main()):

    from tracker import log_run

    log_run(
        exp_id       = "EXP-01+07+04",
        model        = "WaferResNet",
        loss_fn      = "FocalLoss(gamma=2)",
        epochs       = 20,
        lr           = 3e-4,
        batch_size   = 128,
        augmentation = "rot90+hflip",
        val_accuracy = 0.9750,
        val_loss     = 0.1234,
        macro_f1     = 0.85,
        per_class_f1 = {"Center": 0.95, "Donut": 0.72, ...},
        best_epoch   = 17,
        train_time_s = 423.1,
        checkpoint   = "checkpoints/best_resnet_focal.pt",
        notes        = "Run 1: ResBlocks + FocalLoss + SiLU",
    )

How to get per_class_f1 from sklearn:
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    per_class_f1 = {cls: report[cls]["f1-score"] for cls in classes}

Output:
    results.csv — one header row, one data row per run, in Yield CNN/
    If the file does not exist it is created with headers on the first call.
    If a new experiment introduces a class the previous runs did not have,
    the new column is added and old rows get NaN for that field.
"""

import csv
from datetime import datetime
from pathlib import Path

# results.csv lives next to tracker.py, i.e. in Yield CNN/
RESULTS_CSV = Path(__file__).parent / "results.csv"

# Fixed column order for the non-class columns
_FIXED_COLS = [
    "timestamp",
    "exp_id",
    "model",
    "loss_fn",
    "epochs",
    "lr",
    "batch_size",
    "augmentation",
    "val_accuracy",
    "val_loss",
    "macro_f1",
    "best_epoch",
    "train_time_s",
    "checkpoint",
    "notes",
]


def log_run(
    exp_id: str,
    model: str,
    loss_fn: str,
    epochs: int,
    lr: float,
    batch_size: int,
    val_accuracy: float,
    val_loss: float,
    macro_f1: float,
    per_class_f1: dict,
    best_epoch: int,
    train_time_s: float,
    checkpoint: str,
    augmentation: str = "rot90+hflip",
    notes: str = "",
) -> None:
    """
    Append one experiment result row to results.csv.

    Args:
        exp_id        : Roadmap ID(s) for this run, e.g. "EXP-01+07+04"
        model         : Model class name, e.g. "WaferResNet"
        loss_fn       : Loss function description, e.g. "FocalLoss(gamma=2)"
        epochs        : Number of epochs trained
        lr            : Initial learning rate
        batch_size    : Training batch size
        val_accuracy  : Overall validation accuracy (0–1 float)
        val_loss      : Best validation loss
        macro_f1      : Macro-averaged F1 across all classes
        per_class_f1  : Dict mapping class name → F1 score, e.g. {"Donut": 0.72, ...}
        best_epoch    : Epoch at which best_val_loss was achieved
        train_time_s  : Total training wall-clock time in seconds
        checkpoint    : Relative path to the saved .pt file
        augmentation  : Short description of augmentation strategy (default: "rot90+hflip")
        notes         : Free-text notes about the run
    """
    # Sort class names so columns are always in the same order regardless of
    # the order they appear in per_class_f1
    class_cols = [f"f1_{cls}" for cls in sorted(per_class_f1.keys())]
    all_cols   = _FIXED_COLS + class_cols

    row = {
        "timestamp"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "exp_id"       : exp_id,
        "model"        : model,
        "loss_fn"      : loss_fn,
        "epochs"       : epochs,
        "lr"           : lr,
        "batch_size"   : batch_size,
        "augmentation" : augmentation,
        "val_accuracy" : round(val_accuracy, 6),
        "val_loss"     : round(val_loss, 6),
        "macro_f1"     : round(macro_f1, 6),
        "best_epoch"   : best_epoch,
        "train_time_s" : round(train_time_s, 1),
        "checkpoint"   : checkpoint,
        "notes"        : notes,
    }
    for cls, score in per_class_f1.items():
        row[f"f1_{cls}"] = round(score, 6)

    file_exists = RESULTS_CSV.exists()

    if file_exists:
        with RESULTS_CSV.open("r", newline="") as f:
            reader        = csv.DictReader(f)
            existing_cols = list(reader.fieldnames or [])
            existing_rows = list(reader)

        # Union: keep existing order, append any new columns at the end
        merged_cols    = existing_cols + [c for c in all_cols if c not in existing_cols]
        new_cols_added = len(merged_cols) > len(existing_cols)

        # If new columns appeared, rewrite the whole file so the header stays valid
        if new_cols_added:
            with RESULTS_CSV.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=merged_cols, extrasaction="ignore")
                writer.writeheader()
                for old_row in existing_rows:
                    writer.writerow({col: old_row.get(col, "") for col in merged_cols})
    else:
        merged_cols    = all_cols
        new_cols_added = False

    # Append the new row (or write fresh file)
    mode = "a" if file_exists else "w"
    with RESULTS_CSV.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=merged_cols, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        full_row = {col: row.get(col, "") for col in merged_cols}
        writer.writerow(full_row)

    print(f"[tracker] Logged to {RESULTS_CSV.name}  ({exp_id}  macro_f1={macro_f1:.4f}  acc={val_accuracy:.4f})")


def print_results() -> None:
    """
    Print results.csv to the terminal as a formatted table.
    Useful for a quick comparison across runs without opening a spreadsheet.
    """
    if not RESULTS_CSV.exists():
        print("[tracker] No results.csv found — run an experiment first.")
        return

    with RESULTS_CSV.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("[tracker] results.csv is empty.")
        return

    # Key columns to display in the summary table
    summary_cols = ["timestamp", "exp_id", "model", "loss_fn",
                    "epochs", "val_accuracy", "macro_f1", "best_epoch", "notes"]
    # Add any f1_ columns that exist
    f1_cols = sorted(c for c in rows[0].keys() if c.startswith("f1_"))
    display_cols = summary_cols + f1_cols

    # Column widths
    widths = {col: max(len(col), max(len(str(r.get(col, ""))) for r in rows))
              for col in display_cols}

    header = "  ".join(col.ljust(widths[col]) for col in display_cols)
    sep    = "  ".join("-" * widths[col] for col in display_cols)

    print(header)
    print(sep)
    for r in rows:
        print("  ".join(str(r.get(col, "")).ljust(widths[col]) for col in display_cols))


if __name__ == "__main__":
    print_results()
