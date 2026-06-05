import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef
)
from tqdm import tqdm

from dataset import PROTACDataset, collater
from model import *


# ==================== CONFIG ====================

BATCH_SIZE = 2
CHECKPOINT_PATH = "model/SE3-PROTACs_best.pt"

TRAIN_CSV = "data/train.csv"
VAL_CSV = "data/val.csv"
TEST_CSV = "data/test.csv"

MOL_DIR = "data/mol2_files/"
EMBEDDING_DIR = "data/esm_35/"


# ==================== EVALUATION FUNCTION ====================

def evaluate(model, loader, device, dataset_name="Dataset"):

    criterion = nn.CrossEntropyLoss()

    model.eval()

    y_true = []
    y_pred = []
    y_score = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for data_sample in tqdm(loader, desc=f"Evaluating {dataset_name}"):

            y = data_sample['label'].to(device)
            batch_size = y.size(0)

            outputs, _, _ = model(
                data_sample['ligase_ligand'].to(device),
                data_sample['ligase'].to(device),
                data_sample['target_ligand'].to(device),
                data_sample['target'].to(device),
                data_sample['linker'].to(device),
            )

            loss = criterion(outputs, y)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probs = torch.softmax(outputs, dim=1)

            y_score.extend(probs[:, 1].cpu().tolist())
            y_pred.extend(torch.argmax(outputs, dim=1).cpu().tolist())
            y_true.extend(y.cpu().tolist())

    avg_loss = total_loss / total_samples

    accuracy = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_score)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    aupr = average_precision_score(y_true, y_score)
    mcc = matthews_corrcoef(y_true, y_pred)

    print("\n" + "=" * 60)
    print(f"{dataset_name} METRICS")
    print("=" * 60)

    print(f"Loss:       {avg_loss:.4f}")
    print(f"Accuracy:   {accuracy:.4f}")
    print(f"AUROC:      {auroc:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"AUPR:       {aupr:.4f}")
    print(f"MCC:        {mcc:.4f}")

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "auroc": auroc,
        "precision": precision,
        "recall": recall,
        "aupr": aupr,
        "mcc": mcc
    }


# ==================== MAIN ====================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nUsing device: {device}")

    # ==================== LOAD DATASETS ====================

    # train_dataset = PROTACDataset(
    #     MOL_DIR,
    #     TRAIN_CSV,
    #     EMBEDDING_DIR
    # )

    val_dataset = PROTACDataset(
        MOL_DIR,
        VAL_CSV,
        EMBEDDING_DIR
    )

    # test_dataset = PROTACDataset(
    # MOL_DIR,
    # TEST_CSV,
    # EMBEDDING_DIR
    # )

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     collate_fn=collater,
    #     num_workers=4,
    #     pin_memory=True
    # )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collater,
        num_workers=4,
        pin_memory=True
    )

    # test_loader = DataLoader(
    # test_dataset,
    # batch_size=BATCH_SIZE,
    # shuffle=False,
    # collate_fn=collater,
    # num_workers=4,
    # pin_memory=True
    # )

    # ==================== BUILD MODEL ====================

    ligase_model = ESMWrapper()
    target_model = ESMWrapper()

    ESM_DIM = 128

    target_ligand_model = GraphTransformer(
        num_embeddings=10,
        dim=ESM_DIM
    )

    ligase_ligand_model = GraphTransformer(
        num_embeddings=10,
        dim=ESM_DIM
    )

    linker_model = GraphTransformer(
        num_embeddings=10,
        dim=ESM_DIM
    )

    model = Model(
        ligase_ligand_model,
        ligase_model,
        target_ligand_model,
        target_model,
        linker_model,
        dim=480
    )

    model = model.to(device)

    # ==================== LOAD CHECKPOINT ====================

    print(f"\nLoading checkpoint:\n{CHECKPOINT_PATH}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Validation AUROC: {checkpoint['val_auroc']:.4f}")

    # ==================== EVALUATE ====================

    # evaluate(model, train_loader, device, "TRAIN")

    evaluate(model, val_loader, device, "VALIDATION")

    # evaluate(model, test_loader, device, "TEST")

    # print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    # print(f"Test samples: {len(test_dataset)}")


if __name__ == "__main__":
    main()