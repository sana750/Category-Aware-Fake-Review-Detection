import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader

from config import FAKE_LABEL, GENUINE_LABEL
from transformer_dataset import FakeReviewDataset


def select_f1_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
        pos_label=FAKE_LABEL,
    )
    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], np.finfo(float).eps)
    )
    best_score = f1_scores.max()
    best_indices = np.flatnonzero(np.isclose(f1_scores, best_score))
    best_index = min(
        best_indices,
        key=lambda index: abs(thresholds[index] - 0.5),
    )
    return float(thresholds[best_index])


def apply_threshold(probabilities, threshold):
    return np.where(
        np.asarray(probabilities) >= threshold,
        FAKE_LABEL,
        GENUINE_LABEL,
    )


def predict_transformer_probabilities(
    model,
    tokenizer,
    review_texts,
    max_length,
    batch_size,
):
    from transformers import DataCollatorWithPadding

    encodings = tokenizer(
        review_texts,
        truncation=True,
        max_length=max_length,
    )
    dataset = FakeReviewDataset(
        encodings,
        np.full(len(review_texts), GENUINE_LABEL, dtype=np.int64),
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        pin_memory=use_cuda,
    )
    probabilities = []
    with torch.inference_mode():
        for batch in loader:
            batch.pop("labels")
            batch = {
                name: tensor.to(device, non_blocking=use_cuda)
                for name, tensor in batch.items()
            }
            logits = model(**batch).logits
            batch_probabilities = torch.softmax(logits, dim=1)[:, FAKE_LABEL]
            probabilities.extend(batch_probabilities.cpu().numpy())

    return np.asarray(probabilities)
