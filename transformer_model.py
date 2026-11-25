import shutil

import pandas as pd
import torch
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    set_seed,
)

from config import FAKE_LABEL, GENUINE_LABEL, PROJECT_DIR
from data_loader import load_datasets
from result_cache import (
    create_cache_signature,
    dataset_fingerprint,
    load_cached_result,
    save_cached_result,
    save_result_artifacts,
)
from transformer_dataset import FakeReviewDataset
from transformer_evaluation import (
    compute_metrics,
    evaluate_transformer_test,
    logits_to_predictions,
    plot_transformer_confusion_matrix,
)


MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
MODEL_DIR = PROJECT_DIR / "final_distilbert_fake_review"
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
NUM_TRAIN_EPOCHS = 3
WEIGHT_DECAY = 0.01
SEED = 42


def train_transformer_model(
    model_name=MODEL_NAME,
    max_length=MAX_LENGTH,
    output_dir=None,
    force_retrain=False,
):
    free_disk_gb = shutil.disk_usage(PROJECT_DIR).free / (1024 ** 3)
    minimum_disk_gb = 4
    if free_disk_gb < minimum_disk_gb:
        raise RuntimeError(
            f"Not enough disk space to download and run DistilBERT. "
            f"Only {free_disk_gb:.2f} GB is available on C:, "
            f"but at least {minimum_disk_gb} GB is recommended. "
            "Free at least 4 GB, especially from the Windows cache or Downloads "
            "folder, then run the command again."
        )

    set_seed(SEED)
    en_train, en_val, en_test = load_datasets()
    data_hash = dataset_fingerprint([
        ("train", en_train),
        ("validation", en_val),
        ("test", en_test),
    ])
    settings = {
        "model_name": model_name,
        "max_length": max_length,
        "num_labels": 2,
        "learning_rate": LEARNING_RATE,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "fp16": torch.cuda.is_available(),
    }
    signature = create_cache_signature(
        data_hash,
        settings,
        [
            PROJECT_DIR / "transformer_model.py",
            PROJECT_DIR / "transformer_dataset.py",
            PROJECT_DIR / "transformer_evaluation.py",
            PROJECT_DIR / "data_loader.py",
            PROJECT_DIR / "config.py",
        ],
    )

    cached_result = None
    if force_retrain:
        print("Force retraining transformer.")
    elif (
        (MODEL_DIR / "config.json").is_file()
        and (MODEL_DIR / "tokenizer_config.json").is_file()
    ):
        cached_result = load_cached_result("transformer", signature)
    elif (PROJECT_DIR / "saved_results" / "transformer" / "cache.joblib").exists():
        print("Saved transformer model files are missing; training again.")

    if cached_result is not None:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_DIR)
        )
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        predictions = cached_result["test_predictions"]
        probabilities = cached_result["test_probabilities"]
        validation_results = cached_result["validation_metrics"]
        test_metrics = cached_result["test_metrics"]

        print("SAVED VALIDATION RESULTS")
        print("------------------------")
        for metric, value in validation_results.items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
        evaluate_transformer_test(
            en_test["label"].astype(int).values,
            predictions,
            probabilities,
        )
        plot_transformer_confusion_matrix(
            en_test["label"].astype(int).values,
            predictions,
        )
        predictions_frame = _transformer_predictions_frame(
            en_test,
            predictions,
            probabilities,
        )
        save_result_artifacts(
            "transformer",
            {"validation": validation_results, "test": test_metrics},
            predictions_frame,
        )
        return {
            "trainer": None,
            "model": model,
            "tokenizer": tokenizer,
            "en_test": en_test,
            "test_predictions": predictions,
            "test_probabilities": probabilities,
            "model_dir": MODEL_DIR,
            "validation_metrics": validation_results,
            "test_metrics": test_metrics,
        }

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={
            GENUINE_LABEL: "Genuine",
            1: "Fake",
        },
        label2id={
            "Genuine": GENUINE_LABEL,
            "Fake": 1,
        },
    )

    train_encodings = tokenizer(
        en_train["review_text"].fillna("").astype(str).tolist(),
        truncation=True,
        max_length=max_length,
    )
    val_encodings = tokenizer(
        en_val["review_text"].fillna("").astype(str).tolist(),
        truncation=True,
        max_length=max_length,
    )
    test_encodings = tokenizer(
        en_test["review_text"].fillna("").astype(str).tolist(),
        truncation=True,
        max_length=max_length,
    )

    train_dataset = FakeReviewDataset(
        train_encodings,
        en_train["label"].astype(int),
    )
    val_dataset = FakeReviewDataset(
        val_encodings,
        en_val["label"].astype(int),
    )
    test_dataset = FakeReviewDataset(
        test_encodings,
        en_test["label"].astype(int),
    )

    TrainingArguments = transformers.TrainingArguments
    checkpoint_dir = output_dir or PROJECT_DIR / "distilbert_fake_review"
    try:
        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            weight_decay=WEIGHT_DECAY,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=settings["fp16"],
            logging_steps=100,
            save_total_limit=1,
            report_to="none",
            seed=SEED,
        )
    except TypeError:
        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            weight_decay=WEIGHT_DECAY,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=settings["fp16"],
            logging_steps=100,
            save_total_limit=1,
            report_to="none",
            seed=SEED,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Starting DistilBERT fine-tuning...")
    trainer.train()

    print("BEST VALIDATION RESULTS")
    print("------------------------")
    val_results = trainer.evaluate()
    for metric, value in val_results.items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}")

    test_output = trainer.predict(test_dataset)
    predictions, probabilities = logits_to_predictions(test_output.predictions)
    y_test = en_test["label"].astype(int).values
    test_metrics = evaluate_transformer_test(
        y_test,
        predictions,
        probabilities,
    )
    plot_transformer_confusion_matrix(y_test, predictions)

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print("Fine-tuned DistilBERT saved successfully!")

    predictions_frame = _transformer_predictions_frame(
        en_test,
        predictions,
        probabilities,
    )
    saved_results = {
        "test_predictions": predictions,
        "test_probabilities": probabilities,
        "validation_metrics": val_results,
        "test_metrics": test_metrics,
    }
    save_cached_result("transformer", signature, saved_results)
    save_result_artifacts(
        "transformer",
        {"validation": val_results, "test": test_metrics},
        predictions_frame,
    )

    return {
        "trainer": trainer,
        "model": trainer.model,
        "tokenizer": tokenizer,
        "en_test": en_test,
        "test_predictions": predictions,
        "test_probabilities": probabilities,
        "model_dir": MODEL_DIR,
        "validation_metrics": val_results,
        "test_metrics": test_metrics,
    }


def _transformer_predictions_frame(en_test, predictions, probabilities):
    return pd.DataFrame({
        "review_text": en_test["review_text"].fillna("").astype(str),
        "true_label": en_test["label"].astype(int).map({
            GENUINE_LABEL: "Genuine",
            FAKE_LABEL: "Fake",
        }).values,
        "predicted_label": pd.Series(predictions).map({
            GENUINE_LABEL: "Genuine",
            FAKE_LABEL: "Fake",
        }).values,
        "fake_probability": probabilities,
    })
