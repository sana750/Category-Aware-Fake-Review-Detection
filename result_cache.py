import hashlib
import json
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import PROJECT_DIR


RESULTS_DIR = PROJECT_DIR / "saved_results"
CACHE_FORMAT_VERSION = 1


def dataset_fingerprint(splits):
    digest = hashlib.sha256()
    for split_name, frame in splits:
        digest.update(split_name.encode("utf-8"))
        digest.update(json.dumps(
            {
                "columns": list(frame.columns),
                "dtypes": [str(dtype) for dtype in frame.dtypes],
            },
            sort_keys=True,
        ).encode("utf-8"))
        digest.update(pd.util.hash_pandas_object(
            frame,
            index=True,
        ).values.tobytes())
    return digest.hexdigest()


def create_cache_signature(data_hash, settings, source_files):
    source_hashes = {}
    for source_file in source_files:
        path = Path(source_file)
        source_hashes[path.name] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

    return {
        "format_version": CACHE_FORMAT_VERSION,
        "data_hash": data_hash,
        "settings": settings,
        "source_hashes": source_hashes,
    }


def load_cached_result(name, signature):
    cache_path = RESULTS_DIR / name / "cache.joblib"
    if not cache_path.exists():
        return None

    cached = joblib.load(cache_path)
    cached_signature = cached.get("signature", {})
    if not _signatures_match(cached_signature, signature):
        if cached_signature.get("data_hash") != signature.get("data_hash"):
            reason = "the dataset changed"
        elif cached_signature.get("settings") != signature.get("settings"):
            reason = "the model settings changed"
        else:
            reason = "the model implementation or cache format changed"
        print(f"Saved {name} results do not match: {reason}.")
        return None

    print(f"Reusing saved {name} results.")
    return cached["result"]


def _signatures_match(cached_signature, expected_signature):
    cached_normalized = dict(cached_signature)
    expected_normalized = dict(expected_signature)
    cached_sources = dict(cached_normalized.get("source_hashes", {}))
    expected_sources = dict(expected_normalized.get("source_hashes", {}))
    cached_sources.pop("main.py", None)
    expected_sources.pop("main.py", None)
    cached_normalized["source_hashes"] = cached_sources
    expected_normalized["source_hashes"] = expected_sources
    return cached_normalized == expected_normalized


def save_cached_result(name, signature, result):
    output_dir = RESULTS_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "cache.joblib"

    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_dir,
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        joblib.dump(
            {"signature": signature, "result": result},
            temporary_path,
        )
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    write_json(output_dir / "run_metadata.json", signature)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, default=_json_default)
    os.replace(temporary_path, path)


def save_result_artifacts(name, metrics, predictions):
    output_dir = RESULTS_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "metrics.json", metrics)
    prediction_path = output_dir / "test_predictions.csv"
    temporary_path = prediction_path.with_name(
        prediction_path.name + ".tmp"
    )
    predictions.to_csv(temporary_path, index=False)
    os.replace(temporary_path, prediction_path)


def save_dataframe(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")
