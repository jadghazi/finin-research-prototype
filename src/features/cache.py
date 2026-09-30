"""Cache frozen BGE text vectors and financial TinyBERT sentiment scores."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

from src.data.examples import MARKET_DESCRIPTION
from src.data.nifty import sha256_file


TEXT_NAME = "BAAI/bge-small-en-v1.5"
TEXT_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
TEXT_WIDTH = 384
SENTIMENT_NAME = "mikeysharma/finance-sentiment-analysis"
SENTIMENT_REVISION = "4353ecd593de607cbb64147c8e2d21b82d842014"
MAX_TOKENS = 128


def selected_texts(root: Path) -> list[tuple[str, str]]:
    texts = {}
    for split in ("train", "valid", "test"):
        with (root / "data" / "processed" / f"{split}.jsonl").open(encoding="utf-8") as source:
            for line in source:
                for headline in json.loads(line)["headlines"]:
                    old = texts.setdefault(headline["id"], headline["text"])
                    if old != headline["text"]:
                        raise ValueError(f"Headline ID collision: {headline['id']}")
    return sorted(texts.items())


def source_signature(root: Path, items: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for split in ("train", "valid", "test"):
        digest.update(sha256_file(root / "data" / "processed" / f"{split}.jsonl").encode("ascii"))
    for headline_id, text in items:
        digest.update(headline_id.encode("ascii"))
        digest.update(text.encode("utf-8"))
    digest.update(MARKET_DESCRIPTION.encode("utf-8"))
    digest.update(TEXT_REVISION.encode("ascii"))
    digest.update(SENTIMENT_REVISION.encode("ascii"))
    return digest.hexdigest()


def device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(kind: str, device: str):
    if kind == "text":
        model = AutoModel.from_pretrained(TEXT_NAME, revision=TEXT_REVISION)
        tokenizer = AutoTokenizer.from_pretrained(TEXT_NAME, revision=TEXT_REVISION, use_fast=True)
        if model.config.hidden_size != TEXT_WIDTH:
            raise ValueError("Unexpected BGE hidden width")
        model.eval().to(device)
        return model, tokenizer
    if kind == "sentiment":
        model_file = hf_hub_download(SENTIMENT_NAME, "model.onnx", revision=SENTIMENT_REVISION)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(model_file, sess_options=options, providers=["CPUExecutionProvider"])
        tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_NAME, revision=SENTIMENT_REVISION, use_fast=True)
        return session, tokenizer
    raise ValueError(f"Unknown encoder: {kind}")


def encode_batch(model, tokenizer, texts: list[str], kind: str, device: str) -> tuple[np.ndarray, int]:
    full = tokenizer(texts, add_special_tokens=True, truncation=False)
    truncated_count = sum(len(tokens) > MAX_TOKENS for tokens in full["input_ids"])
    if kind == "text":
        tokens = tokenizer(texts, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.inference_mode():
            output = model(**tokens).last_hidden_state[:, 0, :]
        return output.detach().cpu().numpy().astype(np.float32), truncated_count

    tokens = tokenizer(texts, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="np")
    inputs = {}
    for parameter in model.get_inputs():
        if parameter.name in tokens:
            inputs[parameter.name] = tokens[parameter.name].astype(np.int64)
        elif parameter.name == "token_type_ids":
            inputs[parameter.name] = np.zeros_like(tokens["input_ids"], dtype=np.int64)
        else:
            raise ValueError(f"Unrecognized sentiment ONNX input: {parameter.name}")
    logits = np.asarray(model.run(None, inputs)[0], dtype=np.float32)
    if logits.shape != (len(texts), 3):
        raise ValueError(f"Unexpected sentiment output shape: {logits.shape}")
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    # The published model card lists Negative, Neutral, Positive in this order.
    # Explicit control examples are checked in the 100-headline benchmark.
    return probabilities[:, [2, 1, 0]].astype(np.float32), truncated_count


def check_sentiment_controls(model, tokenizer, device: str) -> dict:
    controls = {
        "positive": "Company profits rise sharply after strong earnings beat expectations.",
        "negative": "Company reports severe losses and cuts jobs after a sharp sales decline.",
        "neutral": "Company announces the date of its next shareholder meeting.",
    }
    values, _ = encode_batch(model, tokenizer, list(controls.values()), "sentiment", device)
    expected_index = {"positive": 0, "neutral": 1, "negative": 2}
    results = {}
    for label, probabilities in zip(controls, values):
        results[label] = probabilities.tolist()
        if label in ("positive", "negative") and int(probabilities.argmax()) != expected_index[label]:
            raise ValueError(f"Sentiment control disagrees with model-card label order: {label}: {probabilities}")
    return results


def benchmark(root: Path, batch_size: int = 8) -> dict:
    torch.set_num_threads(4)
    items = selected_texts(root)
    sample = [text for _, text in items[:100]]
    if not sample:
        raise ValueError("No selected headlines")
    device = device_name()
    results = {"device": device, "selected_unique_headlines": len(items), "sample_size": len(sample)}
    for kind in ("text", "sentiment"):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        model, tokenizer = load_encoder(kind, device)
        load_seconds = time.monotonic() - start
        truncated = 0
        widths = set()
        for offset in range(0, len(sample), batch_size):
            values, count = encode_batch(model, tokenizer, sample[offset:offset + batch_size], kind, device)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite {kind} features")
            widths.add(values.shape[1])
            truncated += count
        if widths != ({TEXT_WIDTH} if kind == "text" else {3}):
            raise ValueError(f"Unexpected {kind} feature width: {widths}")
        result = {
            "load_seconds": round(load_seconds, 2),
            "inference_seconds": round(time.monotonic() - start - load_seconds, 2),
            "output_width": next(iter(widths)),
            "truncated_texts": truncated,
            "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1) if device == "cuda" and kind == "text" else None,
        }
        if kind == "sentiment":
            result["control_examples"] = check_sentiment_controls(model, tokenizer, device)
        results[kind] = result
        del model, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"{kind}: {result}", flush=True)
    return results


def cache_one(root: Path, kind: str, items: list[tuple[str, str]], batch_size: int, device: str, signature: str) -> dict:
    directory = root / "artifacts" / "features"
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / f"{kind}_state.json"
    output_path = directory / f"{kind}.npy"
    width = TEXT_WIDTH if kind == "text" else 3
    expected = {"kind": kind, "signature": signature, "count": len(items), "width": width, "max_tokens": MAX_TOKENS}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if any(state.get(key) != value for key, value in expected.items()) or not output_path.exists():
            raise ValueError(f"Existing {kind} cache does not match current data/configuration")
        array = np.lib.format.open_memmap(output_path, mode="r+")
        if array.shape != (len(items), width):
            raise ValueError(f"Existing {kind} cache has the wrong shape")
    else:
        state = {**expected, "next": 0, "truncated_texts": 0, "elapsed_seconds": 0.0}
        array = np.lib.format.open_memmap(output_path, mode="w+", dtype="float32", shape=(len(items), width))
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    if state["next"] == len(items):
        if state.get("sha256") and sha256_file(output_path) != state["sha256"]:
            raise ValueError(f"Completed {kind} cache failed checksum verification")
        if not state.get("sha256"):
            state["sha256"] = sha256_file(output_path)
            state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        print(f"{kind}: complete cache already present", flush=True)
        return state
    model, tokenizer = load_encoder(kind, device)
    start = time.monotonic()
    for offset in range(state["next"], len(items), batch_size):
        values, truncated = encode_batch(model, tokenizer, [text for _, text in items[offset:offset + batch_size]], kind, device)
        if values.shape != (min(batch_size, len(items) - offset), width) or not np.isfinite(values).all():
            raise ValueError(f"Bad {kind} outputs at offset {offset}")
        if kind == "sentiment" and not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"Sentiment probability check failed at {offset}")
        array[offset:offset + len(values)] = values
        array.flush()
        state["next"] = offset + len(values)
        state["truncated_texts"] += truncated
        state["elapsed_seconds"] = round(state["elapsed_seconds"] + time.monotonic() - start, 2)
        start = time.monotonic()
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        if state["next"] % 512 < batch_size or state["next"] == len(items):
            print(f"{kind}: {state['next']:,}/{len(items):,} unique headlines", flush=True)
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    state["sha256"] = sha256_file(output_path)
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return state


def cache_all(root: Path, batch_size: int = 8) -> dict:
    torch.set_num_threads(4)
    items = selected_texts(root)
    signature = source_signature(root, items)
    device = device_name()
    directory = root / "artifacts" / "features"
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / "headline_index.json"
    ids = [headline_id for headline_id, _ in items]
    if index_path.exists():
        if json.loads(index_path.read_text(encoding="utf-8")) != ids:
            raise ValueError("Headline index differs from prepared data")
    else:
        index_path.write_text(json.dumps(ids) + "\n", encoding="utf-8")
    states = {}
    for kind in ("text", "sentiment"):
        states[kind] = cache_one(root, kind, items, batch_size, device, signature)
    description_path = directory / "market_description.npy"
    if not description_path.exists():
        model, tokenizer = load_encoder("text", device)
        vector, _ = encode_batch(model, tokenizer, [MARKET_DESCRIPTION], "text", device)
        np.save(description_path, vector[0])
        del model, tokenizer
    manifest = {
        "signature": signature,
        "headline_count": len(items),
        "text_model": TEXT_NAME,
        "text_revision": TEXT_REVISION,
        "text_width": TEXT_WIDTH,
        "sentiment_model": SENTIMENT_NAME,
        "sentiment_revision": SENTIMENT_REVISION,
        "sentiment_output_order": ["positive", "neutral", "negative"],
        "max_tokens": MAX_TOKENS,
        "device": device,
        "states": states,
        "headline_index_sha256": sha256_file(index_path),
        "market_description_sha256": sha256_file(description_path),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
