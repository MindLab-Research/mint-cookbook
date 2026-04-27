#!/usr/bin/env python3
"""Rebuild the public FinGPT sentiment benchmark splits from official sources."""

from __future__ import annotations

import json
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import datasets
from datasets import Dataset, DatasetDict, load_dataset

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "benchmarks" / "sentiment"
FPB_ZIP_URL = (
    "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/"
    "data/FinancialPhraseBank-v1.0.zip"
)


def write_split(dataset_name: str, split_name: str, rows: list[dict[str, object]]) -> None:
    path = OUT_ROOT / dataset_name / f"{split_name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for index, row in enumerate(rows, start=1):
            payload = dict(row)
            payload.setdefault("id", f"{dataset_name}-{split_name}-{index:05d}")
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"wrote {path} rows={len(rows)}")


def materialize(dataset_name: str, dataset: DatasetDict) -> None:
    for split_name in ("train", "test"):
        write_split(dataset_name, split_name, list(dataset[split_name]))


def build_fpb() -> DatasetDict:
    raw_zip = urlopen(FPB_ZIP_URL).read()
    archive = zipfile.ZipFile(io.BytesIO(raw_zip))
    rows: list[dict[str, str]] = []
    for raw_line in archive.read("FinancialPhraseBank-v1.0/Sentences_50Agree.txt").decode("latin1").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        input_text, output_text = line.rsplit("@", 1)
        rows.append(
            {
                "input": input_text.strip(),
                "output": output_text.strip(),
                "instruction": (
                    "What is the sentiment of this news? Please choose an answer from "
                    "{negative/neutral/positive}."
                ),
            }
        )
    hf_dataset = Dataset.from_list(rows)
    return hf_dataset.train_test_split(seed=42)


def make_fiqa_label(score: float) -> str:
    if score < -0.1:
        return "negative"
    if score < 0.1:
        return "neutral"
    return "positive"


def make_fiqa_instruction(source_format: str) -> str:
    if source_format == "post":
        return "What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}."
    return "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."


def build_fiqa_sa() -> DatasetDict:
    raw = load_dataset("pauri32/fiqa-2018")
    dataset = datasets.concatenate_datasets([raw["train"], raw["validation"], raw["test"]]).to_pandas()
    dataset["output"] = dataset["sentiment_score"].map(make_fiqa_label)
    dataset["instruction"] = dataset["format"].map(make_fiqa_instruction)
    dataset = dataset[["sentence", "output", "instruction"]]
    dataset.columns = ["input", "output", "instruction"]
    hf_dataset = Dataset.from_pandas(dataset, preserve_index=False)
    return hf_dataset.train_test_split(test_size=0.226, seed=42)


def build_tfns() -> DatasetDict:
    label_map = {0: "negative", 1: "positive", 2: "neutral"}
    raw = load_dataset("zeroshot/twitter-financial-news-sentiment")
    train_frame = raw["train"].to_pandas()
    train_frame["label"] = train_frame["label"].map(label_map)
    train_frame["instruction"] = (
        "What is the sentiment of this tweet? Please choose an answer from "
        "{negative/neutral/positive}."
    )
    train_frame.columns = ["input", "output", "instruction"]

    test_split_name = "validation" if "validation" in raw else "test"
    test_frame = raw[test_split_name].to_pandas()
    test_frame["label"] = test_frame["label"].map(label_map)
    test_frame["instruction"] = (
        "What is the sentiment of this tweet? Please choose an answer from "
        "{negative/neutral/positive}."
    )
    test_frame.columns = ["input", "output", "instruction"]

    return DatasetDict(
        train=Dataset.from_pandas(train_frame, preserve_index=False),
        test=Dataset.from_pandas(test_frame, preserve_index=False),
    )


def build_nwgi() -> DatasetDict:
    raw = load_dataset("oliverwang15/news_with_gpt_instructions")

    def convert(split_name: str) -> Dataset:
        frame = raw[split_name].to_pandas()
        frame["output"] = frame["label"]
        frame["input"] = frame["news"]
        frame["instruction"] = (
            "What is the sentiment of this news? Please choose an answer from "
            "{strong negative/moderately negative/mildly negative/neutral/"
            "mildly positive/moderately positive/strong positive}, then provide "
            "some short reasons."
        )
        return Dataset.from_pandas(frame[["input", "output", "instruction"]], preserve_index=False)

    return DatasetDict(train=convert("train"), test=convert("test"))


def main() -> int:
    materialize("fpb", build_fpb())
    materialize("fiqa-sa", build_fiqa_sa())
    materialize("tfns", build_tfns())
    materialize("nwgi", build_nwgi())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
