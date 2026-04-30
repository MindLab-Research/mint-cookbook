#!/usr/bin/env python3
"""Materialize a local LawBench SFT train artifact from public legal instruction sources."""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
SUPPORTED_LEVELS = {"memory", "understanding", "application"}
PROMPT_KEYS = ("prompt", "prompt_text", "提示")
INSTRUCTION_KEYS = ("instruction", "question", "query", "task", "问题")
CONTEXT_KEYS = ("input", "context", "case", "scenario", "details", "fact", "背景", "案情")
REFERENCE_KEYS = ("reference", "references", "知识", "法条", "依据")
COMPLETION_KEYS = ("completion", "output", "response", "answer", "target", "assistant", "答案带文字", "答案", "解析", "ans")
TASK_KEYS = ("task_name", "task", "category", "dataset", "task_type", "类型")
LEVEL_KEYS = ("cognitive_level", "level", "difficulty")
LANGUAGE_KEYS = ("language", "lang")
OPTION_KEYS = ("A选项", "B选项", "C选项", "D选项", "E选项", "F选项")
REASONING_KEYS = ("think",)
XLSX_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
MEMORY_KEYWORDS = (
    "what is",
    "define",
    "definition",
    "概念",
    "定义",
    "法条内容",
    "条文内容",
    "是什么",
    "列举",
    "依据什么",
)
APPLICATION_KEYWORDS = (
    "case",
    "scenario",
    "consult",
    "advice",
    "remedy",
    "how should",
    "what should",
    "案例",
    "案情",
    "咨询",
    "纠纷",
    "场景",
    "如何处理",
    "怎么办",
    "应当",
    "判决",
)
UNDERSTANDING_KEYWORDS = (
    "interpret",
    "explain",
    "summarize",
    "extract",
    "纠正",
    "解释",
    "理解",
    "摘要",
    "提取",
    "识别",
    "归纳",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Path to a public legal instruction source file (.jsonl/.json/.csv/.tsv/.xlsx). Repeat for multiple sources.",
    )
    parser.add_argument(
        "--output-train",
        default="data/train/full.jsonl",
        help="Output path for the materialized local SFT train artifact.",
    )
    parser.add_argument(
        "--output-meta",
        default="data/train/full.meta.json",
        help="Output path for the provenance and filtering summary.",
    )
    parser.add_argument(
        "--sheet-name",
        default="",
        help="Optional worksheet name when reading .xlsx sources. Defaults to the first sheet.",
    )
    parser.add_argument(
        "--max-rows-per-source",
        type=int,
        default=0,
        help="Optional cap per source after parsing and before materialization. Use 0 for all rows.",
    )
    return parser.parse_args()


class DataError(RuntimeError):
    pass


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def choose_first(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = clean_text(row.get(key))
        if value:
            return value
    return ""


def normalize_completion(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    return cleaned if cleaned[:1].isspace() else f" {cleaned}"


def trim_boxed_answer(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    if re.fullmatch(r"\\boxed\{[^{}]+\}", cleaned):
        return ""
    return cleaned


def render_options(row: dict[str, Any]) -> str:
    parts = []
    for key in OPTION_KEYS:
        value = clean_text(row.get(key))
        if value:
            parts.append(value)
    return "\n".join(parts)


def build_reasoned_completion(row: dict[str, Any]) -> str:
    reasoning = choose_first(row, REASONING_KEYS)
    answer_with_text = clean_text(row.get("答案带文字"))
    concise_answer = clean_text(row.get("答案"))
    parsed_answer = trim_boxed_answer(clean_text(row.get("ans")))
    explanation = clean_text(row.get("解析"))
    final_answer = answer_with_text or concise_answer

    if reasoning and final_answer:
        return normalize_completion(f"<think>\n{reasoning}\n</think>\n<answer>\n{final_answer}\n</answer>")
    if reasoning and parsed_answer and len(parsed_answer) >= 24:
        return normalize_completion(f"<think>\n{reasoning}\n</think>\n<answer>\n{parsed_answer}\n</answer>")

    body_parts: list[str] = []
    if reasoning:
        body_parts.append(reasoning)
    if explanation:
        body_parts.append(explanation)
    if not body_parts and answer_with_text:
        body_parts.append(f"正确答案是：{answer_with_text}")
    elif answer_with_text and answer_with_text not in "\n".join(body_parts):
        body_parts.append(f"最终答案：{answer_with_text}")
    elif concise_answer:
        body_parts.append(f"最终答案：{concise_answer}")

    return normalize_completion("\n\n".join(part for part in body_parts if part))

def build_assistant_text(row: dict[str, Any]) -> str:
    reasoned_completion = build_reasoned_completion(row)
    if reasoned_completion:
        return reasoned_completion
    return normalize_completion(choose_first(row, COMPLETION_KEYS))


def normalize_level(value: str) -> str:
    lowered = clean_text(value).lower()
    mapping = {
        "memory": "memory",
        "remember": "memory",
        "knowledge": "memory",
        "记忆": "memory",
        "understanding": "understanding",
        "understand": "understanding",
        "comprehension": "understanding",
        "理解": "understanding",
        "application": "application",
        "apply": "application",
        "应用": "application",
    }
    return mapping.get(lowered, "")


def infer_cognitive_level(row: dict[str, Any], prompt: str) -> str:
    explicit = normalize_level(choose_first(row, LEVEL_KEYS))
    if explicit in SUPPORTED_LEVELS:
        return explicit
    task_hint = choose_first(row, TASK_KEYS).lower()
    prompt_lower = prompt.lower()
    merged = "\n".join(part for part in (task_hint, prompt_lower) if part)
    if any(keyword in merged for keyword in MEMORY_KEYWORDS):
        return "memory"
    if any(keyword in merged for keyword in APPLICATION_KEYWORDS):
        return "application"
    if any(keyword in merged for keyword in UNDERSTANDING_KEYWORDS):
        return "understanding"
    if any(token in merged for token in ("单选题", "多选题", "推理过程", "下列关于", "以下关于")):
        return "understanding"
    return "understanding"


def infer_task_name(row: dict[str, Any], prompt: str, cognitive_level: str) -> str:
    explicit = choose_first(row, TASK_KEYS)
    if explicit:
        slug = re.sub(r"[^a-z0-9]+", "-", explicit.lower()).strip("-")
        return slug or "legal-qa"
    merged = (prompt + "\n" + json.dumps(row, ensure_ascii=False)).lower()
    if "consult" in merged or "咨询" in merged:
        return "legal-consultation"
    if "definition" in merged or "define" in merged or "定义" in merged or "概念" in merged:
        return "legal-definition"
    if "法条" in merged or "条文" in merged or "statute" in merged or "regulation" in merged:
        return "statute-recitation" if cognitive_level == "memory" else "statute-interpretation"
    if "纠正" in merged or "文书" in merged or "correction" in merged:
        return "document-correction"
    if cognitive_level == "application":
        return "case-application"
    if cognitive_level == "memory":
        return "legal-qa"
    return "statute-interpretation"


def lawbench_mapping_for(task_name: str, cognitive_level: str) -> list[str]:
    mapping = {
        "legal-definition": ["1-2"],
        "legal-qa": ["1-2"],
        "statute-recitation": ["1-1"],
        "statute-interpretation": ["2-4", "2-5"],
        "document-correction": ["2-1"],
        "case-application": ["3-2", "3-6"],
        "legal-consultation": ["3-8"],
    }
    if task_name in mapping:
        return mapping[task_name]
    fallback = {
        "memory": ["1-1", "1-2"],
        "understanding": ["2-4", "2-5", "2-7"],
        "application": ["3-2", "3-6", "3-8"],
    }
    return fallback.get(cognitive_level, [])


def build_prompt(row: dict[str, Any]) -> str:
    if any(clean_text(row.get(key)) for key in OPTION_KEYS) and clean_text(row.get("问题")):
        hint = clean_text(row.get("提示"))
        question = clean_text(row.get("问题"))
        options = render_options(row)
        pieces: list[str] = []
        if hint:
            pieces.append(hint)
        pieces.append(question)
        if options:
            pieces.append(options)
        return "\n".join(part for part in pieces if part)

    prompt = choose_first(row, PROMPT_KEYS)
    if prompt:
        return prompt
    system = clean_text(row.get("system")) or "You are a Chinese legal assistant."
    instruction = choose_first(row, INSTRUCTION_KEYS)
    context = choose_first(row, CONTEXT_KEYS)
    reference = choose_first(row, REFERENCE_KEYS)
    reference_block = f"Reference:\n{reference}" if reference else ""
    pieces = [part for part in (system, instruction, context, reference_block) if part]
    if not pieces:
        return ""
    prompt = "\n".join(pieces)
    answer_markers = ("Answer:", "Answer:", "答案:", "答案：", "assistant:", "Assistant:")
    if not prompt.rstrip().endswith(answer_markers):
        prompt = prompt.rstrip() + "\nAnswer:"
    return prompt

def load_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("data") or payload.get("rows") or payload.get("items")
        if not isinstance(rows, list):
            raise DataError(f"Expected a list-like JSON payload in {path}")
    else:
        raise DataError(f"Expected a list-like JSON payload in {path}")
    return [coerce_row(path, index + 1, item) for index, item in enumerate(rows)]


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Preserve Unicode line separators inside JSON strings by iterating over
    # physical file lines instead of ``splitlines()``.
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            rows.append(coerce_row(path, index, item))
    return rows


def load_delimited_rows(path: Path, delimiter: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return [coerce_row(path, index + 2, row) for index, row in enumerate(reader)]


def column_index_from_ref(cell_ref: str) -> int:
    letters = ""
    for char in cell_ref:
        if char.isalpha():
            letters += char
        else:
            break
    value = 0
    for char in letters.upper():
        value = value * 26 + (ord(char) - ord("A") + 1)
    return max(value - 1, 0)


def load_xlsx_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    path = "xl/sharedStrings.xml"
    if path not in zf.namelist():
        return []
    root = ET.fromstring(zf.read(path))
    values: list[str] = []
    for node in root.findall("main:si", XLSX_NS):
        text = "".join(part.text or "" for part in node.findall(".//main:t", XLSX_NS))
        values.append(text)
    return values


def select_xlsx_sheet(zf: zipfile.ZipFile, sheet_name: str) -> str:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rel_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        relation.attrib["Id"]: relation.attrib["Target"]
        for relation in rel_root.findall("pkgrel:Relationship", XLSX_NS)
    }
    sheets = workbook.findall("main:sheets/main:sheet", XLSX_NS)
    if not sheets:
        raise DataError("Workbook contains no sheets")
    target_sheet = None
    if sheet_name:
        for sheet in sheets:
            if clean_text(sheet.attrib.get("name")) == sheet_name:
                target_sheet = sheet
                break
        if target_sheet is None:
            raise DataError(f"Sheet `{sheet_name}` was not found in workbook")
    else:
        target_sheet = sheets[0]
    relationship_id = target_sheet.attrib.get(f"{{{XLSX_NS['rel']}}}id")
    if not relationship_id or relationship_id not in rel_map:
        raise DataError("Could not resolve worksheet relationship")
    target = rel_map[relationship_id]
    if not target.startswith("xl/"):
        target = "xl/" + target.lstrip("/")
    return target


def load_xlsx_rows(path: Path, sheet_name: str) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        shared_strings = load_xlsx_shared_strings(zf)
        sheet_path = select_xlsx_sheet(zf, sheet_name)
        root = ET.fromstring(zf.read(sheet_path))
    rows: list[list[str]] = []
    for row_node in root.findall(".//main:sheetData/main:row", XLSX_NS):
        cell_map: dict[int, str] = {}
        for cell in row_node.findall("main:c", XLSX_NS):
            cell_type = cell.attrib.get("t")
            cell_ref = cell.attrib.get("r", "A1")
            index = column_index_from_ref(cell_ref)
            value = ""
            if cell_type == "s":
                value_node = cell.find("main:v", XLSX_NS)
                if value_node is not None and value_node.text is not None:
                    shared_index = int(value_node.text)
                    if 0 <= shared_index < len(shared_strings):
                        value = shared_strings[shared_index]
            elif cell_type == "inlineStr":
                value = "".join(part.text or "" for part in cell.findall(".//main:t", XLSX_NS))
            else:
                value_node = cell.find("main:v", XLSX_NS)
                if value_node is not None and value_node.text is not None:
                    value = value_node.text
            cell_map[index] = value
        if not cell_map:
            continue
        width = max(cell_map) + 1
        rows.append([cell_map.get(index, "") for index in range(width)])
    if not rows:
        return []
    headers = [clean_text(header) or f"column_{index}" for index, header in enumerate(rows[0])]
    output_rows: list[dict[str, Any]] = []
    for row_index, values in enumerate(rows[1:], start=2):
        row = {headers[index]: values[index] if index < len(values) else "" for index in range(len(headers))}
        output_rows.append(coerce_row(path, row_index, row))
    return output_rows


def coerce_row(path: Path, index: int, item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise DataError(f"Expected object rows in {path}:{index}")
    return dict(item)


def load_records(path: Path, *, sheet_name: str = "") -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl_rows(path)
    if suffix == ".json":
        return load_json_rows(path)
    if suffix == ".csv":
        return load_delimited_rows(path, ",")
    if suffix == ".tsv":
        return load_delimited_rows(path, "\t")
    if suffix == ".xlsx":
        return load_xlsx_rows(path, sheet_name)
    raise DataError(f"Unsupported source format: {path}")


def resolve_experiment_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists() or cwd_path.parent.exists():
        return cwd_path
    return EXPERIMENT_DIR / path


def portable_path(path_like: str | Path) -> str:
    path = resolve_experiment_path(path_like)
    try:
        return path.resolve().relative_to(EXPERIMENT_DIR).as_posix()
    except ValueError:
        return str(path)


def build_train_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not args.source:
        raise DataError("At least one --source path is required")

    kept_rows: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()
    level_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    drop_counts: Counter[str] = Counter()

    for source_path_str in args.source:
        source_path = resolve_experiment_path(source_path_str)
        source_records = load_records(source_path, sheet_name=args.sheet_name)
        if args.max_rows_per_source > 0:
            source_records = source_records[: args.max_rows_per_source]
        source_name = source_path.stem
        source_counts[f"{source_name}:raw"] += len(source_records)

        for source_index, raw_row in enumerate(source_records, start=1):
            language = choose_first(raw_row, LANGUAGE_KEYS).lower()
            if language and language not in {"zh", "zh-cn", "zh-hans", "cn", "chinese", "en", "english"}:
                drop_counts["unsupported_language"] += 1
                continue
            prompt = build_prompt(raw_row)
            if not prompt:
                drop_counts["missing_prompt"] += 1
                continue
            assistant_text = build_assistant_text(raw_row)
            if not assistant_text:
                drop_counts["missing_assistant_text"] += 1
                continue

            cognitive_level = infer_cognitive_level(raw_row, prompt)
            task_name = infer_task_name(raw_row, prompt, cognitive_level)
            lawbench_mapping = lawbench_mapping_for(task_name, cognitive_level)
            example_id = f"lawbench-sft-{len(kept_rows) + 1:06d}"
            kept_rows.append(
                {
                    "example_id": example_id,
                    "task_name": task_name,
                    "cognitive_level": cognitive_level,
                    "prompt": prompt,
                    "assistant_text": assistant_text,
                    "metadata": {
                        "source_name": source_name,
                        "source_path": portable_path(source_path),
                        "source_row": source_index,
                        "lawbench_mapping": lawbench_mapping,
                    },
                }
            )
            task_counts[task_name] += 1
            level_counts[cognitive_level] += 1
            source_counts[f"{source_name}:kept"] += 1

    meta = {
        "input_sources": [portable_path(item) for item in args.source],
        "output_train": portable_path(args.output_train),
        "row_count": len(kept_rows),
        "task_counts": dict(sorted(task_counts.items())),
        "cognitive_level_counts": dict(sorted(level_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "drop_counts": dict(sorted(drop_counts.items())),
        "notes": [
            "Train rows are materialized from local public-source exports without train-side deduplication or eval-overlap filtering.",
            "The builder only normalizes schema, preserves source provenance, and drops rows that cannot form a prompt/assistant pair.",
            "The maintained train-source line is the public DISC-Law-SFT release plus any explicitly documented supplemental sources.",
        ],
    }
    return kept_rows, meta


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_outputs(output_train: Path, output_meta: Path, train_rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    write_jsonl(output_train, train_rows)
    write_json(output_meta, meta)


def main() -> int:
    args = parse_args()
    train_rows, meta = build_train_rows(args)
    if not train_rows:
        raise SystemExit("No train rows survived materialization. Check the source files and required fields.")
    output_train = resolve_experiment_path(args.output_train)
    output_meta = resolve_experiment_path(args.output_meta)
    write_outputs(output_train, output_meta, train_rows, meta)
    print(f"Wrote {len(train_rows)} rows to {output_train}")
    print(f"Wrote summary to {output_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
