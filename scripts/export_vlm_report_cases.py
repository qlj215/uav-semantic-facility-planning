#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "fmow_key_subset_imagefolder"
DEFAULT_CASE_ROOT = ROOT / "outputs" / "vlm_hpe" / "case_manifests"
DEFAULT_RESULT_ROOT = ROOT / "outputs" / "vlm_hpe"
DEFAULT_OUTPUT = ROOT / "outputs" / "vlm_hpe" / "report_cases"


CASE_GROUPS = [
    ("hpe_v3_fixes_flat", "hpe_v3_fixes_flat.csv", "fixes_limit", True, "HPE-v3 fixes flat errors"),
    ("hpe_v3_abstain", "hpe_v3_abstain.csv", "abstain_limit", True, "HPE-v3 abstain/review cases"),
    (
        "shipyard_to_port_failures",
        "shipyard_to_port_failures.csv",
        "failure_limit",
        False,
        "shipyard predicted as port",
    ),
    (
        "runway_to_airport_failures",
        "runway_to_airport_failures.csv",
        "failure_limit",
        False,
        "runway predicted as airport",
    ),
    (
        "storage_tank_to_oil_gas_failures",
        "storage_tank_to_oil_gas_failures.csv",
        "failure_limit",
        False,
        "storage_tank predicted as oil_or_gas_facility",
    ),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Case manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prediction_map(path: Path) -> dict[str, dict[str, object]]:
    return {str(row["image"]): row for row in read_jsonl(path)}


def pred_label(row: dict[str, object] | None) -> str:
    if row is None:
        return "NO_OUTPUT"
    return str(row.get("pred_label") or "NO_OUTPUT")


def is_correct(row: dict[str, object] | None) -> bool:
    return bool(row and row.get("correct") is True)


def hpe_evidence(row: dict[str, object] | None) -> str:
    if not row:
        return ""
    parsed = row.get("parsed")
    if not isinstance(parsed, dict):
        return ""
    evidence = parsed.get("evidence") or parsed.get("visual_evidence") or ""
    if isinstance(evidence, list):
        return "; ".join(str(item) for item in evidence)
    return str(evidence)


def hpe_need_review(row: dict[str, object] | None) -> str:
    if not row:
        return ""
    parsed = row.get("parsed")
    if not isinstance(parsed, dict):
        return ""
    return str(bool(parsed.get("need_review", False)))


def common_case_row(
    image: str,
    flat: dict[str, object] | None,
    hpe_v2: dict[str, object] | None,
    hpe_v3: dict[str, object],
) -> dict[str, str]:
    return {
        "image": image,
        "image_path": str(hpe_v3.get("image_path") or (flat or {}).get("image_path") or ""),
        "true_label": str(hpe_v3.get("true_label") or (flat or {}).get("true_label") or ""),
        "flat_pred": pred_label(flat),
        "hpe_v2_pred": pred_label(hpe_v2),
        "hpe_v3_pred": pred_label(hpe_v3),
        "flat_correct": str(is_correct(flat)),
        "hpe_v2_correct": str(is_correct(hpe_v2)),
        "hpe_v3_correct": str(is_correct(hpe_v3)),
        "hpe_v3_evidence": hpe_evidence(hpe_v3),
        "hpe_v3_need_review": hpe_need_review(hpe_v3),
        "hpe_v3_raw_output": str(hpe_v3.get("raw_output") or ""),
    }


def rows_from_predictions(result_root: Path, flat_run: str, hpe_v2_run: str, hpe_v3_run: str) -> dict[str, list[dict[str, str]]]:
    flat_map = prediction_map(result_root / flat_run / "predictions.jsonl")
    hpe_v2_map = prediction_map(result_root / hpe_v2_run / "predictions.jsonl")
    hpe_v3_map = prediction_map(result_root / hpe_v3_run / "predictions.jsonl")

    groups = {group: [] for group, *_ in CASE_GROUPS}
    for image, hpe_v3 in hpe_v3_map.items():
        flat = flat_map.get(image)
        hpe_v2 = hpe_v2_map.get(image)
        row = common_case_row(image, flat, hpe_v2, hpe_v3)

        true_label = row["true_label"]
        hpe_v3_pred = row["hpe_v3_pred"]
        if not is_correct(flat) and is_correct(hpe_v3):
            groups["hpe_v3_fixes_flat"].append(row)
        if hpe_v3_pred == "other_uncertain":
            groups["hpe_v3_abstain"].append(row)
        if true_label == "shipyard" and hpe_v3_pred == "port":
            groups["shipyard_to_port_failures"].append(row)
        if true_label == "runway" and hpe_v3_pred == "airport":
            groups["runway_to_airport_failures"].append(row)
        if true_label == "storage_tank" and hpe_v3_pred == "oil_or_gas_facility":
            groups["storage_tank_to_oil_gas_failures"].append(row)
    return groups


def rows_from_manifests(case_root: Path) -> dict[str, list[dict[str, str]]]:
    return {group: read_csv(case_root / csv_name) for group, csv_name, *_ in CASE_GROUPS}


def case_manifest_files_exist(case_root: Path) -> bool:
    return all((case_root / csv_name).exists() for _, csv_name, *_ in CASE_GROUPS)


def select_cases(rows: list[dict[str, str]], limit: int, diverse_labels: bool) -> list[dict[str, str]]:
    if limit <= 0:
        return []
    if not diverse_labels:
        return rows[:limit]

    selected: list[dict[str, str]] = []
    seen_labels: set[str] = set()
    for row in rows:
        label = row.get("true_label", "")
        if label in seen_labels:
            continue
        selected.append(row)
        seen_labels.add(label)
        if len(selected) == limit:
            return selected

    for row in rows:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) == limit:
            break
    return selected


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")


def copy_case_image(row: dict[str, str], data_root: Path, group_dir: Path, index: int) -> tuple[str, str]:
    source_image = row["image"]
    src = data_root / source_image
    filename = (
        f"{index:02d}__true-{safe_name(row.get('true_label', 'na'))}__"
        f"flat-{safe_name(row.get('flat_pred', 'na'))}__"
        f"hpe3-{safe_name(row.get('hpe_v3_pred', 'na'))}__{Path(source_image).name}"
    )
    dst = group_dir / filename
    if not src.exists():
        return "", str(src)
    shutil.copy2(src, dst)
    return dst.name, ""


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path) -> None:
    path.write_text(
        "# VLM/HPE report cases\n\n"
        "This directory contains selected qualitative cases for the VLM + HPE report section.\n\n"
        "- `hpe_v3_fixes_flat/`: HPE-v3 corrects flat prompt errors.\n"
        "- `hpe_v3_abstain/`: HPE-v3 outputs `other_uncertain` for review.\n"
        "- `shipyard_to_port_failures/`: shipyard predicted as port.\n"
        "- `runway_to_airport_failures/`: runway predicted as airport.\n"
        "- `storage_tank_to_oil_gas_failures/`: storage_tank predicted as oil_or_gas_facility.\n\n"
        "Use `selected_report_cases.csv` for labels, predictions, and HPE evidence. "
        "If `missing_report_cases.csv` exists, rerun with the correct `--data-root`.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export selected VLM/HPE qualitative cases for reporting.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source",
        choices=["auto", "case-manifests", "predictions"],
        default="auto",
        help="Read prepared case manifests, or build cases from prediction jsonl files.",
    )
    parser.add_argument("--flat-run", default="qwen2_5_vl_7b_flat")
    parser.add_argument("--hpe-v2-run", default="qwen2_5_vl_7b_hpe_v2")
    parser.add_argument("--hpe-v3-run", default="qwen2_5_vl_7b_hpe_v3")
    parser.add_argument("--fixes-limit", type=int, default=4)
    parser.add_argument("--abstain-limit", type=int, default=4)
    parser.add_argument("--failure-limit", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {args.output}. Use --overwrite to replace it.")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    selected_rows: list[dict[str, str]] = []
    missing_rows: list[dict[str, str]] = []
    copied = 0

    use_manifests = args.source == "case-manifests" or (
        args.source == "auto" and case_manifest_files_exist(args.case_root)
    )
    if use_manifests:
        source_rows = rows_from_manifests(args.case_root)
        source_name = "case-manifests"
    else:
        source_rows = rows_from_predictions(args.result_root, args.flat_run, args.hpe_v2_run, args.hpe_v3_run)
        source_name = "predictions"

    for group, csv_name, limit_arg, diverse, note in CASE_GROUPS:
        rows = source_rows[group]
        limit = getattr(args, limit_arg)
        group_rows = select_cases(rows, limit, diverse)
        group_dir = args.output / group
        group_dir.mkdir(parents=True, exist_ok=True)

        for index, row in enumerate(group_rows, 1):
            copied_name, missing_path = copy_case_image(row, args.data_root, group_dir, index)
            if copied_name:
                copied += 1
            out_row = {
                "group": group,
                "note": note,
                "case_image": f"{group}/{copied_name}" if copied_name else "",
                "source_image": row.get("image", ""),
                "missing_path": missing_path,
                "true_label": row.get("true_label", ""),
                "flat_pred": row.get("flat_pred", ""),
                "hpe_v2_pred": row.get("hpe_v2_pred", ""),
                "hpe_v3_pred": row.get("hpe_v3_pred", ""),
                "hpe_v3_need_review": row.get("hpe_v3_need_review", ""),
                "hpe_v3_evidence": row.get("hpe_v3_evidence", ""),
            }
            selected_rows.append(out_row)
            if missing_path:
                missing_rows.append(out_row)

    write_csv(args.output / "selected_report_cases.csv", selected_rows)
    write_csv(args.output / "missing_report_cases.csv", missing_rows)
    write_readme(args.output / "README.md")

    print(f"selected={len(selected_rows)}")
    print(f"copied={copied}")
    print(f"missing={len(missing_rows)}")
    print(f"source={source_name}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
