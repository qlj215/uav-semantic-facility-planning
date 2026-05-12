#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidence_fusion import DetectionEvidence, FacilityPrediction, fuse_facility_and_evidence
from semantic_grid_planner import SemanticRisk, astar, build_cost_grid, path_cost, render_ascii


def main() -> None:
    width, height = 40, 25
    start = (2, 20)
    goal = (36, 4)

    facility = FacilityPrediction(
        class_name="airport",
        confidence=0.76,
        center_xy=(20, 12),
    )
    detections = [
        DetectionEvidence("aircraft", 0.91, (18, 10, 21, 13)),
        DetectionEvidence("runway", 0.84, (14, 12, 28, 14)),
        DetectionEvidence("vehicle", 0.58, (23, 15, 24, 16)),
    ]

    fusion = fuse_facility_and_evidence(facility, detections)

    risk_cost = {"low": 3.0, "medium": 8.0, "high": 20.0}[fusion.risk_level]
    risks = [
        SemanticRisk(
            center_xy=fusion.center_xy,
            radius=4,
            cost=risk_cost,
            label=fusion.class_name,
        )
    ]
    obstacles = [(12, y) for y in range(3, 18)] + [(28, y) for y in range(8, 23)]

    normal_grid = build_cost_grid(width, height, risks=[], obstacles=obstacles)
    semantic_grid = build_cost_grid(width, height, risks=risks, obstacles=obstacles)

    normal_path = astar(normal_grid, start, goal)
    semantic_path = astar(semantic_grid, start, goal)
    if normal_path is None or semantic_path is None:
        raise RuntimeError("No path found in synthetic map.")

    result = {
        "facility_prediction": facility.__dict__,
        "detections": [d.__dict__ for d in detections],
        "fusion": {
            "class_name": fusion.class_name,
            "score": round(fusion.score, 4),
            "risk_level": fusion.risk_level,
            "needs_review": fusion.needs_review,
            "evidence_summary": fusion.evidence_summary,
            "center_xy": fusion.center_xy,
        },
        "normal_path": {
            "length": len(normal_path),
            "cost": round(path_cost(normal_path, normal_grid), 2),
        },
        "semantic_path": {
            "length": len(semantic_path),
            "cost": round(path_cost(semantic_path, semantic_grid), 2),
        },
        "ascii_map": render_ascii(width, height, semantic_path, start, goal, risks, obstacles),
    }

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "synthetic_demo_result.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(result["fusion"], indent=2, ensure_ascii=False))
    print()
    print(result["ascii_map"])
    print()
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

