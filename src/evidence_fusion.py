from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class FacilityPrediction:
    class_name: str
    confidence: float
    center_xy: Tuple[int, int]


@dataclass(frozen=True)
class DetectionEvidence:
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]


@dataclass(frozen=True)
class FusionResult:
    class_name: str
    score: float
    risk_level: str
    needs_review: bool
    evidence_summary: List[str]
    center_xy: Tuple[int, int]


DEFAULT_EVIDENCE_RULES: Dict[str, Dict[str, float]] = {
    "airport": {
        "aircraft": 0.30,
        "airport_region": 0.25,
        "runway": 0.35,
        "airport_hangar": 0.20,
        "vehicle": 0.05,
    },
    "runway": {"runway": 0.50, "aircraft": 0.20, "vehicle": 0.05},
    "port": {"ship": 0.35, "harbor": 0.30, "storage_tank": 0.10, "vehicle": 0.05},
    "storage_tank": {"storage_tank": 0.45, "vehicle": 0.05},
    "oil_or_gas_facility": {
        "storage_tank": 0.30,
        "industrial_building": 0.20,
        "industrial_chimney": 0.15,
        "vehicle": 0.05,
    },
    "factory_or_powerplant": {
        "industrial_chimney": 0.25,
        "storage_tank": 0.10,
        "vehicle": 0.05,
    },
    "military_facility": {"aircraft": 0.20, "vehicle": 0.15, "runway": 0.15, "storage_tank": 0.10},
}


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def fuse_facility_and_evidence(
    facility: FacilityPrediction,
    detections: Iterable[DetectionEvidence],
    rules: Dict[str, Dict[str, float]] | None = None,
    base_weight: float = 0.70,
    evidence_weight: float = 0.30,
) -> FusionResult:
    rules = rules or DEFAULT_EVIDENCE_RULES
    class_rules = rules.get(facility.class_name, {})

    evidence_bonus = 0.0
    evidence_summary: List[str] = []

    for det in detections:
        rule_weight = class_rules.get(det.class_name, 0.0)
        if rule_weight <= 0:
            continue
        contribution = rule_weight * clamp01(det.confidence)
        evidence_bonus += contribution
        evidence_summary.append(
            f"{det.class_name}: conf={det.confidence:.2f}, contribution={contribution:.2f}"
        )

    evidence_bonus = min(evidence_bonus, evidence_weight)
    score = clamp01(base_weight * clamp01(facility.confidence) + evidence_bonus)

    if score >= 0.75:
        risk_level = "high"
        needs_review = False
    elif score >= 0.45:
        risk_level = "medium"
        needs_review = True
    else:
        risk_level = "low"
        needs_review = True

    if not evidence_summary:
        evidence_summary.append("no supporting detection evidence")

    return FusionResult(
        class_name=facility.class_name,
        score=score,
        risk_level=risk_level,
        needs_review=needs_review,
        evidence_summary=evidence_summary,
        center_xy=facility.center_xy,
    )
