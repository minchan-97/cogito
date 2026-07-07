"""
divergence_trace.py — 근거 사슬 위 '분기점' 탐지 + 벡터 전량 기록.

민찬기님 핵심 발상:
  여러 LLM(또는 여러 생성)이 근거 사슬을 따라가다가
  '정확히 어느 마디에서 다른 벡터로 튀는가' = 분기점.
  최종 불일치율(한 숫자)이 아니라, 사슬 위 '어디서 갈라지나'를 본다.

  - 앵커 근처 분기 → 근본 전제가 다름 (심각)
  - 말단 분기 → 표현만 다름 (사소)

기록:
  일치/불일치/환각 문장의 벡터를 모두 남긴다.
  나중에 이 벡터 기록에서 핵심 패턴을 짚기 위함.

정직한 범위:
  - 임베딩은 주입식(embed_fn). 없으면 해시 기반 폴백(구조 검증용).
  - '표현 차이 vs 근거 차이' 구분은 임계값 튜닝 필요 — 기록을 보고 정한다.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional
import numpy as np
import pickle
import time
import hashlib


# ─────────────────────────────────────────────
# 임베딩 (주입식, 없으면 해시 폴백)
# ─────────────────────────────────────────────
def _hash_embed(text: str, dim: int = 64) -> np.ndarray:
    """해시 기반 폴백 임베딩(의미 없음, 구조 검증용)."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    v = rng.standard_normal(dim)
    return v / (np.linalg.norm(v) + 1e-12)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


# ─────────────────────────────────────────────
# 데이터 구조
# ─────────────────────────────────────────────
@dataclass
class ChainNode:
    """근거 사슬의 한 마디."""
    depth: int          # 0=답변, 클수록 앵커에 가까움(소급 깊이)
    statement: str      # 이 마디의 주장/근거 문장
    is_anchor: bool = False


@dataclass
class DivergencePoint:
    """두 사슬이 갈라지는 지점."""
    depth: int              # 몇 번째 마디에서 갈라졌나
    sim: float              # 그 지점의 벡터 유사도(낮을수록 크게 갈라짐)
    kind: str               # "match" | "reword" | "divergence"
    stmt_a: str
    stmt_b: str
    severity: str           # "none" | "minor" | "severe"


@dataclass
class VectorRecord:
    """기록할 벡터 하나 (나중에 핵심 짚기용)."""
    label: str              # "match" | "mismatch" | "hallucination" | ...
    text: str
    vector: list            # 임베딩 (list로 저장)
    depth: int = -1
    meta: dict = field(default_factory=dict)


class DivergenceTracer:
    """
    두(이상의) 근거 사슬을 정렬해 분기점을 찾고,
    모든 문장 벡터를 기록한다.
    """

    def __init__(self, embed_fn: Optional[Callable] = None,
                 reword_threshold: float = 0.85,
                 divergence_threshold: float = 0.5):
        # embed_fn: text -> np.ndarray. 없으면 해시 폴백.
        self.embed = embed_fn or (lambda t: _hash_embed(t))
        self.reword_th = reword_threshold      # 이 이상이면 '표현만 다름'
        self.diverge_th = divergence_threshold  # 이 이하면 '진짜 분기'
        self.records: list = []                 # VectorRecord 전량 기록

    def _embed_and_record(self, text: str, label: str, depth: int = -1, meta=None) -> np.ndarray:
        v = self.embed(text)
        self.records.append(VectorRecord(
            label=label, text=text, vector=list(map(float, v)),
            depth=depth, meta=meta or {}))
        return v

    def compare_chains(self, chain_a: list, chain_b: list) -> dict:
        """
        두 사슬(ChainNode 리스트, depth 오름차순)을 마디별로 비교.
        같은 depth끼리 벡터 유사도를 재서 분기점을 찾는다.
        """
        divergences = []
        max_depth = min(len(chain_a), len(chain_b))
        first_divergence = None

        for d in range(max_depth):
            na, nb = chain_a[d], chain_b[d]
            va = self._embed_and_record(na.statement, "chainA", na.depth,
                                        {"anchor": na.is_anchor})
            vb = self._embed_and_record(nb.statement, "chainB", nb.depth,
                                        {"anchor": nb.is_anchor})
            sim = cos_sim(va, vb)

            # 분기 종류 판정
            if sim >= self.reword_th:
                kind, sev = "match", "none"
            elif sim <= self.diverge_th:
                kind, sev = "divergence", None
            else:
                kind, sev = "reword", "minor"   # 애매대 = 표현 차이로 봄

            # 심각도: 분기가 앵커에 가까울수록 심각
            if kind == "divergence":
                # depth가 클수록(앵커 가까움) severe
                near_anchor = (d >= max_depth - 2) or na.is_anchor or nb.is_anchor
                sev = "severe" if near_anchor else "minor"

            dp = DivergencePoint(
                depth=d, sim=round(sim, 3), kind=kind,
                stmt_a=na.statement[:50], stmt_b=nb.statement[:50], severity=sev)
            divergences.append(dp)

            if kind == "divergence" and first_divergence is None:
                first_divergence = dp

        # 요약
        return {
            "first_divergence": asdict(first_divergence) if first_divergence else None,
            "all_points": [asdict(dp) for dp in divergences],
            "verdict": self._verdict(first_divergence, max_depth),
        }

    def _verdict(self, first_div: Optional[DivergencePoint], depth: int) -> str:
        if first_div is None:
            return "일치: 사슬 전체가 같은 근거 흐름 (믿을 만함)"
        if first_div.severity == "severe":
            return (f"심각 분기: {first_div.depth}단계(앵커 근처)에서 근본 근거가 갈림 "
                    f"— 신뢰 어려움")
        return (f"경미 분기: {first_div.depth}단계(말단)에서 갈림 "
                f"— 표현/세부 차이일 가능성")

    # ── 환각 문장 등 라벨 기록 (핵심 짚기용) ──
    def record_labeled(self, text: str, label: str, meta=None):
        """일치/불일치/환각 문장을 라벨과 함께 벡터 기록."""
        return self._embed_and_record(text, label, meta=meta)

    # ── 기록 저장/로드 ──
    def save_records(self, path: str):
        blob = {
            "records": [asdict(r) for r in self.records],
            "saved_at": time.time(),
            "n": len(self.records),
        }
        with open(path, "wb") as f:
            pickle.dump(blob, f)
        return path

    @staticmethod
    def load_records(path: str) -> dict:
        with open(path, "rb") as f:
            return pickle.load(f)

    def label_stats(self) -> dict:
        """기록된 벡터를 라벨별로 집계 + 라벨 간 평균 거리."""
        from collections import defaultdict
        by = defaultdict(list)
        for r in self.records:
            by[r.label].append(np.array(r.vector))
        stats = {}
        labels = list(by.keys())
        # 라벨별 개수 + 내부 응집도
        for lb, vs in by.items():
            if len(vs) >= 2:
                center = np.mean(vs, axis=0)
                spread = float(np.mean([1 - cos_sim(v, center) for v in vs]))
            else:
                spread = 0.0
            stats[lb] = {"count": len(vs), "spread": round(spread, 3)}
        # 라벨 쌍 간 거리
        pair = {}
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                ca = np.mean(by[labels[i]], axis=0)
                cb = np.mean(by[labels[j]], axis=0)
                pair[f"{labels[i]}~{labels[j]}"] = round(1 - cos_sim(ca, cb), 3)
        return {"per_label": stats, "pair_distance": pair}
