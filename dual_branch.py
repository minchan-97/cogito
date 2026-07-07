"""
dual_branch.py — 분기점 2유형: 근거 분기 + 논리 분기.

민찬기님 통찰:
  "어떻게 해야 해?" 같은 결정/추론 질문은 근거만으론 부족.
  추론 사슬의 각 연결(A니까 B)이 논리 분기점.
  분기별로 보면 논리가 흔들리는 지점이 바로 보인다.

2유형:
  1. 근거 분기 (evidence): 이 단계가 자료/근거에 실재하나 (사실성)
  2. 논리 분기 (logic):    단계 A → 단계 B 연결이 성립하나 (추론)

논리 분기 측정:
  - 연결 강도: 인접 단계 간 의미 거리 (너무 멀면 논리 비약)
  - 방향 일관성: 사슬이 한 방향으로 흐르나, 튀나
  - (선택) 함의: entail_fn 있으면 A→B 함의 점수

핵심: 근거는 다 참인데 연결이 무너지는 환각을 논리 분기에서 잡는다.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, List
import numpy as np


def cos(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


@dataclass
class BranchPoint:
    idx: int
    kind: str            # "evidence" | "logic"
    statement: str
    score: float         # 근거: 자료 실재도 / 논리: 연결 강도
    shaky: bool          # 흔들리는 분기인가
    note: str = ""


@dataclass
class DualBranchResult:
    evidence_branches: list
    logic_branches: list
    shaky_points: list   # 흔들리는 지점 (근거+논리 통합)
    verdict: str


class DualBranchAnalyzer:
    """
    추론 사슬을 받아 근거 분기 + 논리 분기로 분석.
    embed_fn: text -> vector
    corpus: 근거 실재 확인용 자료 (선택)
    entail_fn: (premise, hypothesis) -> float, A→B 함의 점수 (선택)
    """
    def __init__(self, embed_fn: Callable, corpus: str = "",
                 entail_fn: Optional[Callable] = None,
                 logic_gap_threshold: float = 0.5,
                 logic_leap_z: float = 1.5):
        self.embed = embed_fn
        self.corpus = corpus
        self.corpus_norm = self._norm(corpus)
        self.entail = entail_fn
        self.gap_th = logic_gap_threshold    # 연결 강도 이 밑이면 논리 비약
        self.leap_z = logic_leap_z           # 연결 강도가 이만큼 튀면 흔들림

    @staticmethod
    def _norm(t):
        import re
        return re.sub(r'\s+', '', t)

    def _in_corpus(self, text: str) -> float:
        """근거가 자료에 얼마나 실재하나 (0~1, 대략)."""
        if not self.corpus:
            return -1  # 자료 없음
        import re
        words = set(re.findall(r'[가-힣a-zA-Z0-9]{2,}', text))
        if not words:
            return 0.0
        hit = sum(1 for w in words if w in self.corpus_norm)
        return hit / len(words)

    def analyze(self, steps: List[str]) -> DualBranchResult:
        """
        steps: 추론 사슬의 단계들 (순서대로).
          예: ["비가 온다", "땅이 젖는다", "그래서 우산이 필요없다"]
        """
        vecs = [self.embed(s) for s in steps]

        # ── 1. 근거 분기: 각 단계가 자료에 실재하나 ──
        evidence = []
        for i, s in enumerate(steps):
            ground = self._in_corpus(s)
            shaky = (ground >= 0 and ground < 0.3)  # 자료 있고, 실재도 낮으면 흔들림
            evidence.append(BranchPoint(
                idx=i, kind="evidence", statement=s,
                score=round(ground, 3) if ground >= 0 else -1,
                shaky=shaky,
                note="근거 자료에 희박" if shaky else ""))

        # ── 2. 논리 분기: 인접 단계 연결 강도 ──
        gaps = []
        for i in range(len(steps) - 1):
            link = cos(vecs[i], vecs[i+1])  # 연결 강도(의미 인접성)
            gaps.append(link)

        # 연결 강도의 통계로 '튀는 연결'(논리 비약) 탐지
        logic = []
        if gaps:
            arr = np.array(gaps)
            mu, sd = arr.mean(), arr.std()
            for i, link in enumerate(gaps):
                # 흔들림: 연결이 너무 약하거나(비약), 통계적으로 튐
                z = (link - mu) / sd if sd > 1e-9 else 0
                too_weak = link < self.gap_th
                spikes = abs(z) >= self.leap_z
                shaky = too_weak or spikes

                # 함의 검사 (있으면)
                entail_score = None
                if self.entail is not None:
                    entail_score = self.entail(steps[i], steps[i+1])
                    if entail_score < 0.3:
                        shaky = True

                note = []
                if too_weak: note.append("연결 약함(논리 비약)")
                if spikes: note.append(f"연결 강도 튐(z={z:.1f})")
                if entail_score is not None and entail_score < 0.3:
                    note.append(f"함의 낮음({entail_score:.2f})")

                logic.append(BranchPoint(
                    idx=i, kind="logic",
                    statement=f"[{steps[i][:20]}] → [{steps[i+1][:20]}]",
                    score=round(link, 3), shaky=shaky,
                    note=" / ".join(note)))

        # ── 흔들리는 지점 통합 ──
        shaky_points = ([b for b in evidence if b.shaky] +
                        [b for b in logic if b.shaky])

        return DualBranchResult(
            evidence_branches=evidence,
            logic_branches=logic,
            shaky_points=shaky_points,
            verdict=self._verdict(evidence, logic))

    def _verdict(self, evidence, logic) -> str:
        msgs = []
        ev_shaky = [b for b in evidence if b.shaky]
        lg_shaky = [b for b in logic if b.shaky]
        if ev_shaky:
            msgs.append(f"근거 흔들림 {len(ev_shaky)}곳 — 자료에 없는 단계 존재")
        if lg_shaky:
            idxs = ", ".join(f"{b.idx}→{b.idx+1}" for b in lg_shaky)
            msgs.append(f"논리 흔들림 {len(lg_shaky)}곳 ({idxs}) — 추론 연결이 약한 분기")
        if not msgs:
            return "근거·논리 모두 안정 — 사슬이 고르게 이어짐"
        # 논리만 흔들리면 특별히 강조 (근거는 참인데 연결이 틀린 케이스)
        if lg_shaky and not ev_shaky:
            msgs.append("★ 근거는 참인데 논리가 무너짐 — 근거검증으론 못 잡는 환각")
        return " / ".join(msgs)

