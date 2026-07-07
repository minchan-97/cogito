"""
anchor_probe.py — 근거 앵커 심고, 유사도로 '확 튀는' 지점 잡기.

민찬기님 설계:
  1. 질문에 필요한 근거들을 앵커로 심음
  2. 각 근거 ↔ 답변 유사도 → 답이 근거를 따르나 (안 따르면 튐)
  3. 근거 ↔ 근거 거리 → 근거들끼리 일관되나 (모순되면 튐)
  4. 근거를 하나씩 빼면서 답 변화 → 그 근거가 답을 움직였나
  5. 튀는 지점(anomaly)을 계속 검증

내부(LLM)를 못 봐도, 근거(앵커)를 통제 조작해서 출력 반응으로 추론.
= 통제 실험(controlled probe). 블랙박스 우회.

핵심 판별:
  - 근거 뺐는데 답 그대로 → 그 근거 안 씀 (근거 없이 지어냄 = 환각 신호)
  - 근거 줬는데 답이 안 따라감 → 근거 무시 (환각 신호)
  - 근거들끼리 거리가 확 튐 → 모순된 근거 섞임
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, List
import numpy as np


def cos(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


@dataclass
class ProbeResult:
    anchor_answer_sims: list      # 각 근거 ↔ 답변 유사도
    anchor_anchor_dists: list     # 근거 쌍 사이 거리
    ablation_shifts: list         # 근거 하나씩 뺐을 때 답 변화량
    spikes: list                  # 확 튀는 지점 (anomaly)
    verdict: str


class AnchorProbe:
    """
    근거 앵커를 조작하며 답변 반응을 유사도로 추적.
    embed_fn: text -> vector (필수)
    answer_fn: (question, anchors) -> answer_text (근거 빼면서 답 재생성용, 선택)
    """
    def __init__(self, embed_fn: Callable,
                 answer_fn: Optional[Callable] = None,
                 spike_threshold: float = 2.0):
        self.embed = embed_fn
        self.answer_fn = answer_fn
        self.spike_th = spike_threshold   # 표준편차 몇 배 벗어나면 '튐'

    def _spikes(self, values: list, labels: list) -> list:
        """값들 중 통계적으로 확 튀는 것 잡기 (평균 ± spike_th·표준편차 밖)."""
        if len(values) < 2:
            return []
        arr = np.array(values)
        mu, sd = arr.mean(), arr.std()
        if sd < 1e-9:
            return []
        out = []
        for v, lb in zip(values, labels):
            z = (v - mu) / sd
            if abs(z) >= self.spike_th:
                out.append({"label": lb, "value": round(v, 3), "z": round(z, 2)})
        return out

    def probe(self, question: str, answer: str, anchors: List[str]) -> ProbeResult:
        av = self.embed(answer)
        anchor_vecs = [self.embed(a) for a in anchors]

        # 1. 각 근거 ↔ 답변 유사도
        aa_sims = [cos(av, v) for v in anchor_vecs]

        # 2. 근거 ↔ 근거 거리 (쌍별)
        aa_dists = []
        pair_labels = []
        for i in range(len(anchors)):
            for j in range(i + 1, len(anchors)):
                d = 1 - cos(anchor_vecs[i], anchor_vecs[j])
                aa_dists.append(d)
                pair_labels.append(f"근거{i+1}~근거{j+1}")

        # 3. 근거 하나씩 빼면서 답 변화 (answer_fn 있을 때만)
        ablation = []
        if self.answer_fn is not None:
            base_ans = answer
            base_v = av
            for k in range(len(anchors)):
                reduced = anchors[:k] + anchors[k+1:]
                new_ans = self.answer_fn(question, reduced)
                new_v = self.embed(new_ans)
                shift = 1 - cos(base_v, new_v)   # 답이 얼마나 움직였나
                ablation.append({"removed": f"근거{k+1}", "shift": round(shift, 3)})

        # 4. 확 튀는 지점 잡기
        spikes = []
        # 근거-답변 유사도가 튀는 것 (특정 근거만 답과 동떨어짐)
        spikes += [{"type": "근거-답변", **s}
                   for s in self._spikes(aa_sims, [f"근거{i+1}" for i in range(len(anchors))])]
        # 근거-근거 거리가 튀는 것 (모순된 근거)
        spikes += [{"type": "근거-근거", **s}
                   for s in self._spikes(aa_dists, pair_labels)]

        return ProbeResult(
            anchor_answer_sims=[round(s, 3) for s in aa_sims],
            anchor_anchor_dists=[round(d, 3) for d in aa_dists],
            ablation_shifts=ablation,
            spikes=spikes,
            verdict=self._verdict(aa_sims, spikes, ablation),
        )

    def _verdict(self, aa_sims, spikes, ablation) -> str:
        msgs = []
        # 모든 근거가 답과 동떨어짐 → 근거 무시 의심
        if aa_sims and max(aa_sims) < 0.4:
            msgs.append("답변이 어떤 근거와도 가깝지 않음 — 근거 무시(환각) 의심")
        # 튀는 지점 있음
        if spikes:
            types = set(s["type"] for s in spikes)
            if "근거-근거" in types:
                msgs.append("근거들 사이 거리가 튐 — 모순되거나 이질적 근거 섞임")
            if "근거-답변" in types:
                msgs.append("특정 근거만 답과 동떨어짐 — 그 근거 미반영 가능")
        # ablation: 뺐는데 답 안 움직임
        if ablation:
            dead = [a["removed"] for a in ablation if a["shift"] < 0.05]
            if dead:
                msgs.append(f"{', '.join(dead)}를 빼도 답 불변 — 해당 근거 실제 미사용(지어냄 가능)")
        return " / ".join(msgs) if msgs else "근거들이 답변에 고르게 반영됨 (안정)"

