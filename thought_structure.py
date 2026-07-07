"""
thought_structure.py — 사고 구조 아키텍처 핵심 엔진.

NM 판단 분기 트리 + LLM 이동 + 대화 반응 학습.

구조:
  - 판단 노드: 하나의 의미 있는 판단 지점
  - 전이 확률(NM): 이 판단에서 다음 판단으로 갈 확률
  - LLM: 각 분기점에서 후보 중 선택 (NM=지도, LLM=운전자)
  - 평가: 대화 반응(긍정/부정)으로 전이 확률 강화/약화

핵심 철학:
  - 처음엔 오류 있어도 됨
  - 대화하며 실수에서 점점 벗어남
  - 학습된 전이 확률 = 정체성
  - 경로 기록 = XAI

정직한 범위:
  - LLM 선택은 주입식(choose_fn). 없으면 확률 기반 자동 선택(시뮬).
  - 이건 뼈대다. 실제 LLM 연결 + 대화 반응 감지는 위에 얹는다.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict
import random
import json
import pickle


@dataclass
class JudgmentNode:
    """판단 노드 — 하나의 의미 있는 판단 지점."""
    id: str
    prompt: str          # 이 판단의 이름 (분기 선택 시 LLM에게 보임)
    directive: str = ""  # 답변 생성 시 실제로 강제되는 구체적 실행 지시
    is_terminal: bool = False   # 최종 판단(잎)인가


@dataclass
class PathRecord:
    """한 번의 사고 궤적 기록 (감사 로그)."""
    path: list                  # 지나간 노드 id 순서
    choices: list               # 각 분기점에서의 선택
    answer: str = ""
    feedback: Optional[float] = None   # 대화 반응 (+1 긍정 / -1 부정 / None)
    context: str = ""           # 입력 질문 (감사: 무엇에 답했나)
    sources: list = field(default_factory=list)  # 가져온 근거 자료 (감사: 무엇을 근거로)
    timestamp: str = ""         # 시각 (감사: 언제)


class ThoughtStructure:
    """
    NM 판단 분기 트리. LLM이 그 위를 이동하며 사고한다.
    전이 확률이 대화 반응으로 강화/약화 = 학습 = 정체성 성장.
    """
    def __init__(self, learning_rate: float = 0.1,
                 continuity: float = 0.8):
        self.nodes: Dict[str, JudgmentNode] = {}
        # 전이 확률(NM): {from_id: {to_id: prob}}
        self.transitions: Dict[str, Dict[str, float]] = {}
        self.root_id: str = ""
        self.lr = learning_rate            # 학습률 (전이 변화 폭)
        self.continuity = continuity       # 정체성 유지 강도 (뼈대 보존)
        self.history: List[PathRecord] = []

    # ── 트리 구성 ──
    def add_node(self, node: JudgmentNode, is_root=False):
        self.nodes[node.id] = node
        self.transitions.setdefault(node.id, {})
        if is_root:
            self.root_id = node.id

    def add_branch(self, from_id: str, to_id: str, prob: float = None):
        """판단 분기 추가. prob 없으면 균등 분배."""
        self.transitions.setdefault(from_id, {})
        self.transitions[from_id][to_id] = prob if prob is not None else 0.5
        self._normalize(from_id)

    def _normalize(self, node_id: str):
        """전이 확률 합을 1로."""
        tr = self.transitions.get(node_id, {})
        s = sum(tr.values())
        if s > 0:
            for k in tr:
                tr[k] /= s

    # ── LLM 이동 (사고) ──
    def traverse(self, choose_fn: Optional[Callable] = None,
                 answer_fn: Optional[Callable] = None,
                 context: str = "") -> PathRecord:
        """
        트리를 따라 이동하며 사고.
        choose_fn(node, candidates, probs, context) -> chosen_id
          없으면 전이 확률로 자동 선택(시뮬).
        answer_fn(path, context) -> answer  (최종 답변, 선택)
        """
        path, choices = [], []
        cur = self.root_id
        depth = 0
        while cur and depth < 20:
            path.append(cur)
            node = self.nodes[cur]
            if node.is_terminal:
                break
            candidates = list(self.transitions.get(cur, {}).keys())
            if not candidates:
                break
            probs = [self.transitions[cur][c] for c in candidates]

            # LLM이 선택 (없으면 확률 기반 자동)
            if choose_fn is not None:
                chosen = choose_fn(node, candidates,
                                   [self.nodes[c] for c in candidates],
                                   probs, context)
            else:
                chosen = random.choices(candidates, weights=probs)[0]

            choices.append({"at": cur, "chose": chosen})
            cur = chosen
            depth += 1

        answer = ""
        sources = []
        if answer_fn is not None:
            result = answer_fn(path, context)
            # answer_fn이 (답변, 근거리스트) 튜플이면 분리, 아니면 답변만
            if isinstance(result, tuple) and len(result) == 2:
                answer, sources = result
            else:
                answer = result

        from datetime import datetime
        rec = PathRecord(path=path, choices=choices, answer=answer,
                         context=context, sources=sources or [],
                         timestamp=datetime.now().isoformat(timespec="seconds"))
        return rec

    # ── 대화 반응으로 학습 (전이 강화/약화) ──
    def learn(self, record: PathRecord, feedback: float):
        """
        feedback: +1(긍정) ~ -1(부정)
        지나간 경로의 전이를 강화(긍정) 또는 약화(부정).
        continuity로 뼈대는 보존 (정체성 유지).
        """
        record.feedback = feedback
        self.history.append(record)

        # 경로상의 각 전이를 업데이트
        for ch in record.choices:
            frm, to = ch["at"], ch["chose"]
            tr = self.transitions[frm]
            old = tr.get(to, 0.5)
            # 긍정이면 이 선택 강화, 부정이면 약화
            delta = self.lr * feedback
            # continuity: 급변 방지 (뼈대 보존)
            new = old + delta * (1 - self.continuity) + delta * self.continuity * 0.3
            tr[to] = max(0.01, min(0.99, new))
            self._normalize(frm)

    # ── 정체성 지표 ──
    def dominant_path(self) -> list:
        """현재 가장 강한 판단 경로 = 이 정체성의 기본 사고."""
        path = [self.root_id]
        cur = self.root_id
        seen = {cur}
        while True:
            tr = self.transitions.get(cur, {})
            if not tr:
                break
            nxt = max(tr, key=tr.get)
            if nxt in seen:
                break
            path.append(nxt)
            seen.add(nxt)
            cur = nxt
            if self.nodes[cur].is_terminal:
                break
        return path

    def continuity_rate(self) -> float:
        """최근 궤적들이 얼마나 일관된가 = 정체성 안정도."""
        recent = self.history[-10:]
        if len(recent) < 2:
            return 1.0
        paths = [tuple(r.path) for r in recent]
        most = max(set(paths), key=paths.count)
        return paths.count(most) / len(paths)

    # ── 저장/복원 (정체성 지속) ──
    def save(self, path: str):
        blob = {
            "nodes": {k: v.__dict__ for k, v in self.nodes.items()},
            "transitions": self.transitions,
            "root_id": self.root_id,
            "lr": self.lr, "continuity": self.continuity,
            "history": [r.__dict__ for r in self.history],
        }
        with open(path, "wb") as f:
            pickle.dump(blob, f)
        return path

    @classmethod
    def load(cls, path: str) -> "ThoughtStructure":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        ts = cls(learning_rate=blob["lr"], continuity=blob["continuity"])
        for k, v in blob["nodes"].items():
            ts.nodes[k] = JudgmentNode(**v)
        ts.transitions = blob["transitions"]
        ts.root_id = blob["root_id"]
        ts.history = [PathRecord(**r) for r in blob["history"]]
        return ts
