"""
tree_registry.py — 유형별 사고 트리 저장소.

남들 동적 트리(일회용)와 다른 점:
  남들: 질문마다 트리 생성 → 답 내면 버림
  민찬기: 유형별 트리를 보관 → 재사용 → 각각 진화

핵심:
  - 질문이 오면 '유형'을 판별
  - 그 유형의 트리가 있으면 재사용, 없으면 새로 만들어 등록
  - 사용 후 대화 반응으로 그 유형 트리가 진화 (전이 학습)
  - 유형별로 따로 자라므로, 도메인별 전문성이 축적됨

+ IdentityMemory(예전 정체성 모델) 결합:
  - 트리는 '어떻게 판단하나'(구조)
  - IdentityMemory는 '무엇을 아나/믿나'(내용)
  - 둘이 함께 하나의 정체성
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import pickle
import time

from thought_structure import ThoughtStructure, JudgmentNode


class TreeRegistry:
    """
    유형(type) → 사고 트리 매핑.
    각 유형이 독립적으로 재사용·진화한다.
    """
    def __init__(self):
        self.trees: Dict[str, ThoughtStructure] = {}
        self.type_examples: Dict[str, List[str]] = {}   # 유형별 예시 질문 (판별용)
        self.usage_count: Dict[str, int] = {}

    def register_type(self, type_id: str, tree: ThoughtStructure,
                      examples: List[str] = None):
        """새 유형과 그 트리를 등록."""
        self.trees[type_id] = tree
        self.type_examples[type_id] = examples or []
        self.usage_count.setdefault(type_id, 0)

    def classify_type(self, question: str, embed_fn: Optional[Callable] = None,
                      classify_fn: Optional[Callable] = None) -> Optional[str]:
        """
        질문이 어느 유형인지 판별.
        classify_fn(question, type_ids, examples) -> type_id  (LLM 판별, 우선)
        embed_fn 있으면 예시와의 유사도로.
        없으면 단어 겹침.
        """
        if not self.trees:
            return None
        type_ids = list(self.trees.keys())

        # LLM 판별 (제일 정확)
        if classify_fn is not None:
            chosen = classify_fn(question, type_ids, self.type_examples)
            if chosen in self.trees:
                return chosen

        # 임베딩 유사도
        if embed_fn is not None:
            import numpy as np
            qv = embed_fn(question)
            best, best_sim = None, -1
            for tid in type_ids:
                for ex in self.type_examples.get(tid, []):
                    ev = embed_fn(ex)
                    sim = float(np.dot(qv, ev) /
                                ((np.linalg.norm(qv)*np.linalg.norm(ev))+1e-12))
                    if sim > best_sim:
                        best_sim, best = sim, tid
            if best_sim > 0.3:
                return best

        # 단어 겹침 (폴백)
        import re
        qw = set(re.findall(r'[가-힣a-zA-Z0-9]{2,}', question))
        best, best_ov = None, 0
        for tid in type_ids:
            for ex in self.type_examples.get(tid, []):
                ew = set(re.findall(r'[가-힣a-zA-Z0-9]{2,}', ex))
                ov = len(qw & ew)
                if ov > best_ov:
                    best_ov, best = ov, tid
        return best if best_ov > 0 else type_ids[0]

    def get_or_create(self, type_id: str,
                      factory: Optional[Callable] = None) -> ThoughtStructure:
        """유형의 트리를 재사용, 없으면 생성."""
        if type_id not in self.trees:
            tree = factory() if factory else ThoughtStructure()
            self.register_type(type_id, tree)
        self.usage_count[type_id] = self.usage_count.get(type_id, 0) + 1
        return self.trees[type_id]

    def stats(self) -> dict:
        """유형별 성숙도 (재사용·학습 횟수)."""
        return {
            tid: {
                "usage": self.usage_count.get(tid, 0),
                "learned": len(t.history),
                "memory": len(t.memory),
                "dominant": [t.nodes[n].prompt[:10] for n in t.dominant_path()],
            }
            for tid, t in self.trees.items()
        }

    # ── 저장/복원 (모든 유형 트리 + 성숙도 지속) ──
    def save(self, path: str):
        blob = {
            "trees": {tid: self._tree_blob(t) for tid, t in self.trees.items()},
            "type_examples": self.type_examples,
            "usage_count": self.usage_count,
        }
        with open(path, "wb") as f:
            pickle.dump(blob, f)
        return path

    @staticmethod
    def _tree_blob(t: ThoughtStructure) -> dict:
        return {
            "nodes": {k: v.__dict__ for k, v in t.nodes.items()},
            "transitions": t.transitions, "root_id": t.root_id,
            "lr": t.lr, "continuity": t.continuity,
            "history": [r.__dict__ for r in t.history],
            "memory": t.memory,
        }

    @classmethod
    def load(cls, path: str) -> "TreeRegistry":
        from thought_structure import PathRecord
        with open(path, "rb") as f:
            blob = pickle.load(f)
        reg = cls()
        for tid, tb in blob["trees"].items():
            t = ThoughtStructure(learning_rate=tb["lr"], continuity=tb["continuity"])
            for k, v in tb["nodes"].items():
                t.nodes[k] = JudgmentNode(**v)
            t.transitions = tb["transitions"]
            t.root_id = tb["root_id"]
            t.history = [PathRecord(**r) for r in tb["history"]]
            t.memory = tb.get("memory", [])
            reg.trees[tid] = t
        reg.type_examples = blob.get("type_examples", {})
        reg.usage_count = blob.get("usage_count", {})
        return reg
