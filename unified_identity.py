"""
unified_identity.py — 하나의 정체성 = 유형별 트리 + 정체성 기억.

두 축이 합쳐져 하나의 지속되는 존재가 된다:

  1. TreeRegistry (어떻게 판단하나 — 구조)
     - 유형별 사고 트리, 재사용·진화
     - 사고 계보(궤적) = 감사 로그

  2. IdentityMemory (무엇을 믿나/아나 — 내용)
     - persona(성격), values(가치관), judgment_rules(판단기준)
     - learned_facts(배운 것), episodic(대화 맥락)
     - 매 답변에 system_prompt로 강제 주입

결합 원리:
  질문 → 유형 판별 → 그 유형 트리로 판단 경로 결정 (구조)
       + 정체성 기억을 프롬프트에 주입 (내용)
       → 답변
       → 반응으로 트리 진화 + 기억 흡수
  둘 다 pkl에 담겨 지속된다.
"""
from __future__ import annotations
from typing import Optional, Callable
import pickle

from tree_registry import TreeRegistry
from thought_structure import ThoughtStructure, JudgmentNode, PathRecord
from identity_core import IdentityMemory


class UnifiedIdentity:
    """유형별 트리 + 정체성 기억 = 하나의 정체성."""

    def __init__(self, registry: Optional[TreeRegistry] = None,
                 memory: Optional[IdentityMemory] = None):
        self.registry = registry or TreeRegistry()
        self.identity = memory or IdentityMemory()

    def think(self, question: str,
              choose_fn: Optional[Callable] = None,
              answer_fn: Optional[Callable] = None,
              classify_fn: Optional[Callable] = None,
              embed_fn: Optional[Callable] = None,
              tree_factory: Optional[Callable] = None) -> dict:
        """
        하나의 사고 사이클:
          1. 유형 판별 → 그 유형 트리 재사용/생성
          2. 정체성 기억을 맥락에 주입
          3. 트리 위를 이동하며 답 (구조가 답을 통제)
          4. 결과 반환 (경로=감사, 답변)
        """
        # 1. 유형 판별 → 트리 선택 (재사용/생성)
        type_id = self.registry.classify_type(
            question, embed_fn=embed_fn, classify_fn=classify_fn)
        if type_id is None:
            type_id = "general"
        tree = self.registry.get_or_create(type_id, factory=tree_factory)

        # 2. 정체성 기억 + 트리 관련 기억을 맥락에 주입
        identity_prompt = self.identity.to_system_prompt()
        tree_mem = tree.memory_context(question, embed_fn=embed_fn)
        context_parts = []
        if identity_prompt:
            context_parts.append(identity_prompt)
        if tree_mem:
            context_parts.append(tree_mem)
        context_parts.append(f"질문: {question}")
        full_context = "\n\n".join(context_parts)

        # 3. 트리 위를 이동 (구조가 답 통제)
        rec = tree.traverse(choose_fn=choose_fn, answer_fn=answer_fn,
                            context=full_context)
        rec.context = question   # 감사 로그엔 원 질문만

        return {
            "type": type_id,
            "answer": rec.answer,
            "path": rec.path,
            "record": rec,
            "tree": tree,
        }

    def react(self, result: dict, feedback: float,
              was_corrected: bool = False):
        """
        대화 반응으로 진화:
          - 트리 전이 학습 (구조 진화)
          - 교정이면 확정 기억 + 정체성 흡수 (내용 진화)
        """
        tree = result["tree"]
        rec = result["record"]
        tree.learn(rec, feedback)

        if was_corrected and feedback > 0:
            # 교정 확인 → 트리 확정 기억 + 정체성 learned_facts
            tree.remember_from_correction(rec, None)
            if rec.answer:
                self.identity.absorb(rec.context, rec.answer,
                                     learned=rec.answer[:80])
        # 매 반응 후 망각
        tree.forget_step()

    def status(self) -> dict:
        """현재 정체성 상태 (유형별 성숙 + 정체성 기억)."""
        return {
            "types": self.registry.stats(),
            "identity": {
                "persona": self.identity.persona,
                "values": self.identity.values,
                "rules": self.identity.judgment_rules,
                "facts": len(self.identity.learned_facts),
                "episodes": len(self.identity.episodic),
            },
        }

    # ── 저장/복원 (트리 + 정체성 기억 통째로) ──
    def save(self, path: str):
        import tempfile, os
        # registry 따로 저장 후 합침
        blob = {
            "registry": {
                "trees": {tid: TreeRegistry._tree_blob(t)
                          for tid, t in self.registry.trees.items()},
                "type_examples": self.registry.type_examples,
                "usage_count": self.registry.usage_count,
            },
            "identity": self.identity.__dict__,
        }
        with open(path, "wb") as f:
            pickle.dump(blob, f)
        return path

    @classmethod
    def load(cls, path: str) -> "UnifiedIdentity":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        # registry 복원
        reg = TreeRegistry()
        for tid, tb in blob["registry"]["trees"].items():
            t = ThoughtStructure(learning_rate=tb["lr"], continuity=tb["continuity"])
            for k, v in tb["nodes"].items():
                t.nodes[k] = JudgmentNode(**v)
            t.transitions = tb["transitions"]; t.root_id = tb["root_id"]
            t.history = [PathRecord(**r) for r in tb["history"]]
            t.memory = tb.get("memory", [])
            reg.trees[tid] = t
        reg.type_examples = blob["registry"].get("type_examples", {})
        reg.usage_count = blob["registry"].get("usage_count", {})
        # identity 복원
        mem = IdentityMemory(**blob["identity"])
        return cls(registry=reg, memory=mem)
