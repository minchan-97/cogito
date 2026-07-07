"""
cogito_trace.py — 근거 소급 사슬 추적 + 앵커 검증 (스케치)

핵심 아이디어:
  LLM 답변의 근거를 인용으로 가져오고, 그 근거의 근거를 계속 소급해
  '최종 앵커'까지의 흐름을 하나의 사슬로 기록한다.
  각 연결이 실재하는지 검증하고, 사슬이 신뢰할 앵커에서 끝나는지 본다.

cogito 연결:
  설명은 무한 소급한다("왜? 왜?"). 사슬은 어딘가서 멈춰야 한다.
  그 종결점 = 앵커. 두 종류:
    - HUMAN: 인간이 책임지고 참이라 선언/서명한 지점 (결재 원본 등)
    - AXIOM: 공리적 문서 (법령·헌법 등, 더 소급 안 하기로 한 근거)
  앵커에서 끝나면 '믿을 근거가 있음', 앵커 없이 끊기면 '근거 없는 점프'.

정직한 범위:
  이건 'LLM의 진짜 내부 사고'가 아니라 'LLM이 제시한 근거 사슬'의 추적이다.
  단, 각 연결의 실재를 검증하므로, LLM이 갖다 붙인 가짜 근거는 사슬에서 끊긴다.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import re


class AnchorType(Enum):
    HUMAN = "human"      # 인간 보증 (결재·서명) — cogito 정통 종결
    AXIOM = "axiom"      # 공리적 문서 (법령·헌법)
    NONE = "none"        # 아직 앵커 없음 (소급 미완/끊김)


@dataclass
class Claim:
    """사슬의 한 마디: 하나의 주장과 그 근거 인용."""
    id: str
    statement: str              # 이 마디의 주장
    quote: str = ""             # 이 주장의 근거로 제시된 원문 인용
    source_id: str = ""         # 그 인용이 나온 상위 근거(마디)의 id — 소급 링크
    anchor: AnchorType = AnchorType.NONE
    verified: bool = False      # quote가 상위 근거에 실재하는지 검증됨?
    note: str = ""


@dataclass
class TraceChain:
    """답변 하나에 대한 근거 소급 사슬 전체."""
    answer: str
    claims: dict = field(default_factory=dict)   # id -> Claim
    root_id: str = ""                            # 답변 자체(사슬 시작점)

    def add(self, claim: Claim):
        self.claims[claim.id] = claim

    def chain_from(self, cid: str) -> list:
        """한 주장에서 앵커까지 소급 경로를 순서대로 반환."""
        path = []
        seen = set()
        cur = cid
        while cur and cur in self.claims and cur not in seen:
            seen.add(cur)
            c = self.claims[cur]
            path.append(c)
            if c.anchor != AnchorType.NONE:
                break            # 앵커 도달 → 종결
            cur = c.source_id    # 상위 근거로 소급
        return path

    def evaluate(self, cid: str) -> dict:
        """
        한 주장의 신뢰 상태를 사슬로 판정.
        - 앵커에서 끝나고 전 연결이 검증됨 → 'grounded'
        - 앵커까지 갔지만 중간 연결 미검증 → 'weak'
        - 앵커 없이 끊김 → 'ungrounded' (근거 없는 점프 = 의심)
        """
        path = self.chain_from(cid)
        if not path:
            return {"status": "unknown", "reason": "주장을 찾을 수 없음", "path": []}

        last = path[-1]
        all_verified = all(c.verified for c in path[1:]) if len(path) > 1 else path[0].verified
        reached_anchor = last.anchor != AnchorType.NONE

        if reached_anchor and all_verified:
            status = "grounded"
            reason = f"{last.anchor.value} 앵커까지 {len(path)}단계, 전 연결 검증됨"
        elif reached_anchor and not all_verified:
            status = "weak"
            reason = f"{last.anchor.value} 앵커 도달했으나 일부 연결 미검증"
        else:
            status = "ungrounded"
            reason = f"{len(path)}단계 소급 후 앵커 없이 끊김 — 근거 없는 주장 가능"

        return {
            "status": status,
            "reason": reason,
            "anchor": last.anchor.value,
            "depth": len(path),
            "path": [f"{c.id}: {c.statement[:40]}" for c in path],
        }


def verify_link(quote: str, source_text: str) -> bool:
    """
    한 연결의 실재 검증: 인용(quote)이 상위 근거 원문에 실제로 있나.
    (어제 '블랙박스 가두기'와 같은 원리 — 지어낸 근거는 여기서 끊김)
    투명한 문자열 대조. 공백 무시.
    """
    if not quote or not source_text:
        return False
    q = re.sub(r"\s+", "", quote)
    s = re.sub(r"\s+", "", source_text)
    # 인용의 핵심부(앞 20자)가 원문에 있으면 실재로 봄
    key = q[:20] if len(q) >= 20 else q
    return key in s
