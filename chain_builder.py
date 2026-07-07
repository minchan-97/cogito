"""
chain_builder.py — 근거 사슬 자동 추출 (cogito_trace의 심장).

문제:
  지금까지 사슬(답변→근거B→근거C→앵커)을 사람이 수동으로 넣었다.
  이제 LLM이 자동으로 소급 추출한다.

블랙박스 가두기 (핵심):
  LLM이 근거를 '생성'하면 지어낼 수 있다. 그래서:
    1. LLM은 근거를 자료에서 '인용'만 하게 한다 (원문 그대로).
    2. 인용이 자료에 실재하는지 기계가 검증한다.
    3. 실재하는 인용만 사슬에 넣는다. 지어낸 건 버린다.
    4. 그 인용에 상위 근거(다른 문서/조항 참조)가 있으면 계속 소급.
    5. 앵커(인간보증/공리) 도달 or 소급 불가까지.

정직한 범위:
  - LLM 추출기는 주입식(llm_fn). 없으면 규칙 기반 소급(참조 표현 탐지).
  - '자동 소급'은 자료에 상위 근거가 명시돼 있을 때만 가능.
    끊기면 그 자체가 정보다("이 주장은 앵커까지 못 감").
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import re


@dataclass
class ChainLink:
    depth: int
    statement: str          # 이 마디의 주장
    quote: str = ""         # 자료에서 인용한 근거 원문
    quote_verified: bool = False   # 인용이 자료에 실재하나
    refers_to: str = ""     # 이 근거가 참조하는 상위 근거(법령명 등)
    is_anchor: bool = False
    anchor_type: str = ""   # "human" | "axiom" | ""


@dataclass
class AutoChain:
    answer: str
    links: list = field(default_factory=list)
    terminated: str = ""    # "anchor" | "broken" | "maxdepth"

    def summary(self) -> dict:
        anchor = next((l for l in self.links if l.is_anchor), None)
        all_verified = all(l.quote_verified for l in self.links if l.quote)
        return {
            "depth": len(self.links),
            "terminated": self.terminated,
            "reached_anchor": anchor is not None,
            "anchor_type": anchor.anchor_type if anchor else "",
            "all_verified": all_verified,
            "status": self._status(anchor, all_verified),
        }

    def _status(self, anchor, all_verified):
        if anchor and all_verified:
            return "grounded"      # 앵커까지 + 전부 검증
        if anchor:
            return "weak"          # 앵커 도달했으나 일부 미검증
        return "ungrounded"        # 앵커 없이 끊김


# ─── 참조 표현 탐지 (상위 근거로 소급하는 단서) ───
_REF_PATTERNS = [
    r'([가-힣]*\s*규정\s*제?\d+조)',      # 학교 안전규정 제3조
    r'(초·?중등교육법\s*시행령?)',
    r'(교육기본법|헌법)',
    r'([가-힣]+법\s*시행령?)',
    r'(교육청|교육부|장관|교육감)\s*(고시|지침|훈령)',
    r'(제\d+조)',
]

# ─── 앵커 판정 ───
_HUMAN_ANCHOR = ['결재', '서명', '승인', '의결', '재가', '교육감', '교장', '위원회 의결']
_AXIOM_ANCHOR = ['헌법', '교육기본법', '초·중등교육법', '초중등교육법', '법률', '시행령']


def _norm(t): return re.sub(r'\s+', '', t)


def detect_anchor(text: str) -> tuple:
    """텍스트가 앵커인지 판정. (is_anchor, type)."""
    t = _norm(text)
    for w in _HUMAN_ANCHOR:
        if _norm(w) in t:
            return True, "human"
    for w in _AXIOM_ANCHOR:
        if _norm(w) in t:
            return True, "axiom"
    return False, ""


def detect_reference(text: str) -> str:
    """이 근거가 참조하는 상위 근거 표현을 추출(소급 대상)."""
    for pat in _REF_PATTERNS:
        m = re.search(pat, text)
        if m:
            return m.group(0)
    return ""


class ChainBuilder:
    """근거 사슬을 자동으로 소급 구축."""

    def __init__(self, corpus: str, llm_fn: Optional[Callable] = None,
                 max_depth: int = 6):
        self.corpus = corpus
        self.corpus_norm = _norm(corpus)
        self.corpus_sents = [s.strip() for s in re.split(r'(?<=[.!?。])\s+|\n+', corpus)
                             if len(s.strip()) >= 5]
        self.llm_fn = llm_fn      # (statement, corpus) -> quote (근거 인용)
        self.max_depth = max_depth

    def _verify_quote(self, quote: str) -> bool:
        """인용이 자료에 실재하나 (블랙박스 가두기)."""
        if not quote:
            return False
        q = _norm(quote)
        key = q[:20] if len(q) >= 20 else q
        return key in self.corpus_norm

    def _find_evidence(self, statement: str) -> str:
        """
        주장의 근거를 자료에서 찾음.
        llm_fn 있으면 LLM 인용, 없으면 규칙(가장 유사 문장).
        """
        if self.llm_fn is not None:
            quote = self.llm_fn(statement, self.corpus)
            # 블랙박스 가두기: 실재 확인
            if self._verify_quote(quote):
                return quote
            return ""   # 지어낸 인용은 버림
        # 규칙 폴백: 핵심어 겹침 최대 문장
        return self._best_overlap_sent(statement)

    def _best_overlap_sent(self, statement: str) -> str:
        s_words = set(re.findall(r'[가-힣a-zA-Z0-9]{2,}', statement))
        best, best_idx = 0, -1
        for i, sent in enumerate(self.corpus_sents):
            w = set(re.findall(r'[가-힣a-zA-Z0-9]{2,}', sent))
            overlap = len(s_words & w)
            if overlap > best:
                best, best_idx = overlap, i
        if best < 1:
            return ""
        # 찾은 문장 + 다음 문장(참조가 이어지는 경우 소급 단서 확보)
        result = self.corpus_sents[best_idx]
        if best_idx + 1 < len(self.corpus_sents):
            nxt = self.corpus_sents[best_idx + 1]
            # 다음 문장이 이 주제를 잇는 참조/근거면 포함
            if detect_reference(nxt) or any(w in nxt for w in ['이는', '이', '해당', '위']):
                result = result + " " + nxt
        return result

    def build(self, answer: str) -> AutoChain:
        """답변에서 시작해 근거를 앵커까지 소급."""
        chain = AutoChain(answer=answer)
        current = answer
        seen = set()

        for depth in range(self.max_depth):
            quote = self._find_evidence(current)
            if not quote:
                chain.terminated = "broken"    # 근거 못 찾음 = 끊김
                break
            if _norm(quote) in seen:
                chain.terminated = "broken"
                break
            seen.add(_norm(quote))

            is_anchor, atype = detect_anchor(quote)
            ref = detect_reference(quote)
            verified = self._verify_quote(quote)

            chain.links.append(ChainLink(
                depth=depth, statement=current, quote=quote,
                quote_verified=verified, refers_to=ref,
                is_anchor=is_anchor, anchor_type=atype))

            if is_anchor:
                chain.terminated = "anchor"    # 앵커 도달 = 종결
                break
            if not ref:
                chain.terminated = "broken"    # 더 소급할 상위 근거 없음
                break
            # 다음 소급: 참조된 상위 근거를 새 주장으로
            current = ref
        else:
            chain.terminated = "maxdepth"

        return chain
