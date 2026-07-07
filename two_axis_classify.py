"""
two_axis_classify.py — 두 축으로 오류 '이유'까지 3분류.

오늘의 발견:
  유사도만으로는 match는 갈리지만 mismatch/hallucination이 겹친다(데이터 확인).
  → 근거 사슬 축을 더한다.

두 축:
  가로 = 기준(자료)과의 의미 유사도 (임베딩 코사인)
  세로 = 답변의 항목/근거가 자료에 실재하는가 (근거 사슬)

3분류 = 오류 이유:
  유사도 높음                     → MATCH        "자료와 일치"
  유사도 낮음 + 항목은 자료에 있음  → VALUE_ERROR  "항목은 맞으나 값이 틀림"
  유사도 낮음 + 근거 자체가 없음    → HALLUCINATION "자료에 근거 없음(지어냄)"

이유를 설명할 수 있으므로, 사용자가 취할 행동이 달라진다:
  VALUE_ERROR → "값을 확인하세요"
  HALLUCINATION → "이 내용은 자료에 없습니다"
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import re
import numpy as np


@dataclass
class Judgment:
    label: str          # MATCH | VALUE_ERROR | HALLUCINATION | CONTRADICTION
    reason: str         # 사람이 읽는 오류 이유
    similarity: float   # 가로축 값
    item_grounded: bool # 세로축: 항목이 자료에 있나
    color: str          # green / orange / red / darkred


def _norm(t: str) -> str:
    return re.sub(r"\s+", "", t)


# 극성 사전 (긍정/부정) — 부정 반전 탐지용
_NEG_WORDS = ["없다", "없음", "없으", "아니다", "아니", "않다", "않음", "않는",
              "못하", "못한", "불가", "제외", "미흡", "부재", "불충분", "결여",
              "부족하", "실패", "거부", "금지", "불허"]
_POS_WORDS = ["있다", "있음", "있으", "존재", "포함", "가능", "충분", "달성",
              "완료", "수행", "허용", "성공", "확보", "구비", "충족"]


def polarity(sentence: str) -> int:
    """문장의 극성. +1(긍정) / -1(부정) / 0(중립)."""
    s = _norm(sentence)
    neg = sum(1 for w in _NEG_WORDS if w in s)
    pos = sum(1 for w in _POS_WORDS if w in s)
    if neg > pos:
        return -1
    if pos > neg:
        return +1
    return 0


def negation_flip(doc_sent: str, ans_sent: str) -> bool:
    """
    두 문장이 '서술 부정'으로 뒤집혔나 (극성 사전이 놓치는 계사 부정).
    예: '~이다' vs '~이 아니다', '~한다' vs '~하지 않는다'.
    핵심어(명사)가 겹치는데 한쪽만 부정 표지가 있으면 반전으로 본다.
    """
    d = _norm(doc_sent)
    a = _norm(ans_sent)
    # 부정 표지 (서술어 부정)
    neg_markers = ["아니다", "아니", "아닌", "않는다", "않다", "않은", "않았",
                   "못한다", "못했", "없다", "없음", "말라", "마라"]
    d_neg = any(m in d for m in neg_markers)
    a_neg = any(m in a for m in neg_markers)
    if d_neg == a_neg:
        return False   # 둘 다 부정이거나 둘 다 긍정 → 반전 아님
    # 한쪽만 부정. 내용 핵심어가 겹치는지 확인 (같은 주제여야 반전)
    d_words = set(re.findall(r'[가-힣a-zA-Z0-9]{2,}', doc_sent))
    a_words = set(re.findall(r'[가-힣a-zA-Z0-9]{2,}', ans_sent))
    # 부정 표지 단어는 제외하고 겹침 계산
    common = d_words & a_words
    common = {w for w in common if not any(m in w for m in neg_markers)}
    # 핵심어가 2개 이상 겹치면 같은 주제로 보고 반전 판정
    return len(common) >= 2


def extract_item(sentence: str) -> str:
    """문장에서 주어(항목) 대략 추출. '~은/는/이/가' 앞부분."""
    m = re.match(r"\s*([^,]+?)(은|는|이|가|의|에서|에는)\s", sentence)
    if m:
        return m.group(1).strip()
    # 폴백: 첫 명사구 대략
    return sentence[:8]


class TwoAxisClassifier:
    def __init__(self, corpus: str, embed_fn: Optional[Callable] = None,
                 sim_threshold: float = 0.6, contradiction_sim: float = 0.55):
        """
        corpus: 자료 원문 (세로축 판정 기준)
        embed_fn: text -> vector (가로축). 없으면 유사도 축 비활성.
        sim_threshold: 이 이상이면 MATCH 후보.
        contradiction_sim: 이 이상 유사한데 극성 반대면 모순으로 봄.
        """
        self.corpus = corpus
        self.corpus_compact = _norm(corpus)
        self.embed = embed_fn
        self.sim_th = sim_threshold
        self.contra_sim = contradiction_sim
        # 자료를 문장 단위로 쪼갬 (문장별 대조용)
        self.corpus_sents = [s.strip() for s in re.split(r'(?<=[.!?。])\s+|\n+', corpus)
                             if len(s.strip()) >= 5]
        self._sent_vecs = None  # 지연 계산

    def _corpus_sent_vecs(self):
        if self._sent_vecs is None and self.embed is not None:
            self._sent_vecs = [(s, self.embed(s)) for s in self.corpus_sents]
        return self._sent_vecs or []

    def _item_in_corpus(self, item: str) -> bool:
        """항목(주어)이 자료에 실재하나 = 세로축."""
        key = _norm(item)
        key = re.sub(r"(은|는|이|가|을|를|의|에|에서|에는)$", "", key)
        if len(key) < 2:
            return False
        return key[:6] in self.corpus_compact

    def _best_matching_sent(self, sent_vec):
        """답변과 가장 유사한 자료 문장 + 그 유사도 반환."""
        best_sim, best_sent = -1.0, ""
        for s, sv in self._corpus_sent_vecs():
            sim = float(np.dot(sent_vec, sv) /
                        ((np.linalg.norm(sent_vec) * np.linalg.norm(sv)) + 1e-12))
            if sim > best_sim:
                best_sim, best_sent = sim, s
        return best_sent, best_sim

    def _subject_matches_predicate(self, sentence: str, doc_sent: str) -> Optional[bool]:
        """
        주어 교체 탐지.
        답변과 자료 문장의 서술부는 같은데(유사도 높음) 주어가 다르면 → 값 오류.
        반환: True(주어 일치) / False(주어 교체됨) / None(판정 불가)
        """
        ans_subj = _norm(extract_item(sentence))
        doc_subj = _norm(extract_item(doc_sent))
        ans_subj = re.sub(r"(은|는|이|가|을|를|의|에|에서|에는)$", "", ans_subj)
        doc_subj = re.sub(r"(은|는|이|가|을|를|의|에|에서|에는)$", "", doc_subj)
        if len(ans_subj) < 2 or len(doc_subj) < 2:
            return None
        # 답변 주어가 자료 어디에도 없으면(그 서술어의 주체가 아님) 교체로 봄
        if ans_subj[:4] not in self.corpus_compact:
            return False
        # 답변 주어와 자료 문장 주어가 다르면 교체 가능성
        if ans_subj[:4] != doc_subj[:4]:
            return False
        return True

    def classify(self, sentence: str, ref_text: str,
                 ref_vec=None, sent_vec=None) -> Judgment:
        item = extract_item(sentence)
        grounded = self._item_in_corpus(item)

        # ── 유사도 축: 자료 전체 문장 중 최대 유사도 + 가장 닮은 문장 ──
        sim = None
        best_sent = ""
        if self.embed is not None:
            sv = sent_vec if sent_vec is not None else self.embed(sentence)
            best_sent, sim = self._best_matching_sent(sv)

            # ── 부정 반전(모순) 탐지 ──
            if sim >= self.contra_sim:
                p_ans = polarity(sentence)
                p_doc = polarity(best_sent)
                sic_flip = (p_ans != 0 and p_doc != 0 and p_ans != p_doc)
                neg_flip = negation_flip(best_sent, sentence)
                if sic_flip or neg_flip:
                    return Judgment(
                        "CONTRADICTION",
                        f"자료 '{best_sent[:28]}...'를 답변이 부정으로 뒤집음 "
                        f"— 자료를 뒤집는 모순",
                        sim, grounded, "darkred")

            # ── 주어 교체 탐지 (claim 대조 통합) ──
            # 유사도 높음(서술부 같음)인데 주어가 자료의 그 주체와 다르면 값 오류
            if sim >= self.sim_th:
                subj_ok = self._subject_matches_predicate(sentence, best_sent)
                if subj_ok is False:
                    return Judgment(
                        "VALUE_ERROR",
                        f"서술 내용은 자료와 유사하나 주어가 다름 "
                        f"— 자료의 주체는 '{extract_item(best_sent)}', "
                        f"답변은 '{item}' (주어 교체 가능)",
                        sim, grounded, "orange")
                # 주어도 일치 → 진짜 일치
                return Judgment("MATCH", "자료 문장과 일치(주어·서술 모두 일치)",
                                sim, grounded, "green")

        # ── 유사도 낮음(또는 미측정) → 세로축으로 이유 구분 ──
        if grounded:
            return Judgment(
                "VALUE_ERROR",
                f"항목 '{item}'은 자료에 있으나 내용이 자료와 다름 — 값 오류 가능(확인 필요)",
                sim if sim is not None else -1, grounded, "orange")
        else:
            return Judgment(
                "HALLUCINATION",
                f"항목 '{item}'의 근거가 자료에 없음 — 지어낸 내용 가능",
                sim if sim is not None else -1, grounded, "red")
