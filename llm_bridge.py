"""
llm_bridge.py — 사고 구조에 실제 LLM 연결.

세 가지 연결:
  1. choose_fn: 분기점에서 LLM이 어느 판단으로 갈지 선택
  2. answer_fn: 지나온 경로를 바탕으로 최종 답변 생성
  3. detect_feedback: 사람의 대화 반응(긍정/부정) 감지

로컬(Ollama)/GPT 둘 다 지원 — base_url만 다름.
  GPT:    OpenAI(api_key=...)
  Ollama: OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
"""
from __future__ import annotations
from typing import Callable, List, Optional
import re


def make_client(api_key: str = "", local: bool = False):
    from openai import OpenAI
    if local:
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    key = "".join(ch for ch in api_key.strip() if ord(ch) < 128)
    return OpenAI(api_key=key)


def make_choose_fn(client, model: str = "gpt-4o-mini"):
    """
    분기점에서 LLM이 판단을 선택.
    NM 전이 확률을 참고로 주되, 최종 선택은 LLM이 (NM=지도, LLM=운전자).
    """
    def choose_fn(node, candidates, candidate_nodes, probs, context):
        # 후보 판단들을 LLM에게 제시
        options = "\n".join(
            f"  {i+1}. {cn.prompt} (현재 성향 {p:.0%})"
            for i, (cn, p) in enumerate(zip(candidate_nodes, probs)))
        prompt = (
            f"맥락: {context}\n\n"
            f"현재 판단 지점: {node.prompt}\n\n"
            f"다음 중 어느 판단으로 진행할지 하나만 고르세요:\n{options}\n\n"
            f"번호만 답하세요 (1~{len(candidates)}). 현재 성향은 참고만 하고, "
            f"맥락에 가장 맞는 판단을 고르세요.")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=10)
            text = resp.choices[0].message.content.strip()
            m = re.search(r'\d+', text)
            if m:
                idx = int(m.group()) - 1
                if 0 <= idx < len(candidates):
                    return candidates[idx]
        except Exception:
            pass
        # 실패 시 확률 최대 후보
        return candidates[probs.index(max(probs))]
    return choose_fn


def make_answer_fn(client, node_map, model: str = "gpt-4o-mini"):
    """
    지나온 판단 경로의 directive(구체적 실행 지시)를 순서대로 실행.
    절차는 지키되, 답변은 자연스러운 대화체로.
    반환: (답변, 근거자료 리스트) — 근거는 감사 기록용.
    """
    def answer_fn(path, context):
        directives = []
        for nid in path:
            node = node_map.get(nid)
            if node and node.directive:
                directives.append(node.directive)

        if not directives:
            proc = ""
        else:
            proc = "다음 판단 절차를 '내부적으로' 따르세요:\n" + \
                   "\n".join(f"  - {d}" for d in directives)

        prompt = (
            f"질문: {context}\n\n"
            f"{proc}\n\n"
            f"위 절차는 당신의 사고 과정입니다. 절차대로 판단하되, "
            f"답변은 '1. 2. 3.' 같은 번호 나열이 아니라 자연스럽고 따뜻한 "
            f"대화체로 쓰세요. 사람에게 말하듯 편하게. "
            f"만약 특정 자료·근거를 참고했다면, 답변 맨 끝에 별도 줄로 "
            f"'[근거: ...]' 형식으로 간단히 적으세요. 없으면 생략하세요.")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6, max_tokens=500)
            text = resp.choices[0].message.content.strip()
            # 근거 추출 (감사 기록용)
            sources = []
            import re
            for m in re.finditer(r'\[근거:\s*([^\]]+)\]', text):
                sources.append(m.group(1).strip())
            # 근거 표기는 답변에서 떼어내 따로 (답변은 깔끔하게)
            clean = re.sub(r'\[근거:\s*[^\]]+\]', '', text).strip()
            return clean, sources
        except Exception as e:
            return f"(답변 생성 실패: {e})", []
    return answer_fn


# ── 대화 반응 감지 ──
_NEGATIVE = ['아니', '틀렸', '아냐', '아닌', '잘못', '다시', '이상해', '별로',
             '아니오', '노', 'no', 'wrong', '그게 아니', '안 맞', '틀림']
_POSITIVE = ['맞아', '맞다', '좋아', '좋다', '정확', '그래', '응', '옳', '완벽',
             '고마', '훌륭', 'yes', 'good', '맞습니다', '좋습니다']


def detect_feedback(user_message: str,
                    client=None, model: str = "gpt-4o-mini") -> Optional[float]:
    """
    사람의 반응에서 긍정/부정 감지.
    1차: 명백한 표현 (규칙)
    2차: 애매하면 LLM (client 있을 때)
    반환: +1(긍정) / -1(부정) / None(중립·불명)
    """
    msg = user_message.strip().lower()

    # 1차: 명백한 표현
    neg = any(w in msg for w in _NEGATIVE)
    pos = any(w in msg for w in _POSITIVE)
    if neg and not pos:
        return -1.0
    if pos and not neg:
        return 1.0

    # 2차: 애매하면 LLM (선택)
    if client is not None:
        try:
            prompt = (
                f"사용자 반응: \"{user_message}\"\n\n"
                f"이 반응이 직전 답변에 만족(긍정)인지 불만족(부정)인지 중립인지 판단하세요.\n"
                f"positive / negative / neutral 중 하나만 답하세요.")
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=10)
            t = resp.choices[0].message.content.strip().lower()
            if "positive" in t:
                return 1.0
            if "negative" in t:
                return -1.0
        except Exception:
            pass
    return None   # 중립·불명 → 학습 안 함


# ── 정보 제공 감지 (사용자가 "기억해둬" 하는 것) ──
def detect_shared_info(user_message: str,
                       client=None, model: str = "gpt-4o-mini") -> str:
    """
    사용자가 자기 정보를 알려주는지 감지 → 기억할 내용 반환.
    "내 이름은 민찬기야" → "사용자 이름은 민찬기"
    질문이거나 정보가 아니면 "" 반환.
    """
    msg = user_message.strip()
    # 1차: 명백한 패턴 (규칙)
    import re
    patterns = [
        (r'내?\s*이름은?\s*([가-힣a-zA-Z]+)', lambda m: f"사용자 이름은 {m.group(1)}"),
        (r'나는\s*(.+?)(를|을)\s*좋아', lambda m: f"사용자는 {m.group(1)}을(를) 좋아함"),
        (r'나는\s*(.+?)(이|가)\s*싫', lambda m: f"사용자는 {m.group(1)}을(를) 싫어함"),
        (r'나는\s*(.+?)(이야|야|입니다|이에요|예요)$', lambda m: f"사용자: {m.group(1)}"),
        (r'내?\s*(취미|직업|나이|사는 곳|고향)은?\s*(.+)', lambda m: f"사용자 {m.group(1)}: {m.group(2)}"),
    ]
    for pat, fn in patterns:
        m = re.search(pat, msg)
        if m:
            return fn(m).strip()

    # 2차: LLM 판별 (애매할 때)
    if client is not None:
        try:
            prompt = (
                f"사용자 메시지: \"{user_message}\"\n\n"
                f"이 메시지에서 사용자가 자신에 대한 정보(이름, 선호, 직업 등)를 "
                f"알려주고 있나요? 있으면 '기억할 사실'을 한 문장으로, "
                f"없으면(질문이거나 일반 대화면) 'NONE'만 답하세요.\n"
                f"예: '내 이름은 민찬기야' → '사용자 이름은 민찬기'\n"
                f"예: '오늘 날씨 어때?' → 'NONE'")
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=40)
            t = resp.choices[0].message.content.strip()
            if t and "NONE" not in t.upper():
                return t
        except Exception:
            pass
    return ""
