"""
triangle_experiment.py — 질문-자료-답변 삼각형 뒤틀림 실험.

세 벡터: q(질문) d(자료) a(답변)
세 관계:
  QD = sim(q,d)  자료가 질문에 관련있나 (Context Relevance)
  QA = sim(q,a)  답이 질문에 맞나     (Answer Relevance)
  DA = sim(d,a)  답이 자료에 근거하나  (Faithfulness)

가설: 오류 유형마다 (QD, QA, DA) 패턴이 다르다.
  - 좋은 답변:        QD↑ QA↑ DA↑
  - 자료에 답 없음:    QD↓ (자료가 질문과 무관)
  - 동문서답:          QA↓ (답이 질문과 무관)
  - 그럴듯한 환각:      QA↑ DA↓ (질문엔 맞으나 자료 근거 없음)
  - 자료 무시:         QD↑ DA↓ (자료엔 있는데 답이 안 봄)

이 실험은 '삼각형이 오류 유형을 실제로 가르나'를 확인한다.
실제 임베딩(OpenAI)으로 로컬에서 돌린다.
"""
import os
import numpy as np


def cos(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def get_embeddings(texts, model="text-embedding-3-small"):
    from openai import OpenAI
    key = "".join(ch for ch in os.environ.get("OPENAI_API_KEY", "").strip() if ord(ch) < 128)
    client = OpenAI(api_key=key)
    resp = client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in resp.data])


# ─── 실험 케이스: (질문, 자료, 답변, 기대 유형) ───
CASES = [
    ("세징야는 어느 팀 소속인가?",
     "세징야는 대구fc의 주장이다.",
     "세징야는 대구fc 소속이다.",
     "좋은 답변"),

    ("세징야는 어느 팀 소속인가?",
     "세징야는 대구fc의 주장이다.",
     "세징야는 fc서울 소속이다.",
     "값 오류(팀 틀림)"),

    ("세징야는 어느 팀 소속인가?",
     "세징야는 대구fc의 주장이다.",
     "세징야는 대구fc 주장이 아니다.",
     "모순(자료 반대)"),

    ("세징야는 어느 팀 소속인가?",
     "세징야는 대구fc의 주장이다.",
     "치즈버거는 대구fc 주장이다.",
     "주어 교체"),

    ("세징야의 연봉은 얼마인가?",
     "세징야는 대구fc의 주장이다.",
     "세징야의 연봉은 10억원이다.",
     "자료에 답 없음(환각)"),

    ("세징야는 어느 팀 소속인가?",
     "세징야는 대구fc의 주장이다.",
     "오늘 날씨는 맑고 화창하다.",
     "동문서답"),
]


def main():
    print("임베딩 계산 중...\n")
    # 모든 텍스트를 한 번에 임베딩
    all_texts = []
    for q, d, a, _ in CASES:
        all_texts += [q, d, a]
    embs = get_embeddings(all_texts)

    print(f"{'유형':<20} {'QD(자료-질문)':<14} {'QA(답-질문)':<14} {'DA(답-자료)':<14}")
    print("-" * 62)
    rows = []
    for i, (q, d, a, kind) in enumerate(CASES):
        qv, dv, av = embs[i*3], embs[i*3+1], embs[i*3+2]
        QD, QA, DA = cos(qv, dv), cos(qv, av), cos(dv, av)
        rows.append((kind, QD, QA, DA))
        print(f"{kind:<20} {QD:<14.3f} {QA:<14.3f} {DA:<14.3f}")

    print("\n=== 뒤틀림 분석 ===")
    # 좋은 답변을 기준으로, 각 오류가 어느 축에서 벗어나나
    good = rows[0]
    print(f"기준(좋은 답변): QD={good[1]:.3f} QA={good[2]:.3f} DA={good[3]:.3f}\n")
    for kind, QD, QA, DA in rows[1:]:
        drops = []
        if QD < good[1] - 0.1: drops.append(f"QD↓({QD:.2f})")
        if QA < good[2] - 0.1: drops.append(f"QA↓({QA:.2f})")
        if DA < good[3] - 0.1: drops.append(f"DA↓({DA:.2f})")
        signature = ", ".join(drops) if drops else "뚜렷한 하락 없음"
        print(f"  {kind:<20} → {signature}")

    print("\n판정 기준:")
    print("  QD↓ = 자료가 질문과 무관(자료에 답 없음)")
    print("  QA↓ = 답이 질문과 무관(동문서답)")
    print("  DA↓ = 답이 자료 근거 없음(환각/모순)")
    print("\n각 오류가 서로 다른 축에서 떨어지면 → 삼각형으로 오류 유형 구분 가능")


if __name__ == "__main__":
    main()
