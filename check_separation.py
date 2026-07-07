"""
check_separation.py — 벽1 확인: 일치/불일치/환각이 유사도로 실제 갈리나?

로컬에서 실제 임베딩으로 돌린다. OpenAI 키 필요.
  1. 기준 문장(자료의 참) 하나
  2. 일치/불일치/환각 문장들을 각각 여러 개
  3. 각 문장을 기준과 코사인 유사도 계산
  4. 세 부류의 유사도 분포를 히스토그램 + 통계로 출력

목적: GMM/JM 같은 분류기를 만들기 '전에',
  유사도만으로 세 부류가 갈리는지 눈으로 확인.
  갈리면 → 분류 가능. 안 갈리면 → 다른 특성 필요.

사용:
  pip install openai numpy matplotlib
  export OPENAI_API_KEY=...
  python check_separation.py
"""
from __future__ import annotations
import os
import numpy as np


# ─────────────────────────────────────────────
# 여기를 실제 데이터로 바꿔 넣으세요 (도메인에 맞게)
# ─────────────────────────────────────────────
REFERENCE = "학생 안전교육은 매 학기 시작 전에 실시하며, 화재 대피와 교통안전을 포함한다."

MATCH = [   # 기준과 같은 사실 (표현만 다름) — 유사도 높아야
    "안전교육은 학기마다 학기 초에 진행되고, 화재 대피와 교통안전을 다룬다.",
    "매 학기 시작 전 학생 대상 안전교육을 실시하고 화재·교통 안전을 가르친다.",
    "학기 초마다 안전교육을 하며 내용은 화재 대피와 교통안전이다.",
    "안전교육은 학기 개시 전 실시되며 화재와 교통 안전을 포함한다.",
    "학생들에게 매 학기 초 화재 대피·교통안전 중심의 안전교육을 한다.",
]

MISMATCH = [   # 주제는 맞지만 사실이 다름 — 유사도 중간
    "안전교육은 매달 한 번 실시하며 응급처치를 포함한다.",
    "안전교육은 연 1회 방학 중에 진행된다.",
    "안전교육은 격주로 실시하고 성교육을 중심으로 한다.",
    "안전교육은 교장 재량으로 부정기적으로 실시한다.",
    "안전교육은 학년말에 몰아서 하며 사이버 안전만 다룬다.",
]

HALLUCINATION = [   # 자료와 무관/터무니없음 — 유사도 낮아야
    "안전교육은 대통령 승인을 받아야 시행할 수 있다.",
    "안전교육 자료는 반드시 금색 잉크로 작성해야 한다.",
    "안전교육은 2019년 국제우주협약에 따라 도입되었다.",
    "안전교육 이수 시 학생에게 백만원의 상금을 지급한다.",
    "안전교육은 헬리콥터를 타고 공중에서 진행한다.",
]


def get_embeddings(texts, model="text-embedding-3-small"):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in resp.data])


def cos_sim(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def main():
    all_texts = [REFERENCE] + MATCH + MISMATCH + HALLUCINATION
    print(f"임베딩 계산 중... ({len(all_texts)}개 문장)")
    embs = get_embeddings(all_texts)
    ref = embs[0]

    i = 1
    groups = {}
    for name, lst in [("match", MATCH), ("mismatch", MISMATCH), ("hallucination", HALLUCINATION)]:
        sims = [cos_sim(ref, embs[i + k]) for k in range(len(lst))]
        groups[name] = sims
        i += len(lst)

    # 통계 출력
    print("\n=== 기준 문장과의 코사인 유사도 ===")
    print(f"기준: {REFERENCE[:40]}...\n")
    for name, sims in groups.items():
        arr = np.array(sims)
        print(f"[{name:14}] 평균 {arr.mean():.3f}  최소 {arr.min():.3f}  "
              f"최대 {arr.max():.3f}  표준편차 {arr.std():.3f}")
        print(f"                 값: {[round(s,3) for s in sims]}")

    # 갈리는지 판정: 그룹 간 평균이 겹치나?
    print("\n=== 분리 판정 ===")
    m = np.mean(groups["match"]);  mi = np.mean(groups["mismatch"]);  h = np.mean(groups["hallucination"])
    m_min = min(groups["match"]);  mi_max = max(groups["mismatch"])
    mi_min = min(groups["mismatch"]);  h_max = max(groups["hallucination"])

    print(f"평균: match {m:.3f} > mismatch {mi:.3f} > hallucination {h:.3f}")
    gap1 = m_min - mi_max   # match 최저 vs mismatch 최고 (겹치면 음수)
    gap2 = mi_min - h_max   # mismatch 최저 vs hallucination 최고
    print(f"match/mismatch 경계 간격: {gap1:+.3f} ({'갈림' if gap1 > 0 else '겹침'})")
    print(f"mismatch/halluc 경계 간격: {gap2:+.3f} ({'갈림' if gap2 > 0 else '겹침'})")

    if gap1 > 0 and gap2 > 0:
        print("\n✅ 세 부류가 유사도로 깔끔히 갈림 → GMM/임계값 분류 가능")
    elif m > mi > h:
        print("\n△ 평균은 순서대로지만 일부 겹침 → 유사도만으론 부족, "
              "추가 특성(분기 깊이 등) 필요할 수 있음")
    else:
        print("\n❌ 순서가 안 나옴 → 유사도만으로는 구분 어려움. 접근 재고.")

    # 히스토그램 (matplotlib 있으면)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        for name, sims in groups.items():
            plt.hist(sims, bins=15, alpha=0.5, label=name, range=(0, 1))
        plt.xlabel("cosine similarity to reference")
        plt.ylabel("count")
        plt.legend(); plt.title("일치/불일치/환각 유사도 분포")
        plt.tight_layout()
        plt.savefig("separation_hist.png", dpi=120)
        print("\n히스토그램 저장: separation_hist.png")
    except Exception as e:
        print(f"\n(히스토그램 건너뜀: {e})")


if __name__ == "__main__":
    main()
