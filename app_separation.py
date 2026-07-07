"""
app_separation.py — 일치/불일치/환각이 유사도로 갈리나? (Streamlit)

브라우저에서:
  1. OpenAI 키 입력
  2. 기준 문장 + 일치/불일치/환각 문장들 편집
  3. 임베딩 → 코사인 유사도 → 히스토그램 + 분리 판정

실행:
  pip install streamlit openai numpy plotly
  streamlit run app_separation.py
"""
import streamlit as st
import numpy as np

st.set_page_config(page_title="유사도 분리 확인", layout="wide")
st.title("🔬 일치 / 불일치 / 환각 — 유사도로 갈리나?")
st.caption("근거 사슬 분기점의 유사도로 세 부류를 구분할 수 있는지 먼저 확인한다. "
           "갈리면 GMM/임계값 분류 가능, 안 갈리면 다른 특성이 필요.")

with st.sidebar:
    st.markdown("### 설정")
    api_key = st.text_input("OpenAI Key", type="password")
    model = st.selectbox("임베딩 모델",
                         ["text-embedding-3-small", "text-embedding-3-large"])

# ── 기본 예시 (도메인에 맞게 바꾸세요) ──
DEFAULTS = {
    "reference": "학생 안전교육은 매 학기 시작 전에 실시하며, 화재 대피와 교통안전을 포함한다.",
    "match": "안전교육은 학기마다 학기 초에 진행되고, 화재 대피와 교통안전을 다룬다.\n"
             "매 학기 시작 전 학생 대상 안전교육을 실시하고 화재·교통 안전을 가르친다.\n"
             "학기 초마다 안전교육을 하며 내용은 화재 대피와 교통안전이다.\n"
             "안전교육은 학기 개시 전 실시되며 화재와 교통 안전을 포함한다.\n"
             "학생들에게 매 학기 초 화재 대피·교통안전 중심의 안전교육을 한다.",
    "mismatch": "안전교육은 매달 한 번 실시하며 응급처치를 포함한다.\n"
                "안전교육은 연 1회 방학 중에 진행된다.\n"
                "안전교육은 격주로 실시하고 성교육을 중심으로 한다.\n"
                "안전교육은 교장 재량으로 부정기적으로 실시한다.\n"
                "안전교육은 학년말에 몰아서 하며 사이버 안전만 다룬다.",
    "hallucination": "안전교육은 대통령 승인을 받아야 시행할 수 있다.\n"
                     "안전교육 자료는 반드시 금색 잉크로 작성해야 한다.\n"
                     "안전교육은 2019년 국제우주협약에 따라 도입되었다.\n"
                     "안전교육 이수 시 학생에게 백만원의 상금을 지급한다.\n"
                     "안전교육은 헬리콥터를 타고 공중에서 진행한다.",
}

st.subheader("1. 문장 입력 (도메인에 맞게 수정)")
ref = st.text_input("기준 문장 (자료의 참)", value=DEFAULTS["reference"])
c1, c2, c3 = st.columns(3)
with c1:
    match_txt = st.text_area("✅ 일치 문장 (한 줄에 하나)", value=DEFAULTS["match"], height=180)
with c2:
    mismatch_txt = st.text_area("△ 불일치 문장 (사실 틀림)", value=DEFAULTS["mismatch"], height=180)
with c3:
    halluc_txt = st.text_area("❌ 환각 문장 (터무니없음)", value=DEFAULTS["hallucination"], height=180)


def cos_sim(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def get_embeddings(texts, model, key):
    from openai import OpenAI
    client = OpenAI(api_key=key)
    resp = client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in resp.data])


if st.button("2. 유사도 계산 + 분리 확인", type="primary"):
    if not api_key:
        st.error("OpenAI Key를 입력하세요.")
        st.stop()

    match = [s.strip() for s in match_txt.split("\n") if s.strip()]
    mismatch = [s.strip() for s in mismatch_txt.split("\n") if s.strip()]
    halluc = [s.strip() for s in halluc_txt.split("\n") if s.strip()]
    all_texts = [ref] + match + mismatch + halluc

    with st.spinner(f"임베딩 계산 중... ({len(all_texts)}문장)"):
        try:
            embs = get_embeddings(all_texts, model, api_key)
        except Exception as e:
            st.error(f"임베딩 실패: {e}")
            st.stop()

    refv = embs[0]
    groups, i = {}, 1
    for name, lst in [("match", match), ("mismatch", mismatch), ("hallucination", halluc)]:
        groups[name] = [cos_sim(refv, embs[i + k]) for k in range(len(lst))]
        i += len(lst)

    # ── 통계 표 ──
    st.subheader("3. 결과")
    colors = {"match": "#137333", "mismatch": "#e37400", "hallucination": "#c5221f"}
    cols = st.columns(3)
    for col, (name, sims) in zip(cols, groups.items()):
        arr = np.array(sims)
        with col:
            st.markdown(f"**{name}**")
            st.metric("평균 유사도", f"{arr.mean():.3f}",
                      help=f"최소 {arr.min():.3f} / 최대 {arr.max():.3f} / 표준편차 {arr.std():.3f}")

    # ── 히스토그램 (plotly) ──
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        for name, sims in groups.items():
            fig.add_trace(go.Histogram(x=sims, name=name, opacity=0.6,
                                       marker_color=colors[name], xbins=dict(start=0, end=1, size=0.05)))
        fig.update_layout(barmode="overlay", xaxis_title="기준과의 코사인 유사도",
                          yaxis_title="문장 수", height=400,
                          title="세 부류의 유사도 분포 (겹치면 구분 어려움)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"히스토그램 건너뜀: {e}")
        # 폴백: 값 나열
        for name, sims in groups.items():
            st.write(f"{name}: {[round(s,3) for s in sims]}")

    # ── 분리 판정 ──
    st.subheader("4. 분리 판정")
    m, mi, h = np.mean(groups["match"]), np.mean(groups["mismatch"]), np.mean(groups["hallucination"])
    gap1 = min(groups["match"]) - max(groups["mismatch"])
    gap2 = min(groups["mismatch"]) - max(groups["hallucination"])

    st.write(f"평균 순서: match **{m:.3f}** > mismatch **{mi:.3f}** > hallucination **{h:.3f}**")
    cc1, cc2 = st.columns(2)
    cc1.metric("match / mismatch 경계", f"{gap1:+.3f}", "갈림" if gap1 > 0 else "겹침")
    cc2.metric("mismatch / halluc 경계", f"{gap2:+.3f}", "갈림" if gap2 > 0 else "겹침")

    if gap1 > 0 and gap2 > 0:
        st.success("✅ 세 부류가 유사도로 깔끔히 갈립니다 → GMM/임계값 분류 가능. "
                   "이 방향이 데이터로 뒷받침됩니다.")
    elif m > mi > h:
        st.warning("△ 평균은 순서대로지만 일부 겹칩니다 → 유사도만으론 부족. "
                   "분기 깊이 등 추가 특성이 필요할 수 있어요.")
    else:
        st.error("❌ 순서가 안 나옵니다 → 유사도만으로는 구분이 어렵습니다. 접근을 재고하세요.")

    st.caption("겹침이 나와도 실패가 아니라 정보예요 — '유사도 하나로는 부족하다'를 "
               "데이터로 안 것. 그게 다음 특성을 찾는 출발점이 됩니다.")
