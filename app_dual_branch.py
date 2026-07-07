"""
app_dual_branch.py — 근거 분기 + 논리 분기 (Streamlit)

추론 사슬(단계들)을 넣으면:
  - 근거 분기: 각 단계가 자료에 실재하나
  - 논리 분기: 인접 단계 연결이 성립하나 (약하면 논리 비약)
  - 흔들리는 지점 표시

핵심: 근거는 다 참인데 논리가 무너지는 환각을 논리 분기에서 잡는다.

실행:
  pip install streamlit openai numpy
  streamlit run app_dual_branch.py
"""
import streamlit as st
import numpy as np

st.set_page_config(page_title="근거+논리 분기", layout="wide")
st.title("🌿 근거 분기 + 논리 분기")
st.caption("추론 사슬을 두 유형으로 검증한다. 근거 분기(사실이 실재하나) + "
           "논리 분기(연결이 성립하나). 근거는 참인데 논리가 무너지는 환각을 잡는다.")

with st.sidebar:
    api_key = st.text_input("OpenAI Key", type="password")
    gap_th = st.slider("논리 연결 최소 강도", 0.2, 0.8, 0.45, 0.05,
                       help="인접 단계 유사도가 이 밑이면 논리 비약으로 의심")

st.subheader("자료 (근거 실재 확인용, 선택)")
corpus = st.text_area("자료", height=70,
    value="안전교육은 매 학기 실시한다. 화재 대피와 교통안전을 포함한다.")

st.subheader("추론 사슬 (한 줄에 한 단계, 순서대로)")
steps_txt = st.text_area("단계들", height=130,
    value="비가 온다\n땅이 젖는다\n그러므로 우산은 필요 없다")


def cos(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def get_embeddings(texts, key):
    from openai import OpenAI
    k = "".join(ch for ch in key.strip() if ord(ch) < 128)
    client = OpenAI(api_key=k)
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in resp.data])


if st.button("분기 분석", type="primary"):
    if not api_key:
        st.error("OpenAI Key 필요")
        st.stop()
    steps = [s.strip() for s in steps_txt.split("\n") if s.strip()]
    if len(steps) < 2:
        st.warning("단계를 2개 이상 넣으세요")
        st.stop()

    with st.spinner("임베딩 계산 중..."):
        try:
            embs = get_embeddings(steps, api_key)
        except Exception as e:
            st.error(f"임베딩 실패: {e}")
            st.stop()

    import re
    corpus_norm = re.sub(r'\s+', '', corpus)

    # 근거 분기: 각 단계가 자료에 실재하나
    st.subheader("1. 근거 분기 (각 단계가 자료에 실재하나)")
    for i, s in enumerate(steps):
        words = set(re.findall(r'[가-힣a-zA-Z0-9]{2,}', s))
        if corpus.strip() and words:
            hit = sum(1 for w in words if w in corpus_norm) / len(words)
            if hit < 0.3:
                st.markdown(f"🔴 **단계{i+1}** (자료 실재 {hit:.2f}) {s} — 자료에 희박")
            else:
                st.markdown(f"✓ 단계{i+1} (자료 실재 {hit:.2f}) {s}")
        else:
            st.markdown(f"– 단계{i+1} {s} (자료 없음, 근거 검증 생략)")

    # 논리 분기: 인접 단계 연결 강도
    st.subheader("2. 논리 분기 (단계 연결이 성립하나)")
    gaps = [cos(embs[i], embs[i+1]) for i in range(len(steps)-1)]
    arr = np.array(gaps)
    mu, sd = arr.mean(), arr.std()

    shaky_logic = []
    for i, link in enumerate(gaps):
        z = (link - mu) / sd if sd > 1e-9 else 0
        too_weak = link < gap_th
        spike = abs(z) >= 1.5
        shaky = too_weak or spike
        if shaky:
            shaky_logic.append(i)
        mark = "🔴" if shaky else "✓"
        note = ""
        if too_weak: note += " 연결 약함(논리 비약)"
        if spike: note += f" 강도 튐(z={z:.1f})"
        st.markdown(f"{mark} **[{steps[i][:22]}] → [{steps[i+1][:22]}]** "
                    f"연결 {link:.3f}{note}")

    # 연결 강도 막대 그래프
    try:
        import plotly.graph_objects as go
        colors = ["#c5221f" if i in shaky_logic else "#1a73e8" for i in range(len(gaps))]
        fig = go.Figure(data=go.Bar(
            x=[f"{i+1}→{i+2}" for i in range(len(gaps))],
            y=gaps, marker_color=colors))
        fig.add_hline(y=gap_th, line_dash="dash", line_color="red",
                      annotation_text="논리 비약 경계")
        fig.update_layout(title="단계 간 연결 강도 (빨강=흔들림)",
                          yaxis_title="연결 강도", height=350, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    # 판정
    st.subheader("3. 판정")
    ev_shaky = []
    if corpus.strip():
        for i, s in enumerate(steps):
            words = set(re.findall(r'[가-힣a-zA-Z0-9]{2,}', s))
            if words and sum(1 for w in words if w in corpus_norm)/len(words) < 0.3:
                ev_shaky.append(i)

    if not ev_shaky and not shaky_logic:
        st.success("✅ 근거·논리 모두 안정 — 사슬이 고르게 이어짐")
    else:
        if ev_shaky:
            st.warning(f"근거 흔들림: 단계 {[i+1 for i in ev_shaky]} — 자료에 없는 단계")
        if shaky_logic:
            st.warning(f"논리 흔들림: 연결 {[f'{i+1}→{i+2}' for i in shaky_logic]} — 추론 비약")
        if shaky_logic and not ev_shaky:
            st.error("★ 근거는 참인데 논리가 무너짐 — 근거검증으론 못 잡는 환각. "
                     "이게 이 도구의 핵심 자리.")

    st.caption("논리 분기는 '인접 단계가 의미적으로 이어지나'로 근사한다. "
               "연결이 확 약해지는 지점 = 논리 비약. 실제 임베딩이라야 인과 연결을 제대로 본다.")

