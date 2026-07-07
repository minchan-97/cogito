"""
app_anchor.py — 근거 앵커 프로브 (Streamlit)

질문 + 답변 + 근거 앵커들을 넣으면:
  - 각 근거 ↔ 답변 유사도 (답이 근거를 따르나)
  - 근거 ↔ 근거 거리 (근거들 일관되나)
  - 확 튀는 지점 탐지 + 판정

실행:
  pip install streamlit openai numpy plotly
  streamlit run app_anchor.py
"""
import streamlit as st
import numpy as np

st.set_page_config(page_title="근거 앵커 프로브", layout="wide")
st.title("⚓ 근거 앵커 프로브 — 답이 근거를 따르나")
st.caption("근거를 앵커로 심고, 유사도로 '확 튀는' 지점을 잡는다. "
           "LLM 내부는 못 봐도, 근거 조작으로 답의 반응을 밖에서 추적.")

with st.sidebar:
    api_key = st.text_input("OpenAI Key", type="password")
    spike_th = st.slider("튐 민감도 (표준편차 배수)", 1.0, 3.0, 1.5, 0.1)

st.subheader("입력")
question = st.text_input("질문", value="안전교육 주기는?")
answer = st.text_area("답변", height=70,
    value="안전교육은 매 학기 초에 화재대피와 교통안전을 포함해 실시한다")
anchors_txt = st.text_area("근거 앵커 (한 줄에 하나)", height=110,
    value="안전교육은 매 학기 실시한다\n화재대피 훈련을 포함한다\n교통안전 교육을 포함한다")


def cos(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def get_embeddings(texts, key):
    from openai import OpenAI
    k = "".join(ch for ch in key.strip() if ord(ch) < 128)
    client = OpenAI(api_key=k)
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in resp.data])


if st.button("프로브 실행", type="primary"):
    if not api_key:
        st.error("OpenAI Key 필요")
        st.stop()
    anchors = [a.strip() for a in anchors_txt.split("\n") if a.strip()]
    if len(anchors) < 2:
        st.warning("근거를 2개 이상 넣으세요")
        st.stop()

    with st.spinner("임베딩 계산 중..."):
        try:
            embs = get_embeddings([answer] + anchors, api_key)
        except Exception as e:
            st.error(f"임베딩 실패: {e}")
            st.stop()

    av = embs[0]
    anchor_vecs = embs[1:]

    # 근거-답변 유사도
    aa_sims = [cos(av, v) for v in anchor_vecs]
    # 근거-근거 거리
    dists, pair_labels = [], []
    for i in range(len(anchors)):
        for j in range(i+1, len(anchors)):
            dists.append(1 - cos(anchor_vecs[i], anchor_vecs[j]))
            pair_labels.append(f"근거{i+1}~근거{j+1}")

    def spikes(values, labels):
        if len(values) < 2: return []
        arr = np.array(values); mu, sd = arr.mean(), arr.std()
        if sd < 1e-9: return []
        return [{"label": lb, "value": round(v,3), "z": round((v-mu)/sd,2)}
                for v, lb in zip(values, labels) if abs((v-mu)/sd) >= spike_th]

    st.subheader("1. 근거 ↔ 답변 유사도 (답이 각 근거를 따르나)")
    sp1 = spikes(aa_sims, [f"근거{i+1}" for i in range(len(anchors))])
    sp1_labels = {s["label"] for s in sp1}
    for i, (a, s) in enumerate(zip(anchors, aa_sims)):
        spike_mark = " 🔴 튐!" if f"근거{i+1}" in sp1_labels else ""
        bar = "█" * int(s * 30)
        st.markdown(f"**근거{i+1}** ({s:.3f}){spike_mark}  `{bar}`  {a[:40]}")

    st.subheader("2. 근거 ↔ 근거 거리 (근거들끼리 일관되나)")
    sp2 = spikes(dists, pair_labels)
    sp2_labels = {s["label"] for s in sp2}
    for lb, d in zip(pair_labels, dists):
        spike_mark = " 🔴 튐!" if lb in sp2_labels else ""
        st.markdown(f"**{lb}**: 거리 {d:.3f}{spike_mark}")

    # 히트맵
    try:
        import plotly.graph_objects as go
        n = len(anchors)
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                mat[i][j] = cos(anchor_vecs[i], anchor_vecs[j])
        fig = go.Figure(data=go.Heatmap(
            z=mat, x=[f"근거{i+1}" for i in range(n)],
            y=[f"근거{i+1}" for i in range(n)], colorscale="Viridis",
            zmin=0, zmax=1))
        fig.update_layout(title="근거 간 유사도 (어두운 곳=이질적 근거)", height=350)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    # 판정
    st.subheader("3. 판정")
    msgs = []
    if max(aa_sims) < 0.4:
        msgs.append("⚠️ 답변이 어떤 근거와도 가깝지 않음 — 근거 무시(환각) 의심")
    if sp2:
        msgs.append("⚠️ 근거들 사이 거리가 튐 — 모순되거나 이질적 근거 섞임")
    if sp1:
        msgs.append("⚠️ 특정 근거만 답과 동떨어짐 — 그 근거 미반영 가능")
    lowest = min(range(len(aa_sims)), key=lambda i: aa_sims[i])
    if aa_sims[lowest] < 0.3:
        msgs.append(f"⚠️ 근거{lowest+1}가 답에 거의 반영 안 됨 (유사도 {aa_sims[lowest]:.3f})")

    if msgs:
        for m in msgs:
            st.warning(m)
    else:
        st.success("✅ 근거들이 답변에 고르게 반영됨 (안정)")

    st.caption("근거를 뺐다 넣었다 하며 답 변화를 보려면 answer_fn(LLM 재생성)이 필요. "
               "이 앱은 정적 분석(근거-답변, 근거-근거 유사도)까지. "
               "튀는 지점 = 답이 그 근거를 안 따르거나, 근거끼리 안 맞는 신호.")
