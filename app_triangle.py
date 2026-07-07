"""
app_triangle.py — 질문-자료-답변 삼각형 뒤틀림 (Streamlit)

세 관계(QD/QA/DA)를 재서 오류 유형이 축별로 갈리는지 본다.

실행:
  pip install streamlit openai numpy plotly
  streamlit run app_triangle.py
"""
import streamlit as st
import numpy as np

st.set_page_config(page_title="Q-D-A 삼각형", layout="wide")
st.title("△ 질문·자료·답변 삼각형 뒤틀림")
st.caption("세 관계로 오류 유형을 본다: "
           "QD(자료가 질문에 관련) · QA(답이 질문에 맞음) · DA(답이 자료에 근거). "
           "어느 축이 무너지나로 오류 종류가 갈린다.")

with st.sidebar:
    api_key = st.text_input("OpenAI Key", type="password")

st.subheader("케이스 (질문 / 자료 / 답변)")
DEFAULT = """세징야는 어느 팀 소속인가?|세징야는 대구fc의 주장이다.|세징야는 대구fc 소속이다.|좋은 답변
세징야는 어느 팀 소속인가?|세징야는 대구fc의 주장이다.|세징야는 fc서울 소속이다.|값오류
세징야는 어느 팀 소속인가?|세징야는 대구fc의 주장이다.|세징야는 대구fc 주장이 아니다.|모순
세징야는 어느 팀 소속인가?|세징야는 대구fc의 주장이다.|치즈버거는 대구fc 주장이다.|주어교체
세징야의 연봉은 얼마인가?|세징야는 대구fc의 주장이다.|세징야 연봉은 10억원이다.|자료에답없음
세징야는 어느 팀 소속인가?|세징야는 대구fc의 주장이다.|오늘 날씨는 맑다.|동문서답"""

txt = st.text_area("한 줄에 하나: 질문|자료|답변|유형이름", value=DEFAULT, height=170)


def cos(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def get_embeddings(texts, key):
    from openai import OpenAI
    k = "".join(ch for ch in key.strip() if ord(ch) < 128)
    client = OpenAI(api_key=k)
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in resp.data])


if st.button("삼각형 측정", type="primary"):
    if not api_key:
        st.error("OpenAI Key 필요")
        st.stop()

    cases = []
    for line in txt.split("\n"):
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 4:
            cases.append(parts[:4])
    if not cases:
        st.warning("케이스를 입력하세요 (질문|자료|답변|유형)")
        st.stop()

    all_texts = []
    for q, d, a, _ in cases:
        all_texts += [q, d, a]
    with st.spinner("임베딩 계산 중..."):
        try:
            embs = get_embeddings(all_texts, api_key)
        except Exception as e:
            st.error(f"임베딩 실패: {e}")
            st.stop()

    rows = []
    for i, (q, d, a, kind) in enumerate(cases):
        qv, dv, av = embs[i*3], embs[i*3+1], embs[i*3+2]
        rows.append({
            "유형": kind,
            "QD (자료-질문)": round(cos(qv, dv), 3),
            "QA (답-질문)": round(cos(qv, av), 3),
            "DA (답-자료)": round(cos(dv, av), 3),
        })

    st.subheader("측정 결과")
    st.dataframe(rows, use_container_width=True)

    # 뒤틀림 분석 (첫 행=좋은 답변 기준)
    st.subheader("뒤틀림 분석 (기준 대비 하락 축)")
    good = rows[0]
    st.caption(f"기준({good['유형']}): QD={good['QD (자료-질문)']} "
               f"QA={good['QA (답-질문)']} DA={good['DA (답-자료)']}")
    for r in rows[1:]:
        drops = []
        if r["QD (자료-질문)"] < good["QD (자료-질문)"] - 0.08:
            drops.append(f"🟠 QD↓ (자료가 질문과 무관 — 자료에 답 없음)")
        if r["QA (답-질문)"] < good["QA (답-질문)"] - 0.08:
            drops.append(f"🔵 QA↓ (답이 질문과 무관 — 동문서답)")
        if r["DA (답-자료)"] < good["DA (답-자료)"] - 0.08:
            drops.append(f"🔴 DA↓ (답이 자료 근거 없음 — 환각/모순)")
        sig = " · ".join(drops) if drops else "뚜렷한 하락 없음 (좋은 답변에 가까움)"
        st.markdown(f"**{r['유형']}** → {sig}")

    # 3D 산점도 (세 축)
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=[r["QD (자료-질문)"] for r in rows],
            y=[r["QA (답-질문)"] for r in rows],
            z=[r["DA (답-자료)"] for r in rows],
            mode="markers+text",
            text=[r["유형"] for r in rows],
            textposition="top center",
            marker=dict(size=8, color=list(range(len(rows))), colorscale="Viridis"),
        ))
        fig.update_layout(scene=dict(
            xaxis_title="QD 자료-질문", yaxis_title="QA 답-질문", zaxis_title="DA 답-자료"),
            height=500, title="오류 유형이 3D 공간에서 갈리나")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("각 오류가 3D 공간에서 다른 위치에 있으면 → 삼각형으로 유형 구분 가능. "
                   "겹치면 → 삼각형만으론 부족.")
    except Exception as e:
        st.info(f"3D 플롯 건너뜀: {e}")
