"""
app_classify.py — 두 축 3분류: 오류 '이유'까지 설명 (Streamlit)

가로축(유사도) + 세로축(근거 실재)로 답변 문장을 3분류:
  MATCH / VALUE_ERROR / HALLUCINATION

실행:
  pip install streamlit openai numpy
  streamlit run app_classify.py
"""
import streamlit as st
import numpy as np
import re

st.set_page_config(page_title="오류 이유 3분류", layout="wide")
st.title("🔎 오류 이유까지 — 두 축 3분류")
st.caption("유사도(자료와 얼마나 가까운가) + 근거 실재(항목이 자료에 있는가)로 "
           "일치 / 값 오류 / 환각을 구분하고 그 '이유'를 설명한다.")

with st.sidebar:
    st.markdown("### 설정")
    api_key = st.text_input("OpenAI Key (선택)", type="password",
                            help="넣으면 유사도 축 사용. 없으면 근거 축만으로 판정.")
    sim_th = st.slider("유사도 임계값 (이상이면 일치)", 0.3, 0.9, 0.6, 0.05)

st.subheader("1. 자료 (판정 기준)")
corpus = st.text_area("자료 원문", height=120,
    value="학생 안전교육은 매 학기 시작 전에 실시하며, 화재 대피와 교통안전을 포함한다.\n"
          "학생 수는 445명이다. 수업일수는 190일이다.")

st.subheader("2. 검증할 답변 (한 줄에 한 문장)")
answer = st.text_area("답변 문장들", height=140,
    value="안전교육은 매 학기 초에 실시한다.\n"
          "안전교육은 격주로 실시한다.\n"
          "학생 수는 500명이다.\n"
          "대통령 승인이 필요하다.\n"
          "헬리콥터로 등교한다.")


def cos_sim(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def get_embeddings(texts, key):
    from openai import OpenAI
    # 키에 섞인 공백/개행/비ascii 문자 제거 (ascii 인코딩 에러 방지)
    key = "".join(ch for ch in key.strip() if ord(ch) < 128)
    client = OpenAI(api_key=key)
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in resp.data])


if st.button("3. 분류 + 이유 설명", type="primary"):
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from two_axis_classify import TwoAxisClassifier

    sentences = [s.strip() for s in answer.split("\n") if s.strip()]
    if not sentences:
        st.warning("답변 문장을 입력하세요.")
        st.stop()

    # 임베딩 (키 있으면)
    embed_fn = None
    ref_vec = None
    vec_cache = {}
    if api_key:
        with st.spinner("임베딩 계산 중..."):
            try:
                import re as _re
                # 자료를 문장 단위로 쪼갬 (classify 내부와 동일 기준)
                corpus_sents = [s.strip() for s in _re.split(r'(?<=[.!?。])\s+|\n+', corpus)
                                if len(s.strip()) >= 5]
                # 답변 + 자료문장 + 자료전체를 모두 임베딩해 캐시
                texts = sentences + corpus_sents + [corpus]
                # 중복 제거(순서 유지)
                uniq = list(dict.fromkeys(texts))
                embs = get_embeddings(uniq, api_key)
                for t, v in zip(uniq, embs):
                    vec_cache[t] = v
                ref_vec = vec_cache.get(corpus)
                embed_fn = lambda t: vec_cache.get(t) if vec_cache.get(t) is not None else ref_vec
            except Exception as e:
                st.error(f"임베딩 실패(근거 축만 사용): {e}")

    clf = TwoAxisClassifier(corpus, embed_fn=embed_fn, sim_threshold=sim_th)

    st.subheader("4. 판정 결과")
    cmap = {"green": ("#e6f4ea", "#137333", "✅"),
            "orange": ("#fef7e0", "#e37400", "△"),
            "red": ("#fce8e6", "#c5221f", "❌"),
            "darkred": ("#fadbd8", "#7b1fa2", "🔴")}
    label_ko = {"MATCH": "일치", "VALUE_ERROR": "값 오류",
                "HALLUCINATION": "환각", "CONTRADICTION": "모순(자료 반대)"}

    counts = {"MATCH": 0, "VALUE_ERROR": 0, "HALLUCINATION": 0, "CONTRADICTION": 0}
    for s in sentences:
        j = clf.classify(s, corpus)
        counts[j.label] = counts.get(j.label, 0) + 1
        bg, fg, icon = cmap[j.color]
        sim_str = f"유사도 {j.similarity:.3f}" if j.similarity >= 0 else "유사도 미측정"
        st.markdown(f"""
        <div style="background:{bg}; border-left:5px solid {fg};
                    padding:10px 14px; margin:6px 0; border-radius:4px;">
          <div style="color:#202124; font-size:15px;">
             {icon} <b>[{label_ko[j.label]}]</b> {s}
          </div>
          <div style="color:{fg}; font-size:13px; margin-top:4px;">{j.reason}</div>
          <div style="color:#5f6368; font-size:11px; margin-top:4px;">
             {sim_str} · 항목 자료 존재: {'예' if j.item_grounded else '아니오'}
          </div>
        </div>""", unsafe_allow_html=True)

    # 요약
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ 일치", counts["MATCH"])
    c2.metric("🔴 모순", counts["CONTRADICTION"])
    c3.metric("△ 값 오류", counts["VALUE_ERROR"])
    c4.metric("❌ 환각", counts["HALLUCINATION"])
    st.caption("모순 = 자료를 정면으로 뒤집음(없다↔있다) → 가장 위험한 환각. "
               "값 오류 = 항목은 맞으나 내용 다름 → 확인 필요. "
               "환각 = 근거 자체가 자료에 없음. 이 구분이 곧 '왜 틀렸는가'의 설명이다.")

    if not api_key:
        st.info("💡 OpenAI Key를 넣으면 유사도 축이 더해져 '일치' 판정이 정밀해집니다. "
                "지금은 근거 축(항목 존재)만으로 판정 중입니다.")
