"""
app_thought.py — 사고 구조 아키텍처 (Streamlit)

LLM이 판단 분기 트리 위를 이동하며 답한다.
사고 궤적(XAI)이 보이고, 사용자 반응으로 전이가 학습된다(정체성 성장).

실행:
  pip install streamlit openai
  streamlit run app_thought.py

  - GPT: 사이드바에 OpenAI Key
  - Ollama(로컬): "로컬 LLM" 체크 (http://localhost:11434)
"""
import streamlit as st
import os, sys
sys.path.append(os.path.dirname(__file__))

from thought_structure import ThoughtStructure, JudgmentNode
from llm_bridge import make_client, make_choose_fn, make_answer_fn, detect_feedback

st.set_page_config(page_title="사고 구조", layout="wide")
st.title("🧠 사고 구조 — 판단 트리 위를 걷는 LLM")
st.caption("LLM이 판단 분기 트리를 따라 이동하며 답한다. 궤적이 곧 설명(XAI), "
           "학습된 전이가 곧 정체성. 반응으로 실수에서 벗어난다.")

with st.sidebar:
    st.markdown("### 설정")
    local = st.checkbox("로컬 LLM (Ollama)", value=False)
    api_key = st.text_input("OpenAI Key", type="password") if not local else ""
    model = st.text_input("모델", value="llama3.2:3b" if local else "gpt-4o-mini")
    lr = st.slider("학습률", 0.05, 0.3, 0.15, 0.05)
    continuity = st.slider("정체성 유지", 0.5, 0.95, 0.7, 0.05)


# ── 사고 구조 초기화 (세션 유지) ──
def build_default_tree():
    ts = ThoughtStructure(learning_rate=lr, continuity=continuity)
    ts.add_node(JudgmentNode('start', '이 질문의 성격을 파악'), is_root=True)
    ts.add_node(JudgmentNode('fact', '사실 확인이 필요한 질문으로 판단'))
    ts.add_node(JudgmentNode('advice', '조언·판단이 필요한 질문으로 판단'))
    ts.add_node(JudgmentNode('fact_evidence', '근거를 찾아 사실대로 답', is_terminal=True))
    ts.add_node(JudgmentNode('fact_direct', '아는 대로 바로 답', is_terminal=True))
    ts.add_node(JudgmentNode('advice_careful', '상황을 따져 신중히 조언', is_terminal=True))
    ts.add_node(JudgmentNode('advice_quick', '바로 결론부터 조언', is_terminal=True))
    ts.add_branch('start', 'fact', 0.5)
    ts.add_branch('start', 'advice', 0.5)
    ts.add_branch('fact', 'fact_evidence', 0.5)
    ts.add_branch('fact', 'fact_direct', 0.5)
    ts.add_branch('advice', 'advice_careful', 0.5)
    ts.add_branch('advice', 'advice_quick', 0.5)
    return ts

if "ts" not in st.session_state:
    st.session_state.ts = build_default_tree()
    st.session_state.chat = []       # (role, text)
    st.session_state.last_record = None

ts = st.session_state.ts

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("대화")
    # 이전 대화 표시
    for role, text in st.session_state.chat:
        with st.chat_message(role):
            st.write(text)

    user_input = st.chat_input("질문하거나, 직전 답에 반응하세요 (맞아/아니야 등)")

    if user_input:
        if not local and not api_key:
            st.error("OpenAI Key를 넣거나 로컬 LLM을 켜세요")
            st.stop()

        client = make_client(api_key, local=local)
        st.session_state.chat.append(("user", user_input))

        # 1) 직전 답변이 있으면, 이 입력이 '반응'인지 먼저 감지
        fb = None
        if st.session_state.last_record is not None:
            fb = detect_feedback(user_input, client=client, model=model)
            if fb is not None:
                ts.learn(st.session_state.last_record, fb)
                label = "👍 긍정" if fb > 0 else "👎 부정"
                st.session_state.chat.append(
                    ("assistant", f"_({label} 반응 감지 — 그 판단 경로를 "
                                  f"{'강화' if fb>0 else '약화'}했어요)_"))

        # 2) 반응이 아니면(또는 반응이어도) 새 질문으로 사고 진행
        if fb is None:
            choose_fn = make_choose_fn(client, model=model)
            answer_fn = make_answer_fn(client, ts.nodes, model=model)
            with st.spinner("판단 트리 위를 이동 중..."):
                rec = ts.traverse(choose_fn=choose_fn, answer_fn=answer_fn,
                                  context=user_input)
            st.session_state.last_record = rec
            st.session_state.chat.append(("assistant", rec.answer))

        st.rerun()

with col2:
    st.subheader("🔍 사고 궤적 (XAI)")
    rec = st.session_state.last_record
    if rec:
        st.markdown("**이번 답변의 판단 경로:**")
        for i, nid in enumerate(rec.path):
            node = ts.nodes.get(nid)
            arrow = "→ " if i > 0 else "📍 "
            st.markdown(f"{arrow}{node.prompt if node else nid}")
    else:
        st.caption("질문하면 여기에 판단 경로가 보여요")

    st.markdown("---")
    st.subheader("🎭 정체성 (학습된 전이)")
    st.markdown("**현재 지배 경로 (기본 사고):**")
    dom = ts.dominant_path()
    st.markdown(" → ".join(ts.nodes[n].prompt[:12] for n in dom))

    st.markdown("**전이 확률:**")
    for frm in ts.transitions:
        tr = ts.transitions[frm]
        if tr:
            frm_name = ts.nodes[frm].prompt[:14]
            for to, p in tr.items():
                to_name = ts.nodes[to].prompt[:14]
                bar = "█" * int(p * 12)
                st.markdown(f"<small>{frm_name} → {to_name}: `{bar}` {p:.0%}</small>",
                            unsafe_allow_html=True)

    st.markdown(f"**정체성 안정도:** {ts.continuity_rate():.0%}")
    st.markdown(f"**총 학습 횟수:** {len(ts.history)}")

    if st.button("정체성 저장 (pkl)"):
        path = ts.save("/tmp/identity.pkl")
        st.success(f"저장됨: {path}")

st.caption("처음엔 실수해도 됩니다. 답이 맘에 들면 '맞아', 아니면 '아니야'로 반응하세요. "
           "그 반응이 판단 경로를 강화/약화해, 점점 당신에게 맞는 사고로 자랍니다. "
           "이 궤적·전이가 곧 설명이자 정체성입니다.")
