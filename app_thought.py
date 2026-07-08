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

# 정체성 이어가기 (본문 상단 — 모바일에서도 보이게)
with st.expander("💾 정체성 저장 / 불러오기", expanded=False):
    st.caption("대화로 학습한 정체성을 파일로 받아두고, 다음에 올려 이어가세요.")
    uploaded = st.file_uploader(
        "정체성 불러오기 (이전 identity.pkl 업로드)", type=None,
        help="pkl이 아닌 파일은 자동 무시됩니다.")


# ── 사고 구조 초기화 (세션 유지) ──
def build_default_tree():
    ts = ThoughtStructure(learning_rate=lr, continuity=continuity)
    ts.add_node(JudgmentNode(
        'start', '이 질문의 성격을 파악',
        directive="먼저 이 질문이 무엇을 요구하는지 한 문장으로 파악한다."),
        is_root=True)
    ts.add_node(JudgmentNode(
        'fact', '사실 확인이 필요한 질문으로 판단',
        directive="이건 사실을 묻는 질문이다. 확실히 아는 것과 불확실한 것을 구분한다."))
    ts.add_node(JudgmentNode(
        'advice', '조언·판단이 필요한 질문으로 판단',
        directive="이건 조언을 구하는 질문이다. 정답이 아니라 판단이 필요하다."))
    ts.add_node(JudgmentNode(
        'fact_evidence', '근거를 찾아 사실대로 답',
        directive="확실한 근거가 있는 것만 단정하고, 불확실한 것은 '불확실하다'고 "
                  "명시한다. 모르면 모른다고 한다. 추측으로 채우지 않는다.",
        is_terminal=True))
    ts.add_node(JudgmentNode(
        'fact_direct', '아는 대로 바로 답',
        directive="아는 대로 간결하게 바로 답한다.", is_terminal=True))
    ts.add_node(JudgmentNode(
        'advice_careful', '상황을 따져 신중히 조언',
        directive="① 관련된 상황 요소들을 먼저 나열한다. "
                  "② 각 요소를 하나씩 평가한다(좋은 점/걸리는 점). "
                  "③ 그 평가들을 종합해 결론을 낸다. 이 세 단계가 답에 보여야 한다.",
        is_terminal=True))
    ts.add_node(JudgmentNode(
        'advice_quick', '바로 결론부터 조언',
        directive="결론을 한 문장으로 먼저 말하고, 이유를 한 줄 덧붙인다.",
        is_terminal=True))
    ts.add_branch('start', 'fact', 0.5)
    ts.add_branch('start', 'advice', 0.5)
    ts.add_branch('fact', 'fact_evidence', 0.5)
    ts.add_branch('fact', 'fact_direct', 0.5)
    ts.add_branch('advice', 'advice_careful', 0.5)
    ts.add_branch('advice', 'advice_quick', 0.5)
    return ts

def load_identity_from_bytes(raw: bytes):
    """업로드 바이트에서 pkl만 골라 정체성 복원. pkl 아니면 None."""
    import pickle, tempfile
    try:
        blob = pickle.loads(raw)
        # 우리 정체성 pkl인지 최소 확인 (필수 키 존재)
        if not (isinstance(blob, dict) and "transitions" in blob and "nodes" in blob):
            return None
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
            tf.write(raw)
            tmp = tf.name
        return ThoughtStructure.load(tmp)
    except Exception:
        return None   # pkl이 아니거나 형식 안 맞으면 무시


# 업로드된 정체성 처리 (파일명으로 중복 로드 방지)
if uploaded is not None:
    sig = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("loaded_sig") != sig:
        restored = load_identity_from_bytes(uploaded.getvalue())
        if restored is not None:
            st.session_state.ts = restored
            st.session_state.chat = []
            st.session_state.last_record = None
            st.session_state.loaded_sig = sig
            st.success(f"✅ 정체성 복원됨 (학습 {len(restored.history)}회)")
        else:
            st.warning("⚠️ pkl 정체성 파일이 아닙니다 (무시됨)")

if "ts" not in st.session_state:
    st.session_state.ts = build_default_tree()
    st.session_state.chat = []       # (role, text)
    st.session_state.last_record = None

ts = st.session_state.ts

# 정체성 다운로드 (상단 — 모바일에서 바로 보이게)
import pickle as _pkl
ts.save("/tmp/identity.pkl")
with open("/tmp/identity.pkl", "rb") as f:
    _pkl_bytes = f.read()
st.download_button(
    "⬇️ 정체성 다운로드 (identity.pkl)",
    data=_pkl_bytes,
    file_name="identity.pkl",
    mime="application/octet-stream",
    help="지금까지 학습한 정체성을 파일로 받습니다. 다음에 위 '불러오기'로 올리면 이어집니다.",
    use_container_width=True)

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
                prev = st.session_state.last_record
                ts.learn(prev, fb)
                # 교정 사건(직전이 부정 → 이번 긍정)이면 확정 기억
                if fb > 0 and st.session_state.get("prev_feedback", 0) < 0:
                    if hasattr(ts, "remember_from_correction"):
                        ts.remember_from_correction(prev, None)
                    st.session_state.chat.append(
                        ("assistant", "_(교정 확인 — 확정 기억으로 저장했어요)_"))
                else:
                    label = "👍 긍정" if fb > 0 else "👎 부정"
                    st.session_state.chat.append(
                        ("assistant", f"_({label} 반응 감지 — 그 판단 경로를 "
                                      f"{'강화' if fb>0 else '약화'}했어요)_"))
                st.session_state.prev_feedback = fb

        # 2) 반응이 아니면 새 질문으로 사고 진행
        if fb is None:
            choose_fn = make_choose_fn(client, model=model)
            answer_fn = make_answer_fn(client, ts.nodes, model=model)
            # 관련 기억을 불러와 맥락에 주입 (기억흐름)
            mem_ctx = ts.memory_context(user_input) if hasattr(ts, "memory_context") else ""
            full_context = user_input
            if mem_ctx:
                full_context = f"{mem_ctx}\n\n질문: {user_input}"
            with st.spinner("판단 트리 위를 이동 중..."):
                rec = ts.traverse(choose_fn=choose_fn, answer_fn=answer_fn,
                                  context=full_context)
            # 기록엔 원래 질문만 (감사 로그 깔끔하게)
            rec.context = user_input
            st.session_state.last_record = rec
            st.session_state.prev_feedback = 0
            st.session_state.chat.append(("assistant", rec.answer))
            # 매 턴 망각 (안 쓰인 기억 약화)
            if hasattr(ts, "forget_step"):
                ts.forget_step()

        st.rerun()

with col2:
    st.subheader("🔍 사고 궤적 (감사 로그)")
    rec = st.session_state.last_record
    if rec:
        if rec.context:
            st.markdown(f"<small>📥 입력: {rec.context[:40]}</small>",
                        unsafe_allow_html=True)
        if getattr(rec, 'timestamp', ''):
            st.markdown(f"<small>🕐 {rec.timestamp}</small>", unsafe_allow_html=True)
        if getattr(rec, 'sources', None):
            st.markdown(f"<small>📚 근거: {', '.join(rec.sources)[:60]}</small>",
                        unsafe_allow_html=True)
        st.markdown("**이번 답변의 판단 경로:**")
        for i, nid in enumerate(rec.path):
            node = ts.nodes.get(nid)
            arrow = "→ " if i > 0 else "📍 "
            st.markdown(f"{arrow}**{node.prompt if node else nid}**")
            if node and node.directive:
                st.markdown(f"<small style='color:#5f6368;margin-left:14px;'>"
                            f"↳ {node.directive}</small>", unsafe_allow_html=True)
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

    # 기억 (기억흐름 + 망각) — 구버전 엔진 호환
    _mem = getattr(ts, "memory", None)
    if _mem:
        st.markdown("---")
        st.markdown("**🧷 기억 (교정=확정, 나머지는 점차 망각)**")
        for m in sorted(_mem, key=lambda x: -x["trust"]*x["strength"])[:6]:
            tag = "🔒확정" if m["trust"] >= 1.0 else "📎참고"
            bar = "▮" * max(1, int(m["strength"] * 6))
            st.markdown(f"<small>{tag} {m['content'][:32]} "
                        f"<span style='color:#999'>{bar}</span></small>",
                        unsafe_allow_html=True)

st.caption("처음엔 실수해도 됩니다. 답이 맘에 들면 '맞아', 아니면 '아니야'로 반응하세요. "
           "그 반응이 판단 경로를 강화/약화해, 점점 당신에게 맞는 사고로 자랍니다. "
           "이 궤적·전이가 곧 설명이자 정체성입니다.")

