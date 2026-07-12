"""
app_thought.py — Arcogit (사고 구조 아키텍처)

- 사고 궤적 = 감사 로그 / 학습된 전이 = 정체성
- 피드백은 답 아래 👍/👎 버튼 (수동 텍스트 반응 X)
- 기억 자동 (정보 제공 감지) / 유형 자동 생성 (LLM 설계 + 감사 로그)
- 양심 (정체성 통합) + 자기검증 (답 일관성)
"""
import streamlit as st
import os, sys
sys.path.append(os.path.dirname(__file__))

from thought_structure import ThoughtStructure, JudgmentNode
from tree_registry import TreeRegistry
from llm_bridge import (make_client, make_choose_fn, make_answer_fn,
                        detect_shared_info, make_tree_designer)

st.set_page_config(page_title="Arcogit", layout="wide")
st.title("🧠 Arcogit — 사고 구조 위를 걷는 정체성")
st.caption("판단 트리를 따라 사고하고, 기억하고, 스스로 유형을 만들며 자란다. "
           "궤적=감사, 전이=정체성. 답이 맘에 들면 👍, 아니면 👎.")

with st.sidebar:
    st.markdown("### 설정")
    local = st.checkbox("로컬 LLM (Ollama)", value=False)
    api_key = st.text_input("OpenAI Key", type="password") if not local else ""
    model = st.text_input("모델", value="llama3.2:3b" if local else "gpt-4o-mini")
    lr = st.slider("학습률", 0.05, 0.3, 0.15, 0.05)
    continuity = st.slider("정체성 유지", 0.5, 0.95, 0.7, 0.05)
    auto_type = st.checkbox("유형 자동 생성", value=True,
                            help="새 유형 질문이 오면 LLM이 사고 트리를 설계해 생성")

with st.expander("💾 정체성 저장 / 불러오기", expanded=False):
    st.caption("대화로 학습한 정체성을 파일로 받아두고, 다음에 올려 이어가세요.")
    uploaded = st.file_uploader("정체성 불러오기 (identity.pkl)", type=None,
                                help="pkl이 아닌 파일은 자동 무시됩니다.")


def build_base_tree():
    ts = ThoughtStructure(learning_rate=lr, continuity=continuity)
    ts.add_node(JudgmentNode('start', '이 질문의 성격을 파악',
        directive="이 질문이 무엇을 요구하는지 한 문장으로 파악한다."), is_root=True)
    ts.add_node(JudgmentNode('fact', '사실 확인이 필요한 질문',
        directive="사실을 묻는 질문이다. 아는 것과 불확실한 것을 구분한다."))
    ts.add_node(JudgmentNode('advice', '조언·판단이 필요한 질문',
        directive="조언을 구하는 질문이다. 정답이 아니라 판단이 필요하다."))
    ts.add_node(JudgmentNode('fact_evidence', '근거를 찾아 사실대로 답',
        directive="확실한 근거가 있는 것만 단정하고, 불확실하면 그렇다고 명시한다. "
                  "모르면 모른다고 한다. 기억에 있으면 그것을 근거로 답한다.",
        is_terminal=True))
    ts.add_node(JudgmentNode('advice_careful', '상황을 따져 신중히 조언',
        directive="관련 요소를 따지고, 좋은 점과 걸리는 점을 평가해 결론을 낸다.",
        is_terminal=True))
    ts.add_branch('start', 'fact', 0.5)
    ts.add_branch('start', 'advice', 0.5)
    ts.add_branch('fact', 'fact_evidence', 1.0)
    ts.add_branch('advice', 'advice_careful', 1.0)
    return ts


def load_identity_from_bytes(raw: bytes):
    import pickle, tempfile
    try:
        blob = pickle.loads(raw)
        if not (isinstance(blob, dict) and "transitions" in blob and "nodes" in blob):
            return None
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tf:
            tf.write(raw); tmp = tf.name
        return ThoughtStructure.load(tmp)
    except Exception:
        return None


if uploaded is not None:
    sig = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("loaded_sig") != sig:
        restored = load_identity_from_bytes(uploaded.getvalue())
        if restored is not None:
            st.session_state.ts = restored
            # 통합 pkl이면 registry도 복원
            try:
                import pickle as _p
                _blob = _p.loads(uploaded.getvalue())
                if isinstance(_blob, dict) and "__registry__" in _blob:
                    from tree_registry import TreeRegistry as _TR
                    st.session_state.registry = _TR.from_blob(_blob["__registry__"])
                else:
                    # 옛 pkl (registry 없음) → logic_db로 초기화
                    from tree_registry import TreeRegistry as _TR, load_logic_db_types
                    st.session_state.registry = _TR()
                    load_logic_db_types(st.session_state.registry)
            except Exception:
                pass
            st.session_state.chat = []
            st.session_state.last_record = None
            st.session_state.pending_feedback = False
            st.session_state.loaded_sig = sig
            st.success(f"✅ 정체성 복원됨 (학습 {len(restored.history)}회)")
        else:
            st.warning("⚠️ pkl 정체성 파일이 아닙니다 (무시됨)")

if "ts" not in st.session_state:
    st.session_state.ts = build_base_tree()
    st.session_state.registry = TreeRegistry()
    try:
        from tree_registry import load_logic_db_types
        load_logic_db_types(st.session_state.registry)
    except Exception:
        pass
    st.session_state.chat = []
    st.session_state.last_record = None
    st.session_state.pending_feedback = False

ts = st.session_state.ts
if "registry" not in st.session_state:
    st.session_state.registry = TreeRegistry()
registry = st.session_state.registry

ts.save("/tmp/identity.pkl")
# registry(유형 트리 + 생성 로그)도 같은 pkl에 통합 저장
import pickle as _pkl_mod
with open("/tmp/identity.pkl", "rb") as f:
    _base_blob = _pkl_mod.load(f)
if hasattr(registry, "to_blob"):
    _base_blob["__registry__"] = registry.to_blob()
with open("/tmp/identity.pkl", "wb") as f:
    _pkl_mod.dump(_base_blob, f)
with open("/tmp/identity.pkl", "rb") as f:
    _pkl_bytes = f.read()
st.download_button("⬇️ 정체성 다운로드 (identity.pkl)", data=_pkl_bytes,
    file_name="identity.pkl", mime="application/octet-stream",
    use_container_width=True)

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("대화")
    for role, text in st.session_state.chat:
        with st.chat_message(role):
            st.write(text)

    # ── 피드백 버튼 ──
    if st.session_state.last_record is not None and st.session_state.pending_feedback:
        st.markdown("<small style='color:#5f6368'>이 답변이 맘에 드나요? "
                    "학습에 반영됩니다.</small>", unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns([1, 1, 4])
        with fc1:
            if st.button("👍 추천", use_container_width=True):
                ts.learn(st.session_state.last_record, 1.0)
                st.session_state.pending_feedback = False
                st.rerun()
        with fc2:
            if st.button("👎 비추천", use_container_width=True):
                ts.learn(st.session_state.last_record, -1.0)
                st.session_state.pending_feedback = False
                st.rerun()
        with fc3:
            if st.button("건너뛰기", use_container_width=True):
                st.session_state.pending_feedback = False
                st.rerun()

    user_input = st.chat_input("무엇이든 물어보세요")

    if user_input:
        if not local and not api_key:
            st.error("OpenAI Key를 넣거나 로컬 LLM을 켜세요"); st.stop()

        client = make_client(api_key, local=local)
        st.session_state.chat.append(("user", user_input))

        # 1) 정보 제공이면 자동 기억
        shared = detect_shared_info(user_input, client=client, model=model)
        if shared and hasattr(ts, "remember"):
            stored = ts.remember(shared, trust=1.0, context=user_input)
            if stored:
                st.session_state.chat.append(("assistant", f"기억했어요: {shared} 🔒"))
            else:
                st.session_state.chat.append(
                    ("assistant", f"_(기존 정체성과 충돌해 보류: {shared})_"))
            st.session_state.last_record = None
            st.session_state.pending_feedback = False
            st.rerun()

        # 2) 새 질문으로 사고
        choose_fn = make_choose_fn(client, model=model)
        answer_fn = make_answer_fn(client, ts.nodes, model=model)

        # 유형 자동 생성 (감사 로그)
        if auto_type:
            try:
                designer = make_tree_designer(client, model=model)
                existing_types = list(registry.trees.keys())
                design = designer(user_input)
                if design and design.get("type_id") not in existing_types:
                    registry.create_from_design(
                        design, user_input, reason="새 유형 감지 (LLM 설계)")
            except Exception:
                pass

        mem_ctx = ts.memory_context(user_input) if hasattr(ts, "memory_context") else ""
        flow_ctx = ts.recent_context(6) if hasattr(ts, "recent_context") else ""
        parts = []
        if flow_ctx: parts.append(flow_ctx)
        if mem_ctx: parts.append(mem_ctx)
        parts.append(f"질문: {user_input}")
        full_context = "\n\n".join(parts)

        with st.spinner("판단 트리 위를 이동 중..."):
            if hasattr(ts, "traverse_verified"):
                rec = ts.traverse_verified(choose_fn=choose_fn, answer_fn=answer_fn,
                                           context=full_context, max_retry=1)
            else:
                rec = ts.traverse(choose_fn=choose_fn, answer_fn=answer_fn,
                                  context=full_context)
        rec.context = user_input
        if hasattr(ts, "add_episode"):
            ts.add_episode(user_input, rec.answer)
        st.session_state.last_record = rec
        st.session_state.pending_feedback = True
        st.session_state.chat.append(("assistant", rec.answer))
        if hasattr(ts, "forget_step"):
            ts.forget_step()
        st.rerun()

with col2:
    st.subheader("🔍 사고 궤적 (감사 로그)")
    rec = st.session_state.last_record
    if rec:
        if rec.context:
            st.markdown(f"<small>📥 입력: {rec.context[:40]}</small>", unsafe_allow_html=True)
        if getattr(rec, 'timestamp', ''):
            st.markdown(f"<small>🕐 {rec.timestamp}</small>", unsafe_allow_html=True)
        if hasattr(rec, 'verified'):
            vtag = "✅ 자기검증 통과" if rec.verified else "⚠️ 검증 미통과(재계산)"
            st.markdown(f"<small>{vtag}</small>", unsafe_allow_html=True)
        st.markdown("**판단 경로:**")
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
    st.subheader("🎭 정체성")
    dom = ts.dominant_path()
    st.markdown("**기본 사고 경로:** " + " → ".join(ts.nodes[n].prompt[:12] for n in dom))
    st.markdown(f"**정체성 안정도:** {ts.continuity_rate():.0%}  ·  "
                f"**학습:** {len(ts.history)}회")

    _mem = getattr(ts, "memory", None)
    if _mem:
        st.markdown("---")
        st.markdown("**🧷 기억**")
        for i, m in enumerate(sorted(_mem, key=lambda x: -x["trust"]*x["strength"])):
            tag = "🔒" if m["trust"] >= 1.0 else "📎"
            mc1, mc2 = st.columns([5, 1])
            with mc1:
                st.markdown(f"<small>{tag} {m['content'][:38]}</small>",
                            unsafe_allow_html=True)
            with mc2:
                if st.button("🗑", key=f"del_{i}_{m['content'][:6]}"):
                    ts.memory.remove(m); st.rerun()

    clog = registry.creation_history() if hasattr(registry, "creation_history") else []
    if clog:
        st.markdown("---")
        st.markdown("**🌱 자동 생성된 유형 (감사)**")
        for log in clog[-5:]:
            st.markdown(f"<small>• **{log['type_id']}** ← {log['trigger_question'][:24]}<br>"
                        f"<span style='color:#888'>{' → '.join(log['steps'])}</span></small>",
                        unsafe_allow_html=True)

st.caption("Arcogit — 사고가 곧 존재, 궤적이 곧 정체성. 답에 👍/👎로 반응하면 학습됩니다.")
