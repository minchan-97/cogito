"""
app_chain.py — 근거 소급 사슬 자동 추출 (Streamlit)

답변을 넣으면 근거를 자료에서 찾아 앵커(인간보증/법령)까지 소급하고,
사슬 전체를 보여준다. "왜 이 답을 믿는가"가 사슬로 설명된다.

실행:
  pip install streamlit openai
  streamlit run app_chain.py
"""
import streamlit as st

st.set_page_config(page_title="근거 소급 사슬", layout="wide")
st.title("⛓️ 근거 소급 사슬 — 왜 이 답을 믿는가")
st.caption("답변의 근거를 자료에서 찾아 앵커(인간 결재 / 법령)까지 소급한다. "
           "앵커에 닿으면 믿을 근거가 있는 것, 끊기면 근거 없는 주장.")

with st.sidebar:
    st.markdown("### 설정")
    api_key = st.text_input("OpenAI Key (선택)", type="password",
                            help="넣으면 LLM이 근거를 인용. 없으면 규칙(문장 겹침)으로 소급.")
    max_depth = st.slider("최대 소급 깊이", 2, 10, 6)

st.subheader("1. 자료 (근거의 출처)")
corpus = st.text_area("자료 원문", height=140,
    value="안전교육은 매 학기 실시한다. 이는 학교 안전규정 제3조에 따라 시행된다.\n"
          "학교 안전규정 제3조는 초·중등교육법 시행령에 근거한다.\n"
          "이 계획은 2026년 3월 교장 결재로 승인되었다.")

st.subheader("2. 검증할 답변")
answer = st.text_input("답변 문장", value="안전교육은 매 학기 실시한다")


def make_llm_fn(key):
    """LLM이 주장의 근거를 자료에서 인용(원문 그대로)."""
    def fn(statement, corpus):
        try:
            from openai import OpenAI
            k = "".join(ch for ch in key.strip() if ord(ch) < 128)
            client = OpenAI(api_key=k)
            prompt = (f"자료:\n{corpus[:3000]}\n\n"
                      f"주장: \"{statement}\"\n\n"
                      f"이 주장의 근거가 되는 문장을 자료에서 찾아 '원문 그대로' 한 문장만 인용하세요. "
                      f"자료에 없으면 빈 문자열로 답하세요. 다른 말 없이 인용문만.")
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0)
            return resp.choices[0].message.content.strip().strip('"')
        except Exception as e:
            st.warning(f"LLM 호출 실패(규칙으로 폴백): {e}")
            return ""
    return fn


if st.button("3. 근거 소급", type="primary"):
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from chain_builder import ChainBuilder

    llm_fn = make_llm_fn(api_key) if api_key else None
    cb = ChainBuilder(corpus, llm_fn=llm_fn, max_depth=max_depth)
    chain = cb.build(answer)
    summary = chain.summary()

    st.subheader("3. 소급 사슬")
    if not chain.links:
        st.error("❌ 근거를 자료에서 찾지 못했습니다 — 근거 없는 주장(ungrounded)")
    else:
        # 사슬을 위→아래로 시각화
        for i, l in enumerate(chain.links):
            arrow = "⬇️" if i > 0 else "📌"
            anchor_badge = ""
            if l.is_anchor:
                atype_ko = {"human": "🔒 인간 보증(결재/서명)", "axiom": "📜 법령/공리"}
                anchor_badge = f'<span style="background:#137333;color:white;padding:2px 8px;border-radius:10px;font-size:12px;">{atype_ko.get(l.anchor_type, l.anchor_type)}</span>'
            verified = "✓ 자료 실재" if l.quote_verified else "✗ 미검증"
            vcolor = "#137333" if l.quote_verified else "#c5221f"
            ref = f'<span style="color:#5f6368;">→ 상위 근거: {l.refers_to}</span>' if l.refers_to else ""
            st.markdown(f"""
            <div style="border-left:3px solid #1a73e8; padding:8px 14px; margin:4px 0 4px 20px;">
              <div style="font-size:13px;color:#5f6368;">{arrow} 단계 {l.depth}</div>
              <div style="font-size:15px;color:#202124;margin:3px 0;"><b>{l.statement[:60]}</b></div>
              <div style="font-size:13px;color:#3c4043;">근거: {l.quote[:80]}</div>
              <div style="font-size:12px;margin-top:3px;">
                <span style="color:{vcolor};">{verified}</span> · {ref} {anchor_badge}
              </div>
            </div>""", unsafe_allow_html=True)

    # 최종 판정
    st.markdown("---")
    status = summary["status"]
    if status == "grounded":
        st.success(f"✅ **GROUNDED** — {summary['anchor_type']} 앵커까지 "
                   f"{summary['depth']}단계 소급, 전 연결 자료 실재 확인. 믿을 근거가 있음.")
    elif status == "weak":
        st.warning(f"△ **WEAK** — 앵커({summary['anchor_type']})에 닿았으나 일부 연결 미검증.")
    else:
        st.error(f"❌ **UNGROUNDED** — {summary['depth']}단계 후 앵커 없이 끊김 "
                 f"(종결: {summary['terminated']}). 근거 없는 주장 가능.")

    st.caption("사슬이 인간 결재나 법령 같은 앵커에서 끝나면, 그 앵커가 신뢰의 근거다. "
               "cogito: 무한 소급을 외부 앵커(인간 보증)로 끊는다. "
               "이 사슬 전체가 곧 '왜 믿는가'의 설명이다.")
