"""
identity_core.py — 정체성 강제 주입 계층.

철학 (GPT 논의 요약):
  LLM = 매번 호출되는 두뇌 (매 호출마다 평균 언어확률로 복귀)
  pkl = 지속되는 사고 구조 (기억)
  NM/SOM = 경계 감시자 (판단 기준)
  → pkl + 가드레일이 호출되는 LLM에 정체성을 계속 부여한다.

이 파일이 하는 일:
  - 정체성 기억(신념·성격·판단기준·핵심사실)을 구조화해 보관.
  - 매 답변 생성 때 그 정체성을 system_prompt에 강제 주입.
  - 답변 후 새로 배운 것을 정체성에 누적(자기진화).
  - '정체성 유지율'을 측정 가능하게 기록.

핵심: LLM이 매번 달라도(gpt-4o-mini → 다른 모델), 같은 정체성 기억으로
  복귀시키면 출력이 같은 방향으로 당겨진다. = 외재적 정체성 유지 시스템.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field, asdict


@dataclass
class IdentityMemory:
    """
    지속되는 정체성. pkl에 담겨 LLM 호출을 넘어 유지된다.
    Parfit식 '심리적 연속성': 기억·성향·가치관·판단방식의 지속.
    """
    # 핵심 정체성 (거의 안 변함 — 정체성의 뼈대)
    persona: str = ""                        # 성격/말투/역할
    values: list = field(default_factory=list)      # 가치관·신념
    judgment_rules: list = field(default_factory=list)  # 판단 기준

    # 누적 기억 (자기진화로 쌓임)
    learned_facts: list = field(default_factory=list)   # 배운 사실
    episodic: list = field(default_factory=list)        # 대화 이력(Q/A 요약)

    # 정체성 유지 추적
    revisions: list = field(default_factory=list)       # 언제 뭐가 바뀌었나
    created_at: float = field(default_factory=time.time)

    # --- 정체성 → 프롬프트 강제 주입 ---
    def to_system_prompt(self, base: str = "") -> str:
        """정체성을 system_prompt로. 매 호출마다 LLM에 강제 주입된다."""
        parts = []
        if self.persona:
            parts.append(f"[너의 정체성]\n{self.persona}")
        if self.values:
            vs = "\n".join(f"- {v}" for v in self.values)
            parts.append(f"[너의 가치관 — 항상 이에 부합하게 답하라]\n{vs}")
        if self.judgment_rules:
            rs = "\n".join(f"- {r}" for r in self.judgment_rules)
            parts.append(f"[너의 판단 기준 — 이 기준으로 판단하라]\n{rs}")
        if self.learned_facts:
            fs = "\n".join(f"- {f}" for f in self.learned_facts[-30:])
            parts.append(f"[네가 축적한 지식]\n{fs}")
        if self.episodic:
            es = "\n".join(f"- {e}" for e in self.episodic[-10:])
            parts.append(f"[최근 대화에서 형성된 맥락]\n{es}")
        identity_block = "\n\n".join(parts)
        if base:
            return f"{base}\n\n=== 아래는 너의 지속적 정체성이다. 매 답변에서 유지하라 ===\n{identity_block}"
        return identity_block

    # --- 자기진화: 답변 후 정체성에 누적 ---
    def absorb(self, question: str, answer: str, learned: str = ""):
        """대화에서 배운 것을 정체성에 흡수(강제 기억 주입)."""
        summary = f"Q: {question[:60]} / A: {answer[:80]}"
        self.episodic.append(summary)
        if learned:
            self.learned_facts.append(learned)
            self.revisions.append({"at": time.time(), "added": learned[:60]})

    def add_value(self, value: str):
        self.values.append(value)
        self.revisions.append({"at": time.time(), "value_added": value[:60]})

    def add_rule(self, rule: str):
        self.judgment_rules.append(rule)
        self.revisions.append({"at": time.time(), "rule_added": rule[:60]})

    # --- 정체성 유지율 (GPT 논의: '있음/없음'이 아니라 '유지율') ---
    def continuity_snapshot(self) -> dict:
        """현재 정체성의 지문. 두 시점을 비교해 유지율을 잴 수 있다."""
        return {
            "persona": self.persona,
            "values": sorted(self.values),
            "rules": sorted(self.judgment_rules),
            "n_facts": len(self.learned_facts),
        }


def continuity_rate(snap_a: dict, snap_b: dict) -> float:
    """
    두 정체성 스냅샷의 유지율(0~1).
    persona 동일 + values/rules 겹침 비율로 계산.
    'PKL 일부 삭제/가치 충돌 주입 후에도 같은 존재인가'를 수치로.
    """
    score, weight = 0.0, 0.0
    # persona (가중치 큼)
    weight += 2
    if snap_a.get("persona") == snap_b.get("persona"):
        score += 2
    # values 겹침
    va, vb = set(snap_a.get("values", [])), set(snap_b.get("values", []))
    if va or vb:
        weight += 2
        score += 2 * (len(va & vb) / max(len(va | vb), 1))
    # rules 겹침
    ra, rb = set(snap_a.get("rules", [])), set(snap_b.get("rules", []))
    if ra or rb:
        weight += 2
        score += 2 * (len(ra & rb) / max(len(ra | rb), 1))
    return round(score / weight, 3) if weight else 1.0
