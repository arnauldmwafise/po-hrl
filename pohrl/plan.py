"""Planner output parsing and validation (spec Section 2, rules V1-V6)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .poset import CycleError, Poset

TOP = "ans"
MODES = {"compute", "derive", "lemma", "case", "recall", "verify", "combine"}
STEP_RE = re.compile(r"<step\b(?P<attrs>[^>]*)>(?P<body>.*?)</step>", re.S)
ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')


@dataclass
class Step:
    id: str
    mode: str
    parents: List[str]
    text: str
    claim: Optional[str] = None


@dataclass
class ValidationResult:
    poset: Poset
    steps: Dict[str, Step]
    ok: bool
    fallback: bool
    penalty: float
    issues: List[str] = field(default_factory=list)


def fallback_plan(problem_text: str = "") -> ValidationResult:
    st = Step(TOP, "combine", [], "Solve the whole problem.")
    return ValidationResult(Poset([TOP], {}), {TOP: st}, ok=False, fallback=True, penalty=-1.0)


def parse(text: str) -> List[Step]:
    blocks = re.findall(r"<plan>(.*?)</plan>", text, re.S)
    if not blocks:
        raise ValueError("no <plan> block")
    steps = []
    for sm in STEP_RE.finditer(blocks[-1]):
        attrs = dict(ATTR_RE.findall(sm.group("attrs")))   # any order; stray quotes ignored
        if "id" not in attrs:
            continue
        body = sm.group("body").strip()
        claim = None
        if "CLAIM:" in body:
            body, claim = [s.strip() for s in body.split("CLAIM:", 1)]
        ps = [p.strip() for p in attrs.get("parents", "").split(",") if p.strip()]
        steps.append(Step(attrs["id"].strip(), attrs.get("mode", "compute").strip(), ps, body, claim))
    if not steps:
        raise ValueError("no steps")
    return steps


def render_plan(steps: List[Step]) -> str:
    rows = [f'<step id="{s.id}" mode="{s.mode}" parents="{",".join(s.parents)}">{s.text}'
            + (f" CLAIM: {s.claim}" if s.claim else "") + "</step>" for s in steps]
    return "<plan>\n" + "\n".join(rows) + "\n</plan>"


def validate(text: str, n_max: int = 16, max_desc_words: int = 60,
             orphan_penalty: float = 0.1, orphan_policy: str = "next") -> ValidationResult:
    """Steps not below `ans` ("orphans") are repaired according to orphan_policy:
      'next' (default, conservative): each orphan without a child is made a parent of the
             next step the planner WROTE after it, i.e. an unstated link is assumed to be the
             sequential one, so width only comes from independence the planner declared;
      'link': orphans' maximal elements become parents of `ans` (treats them as independent);
      'drop': orphans are removed.
    The number of orphans is recorded in issues and penalized in every case."""
    issues: List[str] = []
    try:
        steps = parse(text)
    except ValueError as e:
        r = fallback_plan()
        r.issues.append(f"V1 parse: {e}")
        return r
    ids = [s.id for s in steps]
    if len(set(ids)) != len(ids):
        r = fallback_plan(); r.issues.append("V1 duplicate ids"); return r
    idset = set(ids)
    for s in steps:
        unknown = [p for p in s.parents if p not in idset or p == s.id]
        if unknown:
            issues.append(f"V1 unknown parents dropped at {s.id}: {unknown}")
            s.parents = [p for p in s.parents if p not in unknown]
        if s.mode not in MODES:
            issues.append(f"unknown mode {s.mode} at {s.id}")
        if len(s.text.split()) > max_desc_words:
            r = fallback_plan(); r.issues.append(f"V5 descriptor too long at {s.id}"); return r
    if TOP not in idset:
        # the planner named its final step differently: treat the LAST written step as `ans`
        last = steps[-1].id
        issues.append(f"V3 last step {last} renamed to ans")
        for st in steps:
            st.parents = [TOP if q == last else q for q in st.parents]
        steps[-1].id = TOP
        ids = [st.id for st in steps]
        idset = set(ids)
    try:
        P = Poset(ids, {s.id: s.parents for s in steps})
    except CycleError as e:
        r = fallback_plan(); r.issues.append(f"V2 {e}"); return r
    penalty = 0.0
    # V3: everything must be below TOP; drop orphans
    keep = P.down(TOP)
    orphans = [g for g in ids if g not in keep]
    if orphans:
        penalty -= orphan_penalty * len(orphans)
        if orphan_policy == "next":
            new_parents = {s.id: list(s.parents) for s in steps}
            pos = {g: i for i, g in enumerate(ids)}
            linked = []
            for g in P.maximal(orphans):
                for h in ids[pos[g] + 1:]:
                    if not P.leq(h, g):          # never create a cycle
                        new_parents[h] = sorted(set(new_parents[h]) | {g}); linked.append(f"{g}->{h}")
                        break
                else:
                    new_parents[TOP] = sorted(set(new_parents[TOP]) | {g}); linked.append(f"{g}->{TOP}")
            issues.append(f"V3 orphans repaired by next-step links: {orphans} ({', '.join(linked)})")
            P = Poset(ids, new_parents)
            steps = [Step(s.id, s.mode, new_parents[s.id], s.text, s.claim) for s in steps]
            # a repair can itself leave steps below a new parent that is still an orphan: repeat
            keep = P.down(TOP)
            if len(keep) < len(ids):
                return validate(render_plan(steps), n_max, max_desc_words, orphan_penalty, orphan_policy)
        elif orphan_policy == "link":
            sinks = P.maximal(orphans)
            issues.append(f"V3 orphans linked to ans: {orphans} (via {sinks})")
            new_parents = {s.id: list(s.parents) for s in steps}
            new_parents[TOP] = sorted(set(new_parents[TOP]) | set(sinks))
            P = Poset(ids, new_parents)
            steps = [Step(s.id, s.mode, new_parents[s.id], s.text, s.claim) for s in steps]
        else:
            issues.append(f"V3 orphans dropped: {orphans}")
            P = P.subposet(keep)
            steps = [s for s in steps if s.id in keep]
    if not (2 <= len(P) <= n_max):
        r = fallback_plan(); r.issues.append(f"V5 size {len(P)} outside [2,{n_max}]"); return r
    # V4: reduce to Hasse diagram
    red = P.redundant_edges()
    if red:
        issues.append(f"V4 removed {len(red)} transitive edges")
    cov = P.covers()
    P = Poset(P.elements, cov)
    smap = {s.id: Step(s.id, s.mode, sorted(cov[s.id]), s.text, s.claim) for s in steps}
    return ValidationResult(P, smap, ok=True, fallback=False, penalty=penalty, issues=issues)


def revision_allowed(old: Poset, new: Poset, accepted: Set[str]) -> bool:
    """V6: the accepted ideal must remain an ideal of the revised plan, with the
    same internal order (finished work is never invalidated)."""
    if not accepted <= set(new.elements):
        return False
    if not new.is_ideal(accepted):
        return False
    return all(new.strict_down(g) == old.strict_down(g) for g in accepted)
