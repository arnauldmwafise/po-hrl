"""Prompt templates for PO-HRL with a single instruction-tuned model.

The same base model plays the planner and the executor (MLR-style). Templates are
plain strings; `chat()` wraps them with the tokenizer's chat template when one is
available, so the code works for Qwen3, Qwen2.5-Math or any chat model.
"""
from __future__ import annotations

from typing import Dict, List, Optional

PLANNER_SYSTEM = (
    "You are a careful mathematician who plans before solving. "
    "Break the problem into a small set of subgoals and state which earlier subgoals each one uses. "
    "Subgoals that do not depend on each other must NOT be linked. Do not solve anything yet."
)

PLANNER_FEWSHOT = """Problem: Let r1 and r2 be the roots of x^2 - 5x + 6 = 0, and let N be the number of integers n with 1 <= n <= 20 and n = 3 (mod 4). Compute (r1^2 + r2^2) + N.
<plan>
<step id="s1" mode="compute" parents="">Sum of the roots r1 + r2 by Vieta. CLAIM: number</step>
<step id="s2" mode="compute" parents="">Product of the roots r1 * r2 by Vieta. CLAIM: number</step>
<step id="s3" mode="derive" parents="s1,s2">r1^2 + r2^2 = (r1 + r2)^2 - 2 r1 r2. CLAIM: number</step>
<step id="s4" mode="case" parents="">Count integers 1..20 congruent to 3 mod 4. CLAIM: number</step>
<step id="ans" mode="combine" parents="s3,s4">Add the two quantities. CLAIM: number</step>
</plan>"""

PLANNER_USER = """Write a plan for the problem below in exactly this format (2 to {n_max} steps, the last step has id "ans"):
<plan>
<step id="s1" mode="compute|derive|lemma|case|recall|verify|combine" parents="comma-separated ids or empty">what this step establishes. CLAIM: number|expression|statement</step>
...
<step id="ans" mode="combine" parents="...">final combination. CLAIM: number</step>
</plan>

Rules:
1. "parents" lists exactly the earlier steps whose RESULTS this step uses. Steps that do not use each other's results must not be linked.
2. Every step except "ans" must be used by at least one later step (it appears in that step's parents).
3. The "ans" step must list as parents every step whose result the final answer uses.

Example:
{fewshot}

Problem: {problem}"""

AUDIT_SYSTEM = (
    "You check the dependencies of a solution plan. You never change, add or remove steps; "
    "you only correct the parents attribute of each step."
)

AUDIT_USER = """Problem: {problem}

Plan:
{plan}

For each step, decide which EARLIER steps' results it actually needs in order to be carried out.
A step that uses a quantity computed in another step must list that step as a parent.
Steps that do not use each other's results must not be linked. Every step except "ans" must be used by some later step,
and "ans" must list every step its final combination uses.
Return the same steps, in the same order and with the same ids and text, inside <plan> ... </plan>, with corrected parents."""

EXECUTOR_SYSTEM = (
    "You are solving ONE step of a larger solution. Use only the problem and the results you are given. "
    "Reason briefly, then end with a line 'RESULT: <value>' giving exactly what the step asks for."
)

EXECUTOR_USER = """Problem: {problem}

Results already established:
{context}

Your step ({mode}): {desc}
End with 'RESULT: <value>'."""

FINAL_HINT = " This is the final step: the RESULT must be the final answer to the problem, simplified."

COT_SYSTEM = "You are a careful mathematician."
COT_USER = "{problem}\n\nReason step by step and put the final answer in \\boxed{{}}."


def planner_messages(problem: str, n_max: int = 12) -> List[Dict[str, str]]:
    return [{"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": PLANNER_USER.format(problem=problem, n_max=n_max, fewshot=PLANNER_FEWSHOT)}]


def audit_messages(problem: str, plan_text: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": AUDIT_SYSTEM},
            {"role": "user", "content": AUDIT_USER.format(problem=problem, plan=plan_text)}]


def executor_messages(problem: str, desc: str, mode: str, context: Dict[str, str],
                      context_desc: Dict[str, str], is_final: bool) -> List[Dict[str, str]]:
    if context:
        ctx = "\n".join(f"- [{k}] {context_desc.get(k, '')}: {v}" for k, v in context.items())
    else:
        ctx = "(none: this step starts from the problem statement)"
    user = EXECUTOR_USER.format(problem=problem, context=ctx, mode=mode, desc=desc)
    if is_final:
        user += FINAL_HINT
    return [{"role": "system", "content": EXECUTOR_SYSTEM}, {"role": "user", "content": user}]


def cot_messages(problem: str) -> List[Dict[str, str]]:
    return [{"role": "system", "content": COT_SYSTEM}, {"role": "user", "content": COT_USER.format(problem=problem)}]


def chat(messages: List[Dict[str, str]], tokenizer=None, enable_thinking: Optional[bool] = False) -> str:
    """Render messages to a prompt string. Uses the tokenizer's chat template if given."""
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        kw = {}
        if enable_thinking is not None:
            kw["enable_thinking"] = enable_thinking  # Qwen3 switch; ignored by other templates
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out = []
    for m in messages:
        out.append(f"<|{m['role']}|>\n{m['content']}\n")
    out.append("<|assistant|>\n")
    return "".join(out)
