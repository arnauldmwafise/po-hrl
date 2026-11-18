"""Generation backends: vLLM (Colab GPU) and a scripted backend for tests."""
from __future__ import annotations

import random
import re
from typing import Dict, List, Optional


class VLLMBackend:
    """Thin wrapper over vllm.LLM. LoRA adapters produced by the trainer are passed as
    `lora=(name, int_id, path)` and loaded by vLLM per request (no weight copying)."""

    def __init__(self, model: str, max_model_len: int = 8192, gpu_memory_utilization: float = 0.45,
                 enable_lora: bool = False, max_lora_rank: int = 64, dtype: str = "bfloat16", seed: int = 0):
        from vllm import LLM  # imported lazily: not needed for tests
        self.llm = LLM(model=model, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                       enable_lora=enable_lora, max_lora_rank=max_lora_rank, dtype=dtype, seed=seed,
                       enable_prefix_caching=True)
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, prompts: List[str], temperature: float = 1.0, max_tokens: int = 1024,
                 lora=None, top_p: float = 1.0) -> List[str]:
        from vllm import SamplingParams
        sp = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=1)
        kw = {}
        if lora is not None:
            from vllm.lora.request import LoRARequest
            kw["lora_request"] = LoRARequest(*lora)
        outs = self.llm.generate(prompts, sp, use_tqdm=False, **kw)
        return [o.outputs[0].text for o in outs]


class ScriptedBackend:
    """A fake model for end-to-end tests on synthetic arithmetic problems.

    * planner prompts: returns the plan registered for the problem text;
    * executor prompts: evaluates the step expression `EVAL[...]`, substituting the
      parent results it was GIVEN in the prompt (so errors propagate), and answers
      wrongly with probability `err` (per step);
    * chain-of-thought prompts: answers correctly with probability `cot_acc`.
    Records every prompt so tests can check what each executor saw.
    """

    def __init__(self, plans: Dict[str, str], golds: Dict[str, str], err: float = 0.2,
                 cot_acc: float = 0.5, seed: int = 0):
        self.plans, self.golds, self.err, self.cot_acc = plans, golds, err, cot_acc
        self.rng = random.Random(seed)
        self.log: List[str] = []
        self.tokenizer = None

    def generate(self, prompts: List[str], temperature: float = 1.0, max_tokens: int = 1024,
                 lora=None, top_p: float = 1.0) -> List[str]:
        out = []
        for p in prompts:
            self.log.append(p)
            if "Write a plan" in p:
                key = p.rsplit("Problem:", 1)[1].strip().split("\n")[0].strip()
                out.append(self.plans.get(key, "no plan"))
            elif "Your step" in p:
                expr = re.search(r"EVAL\[(.+?)\]", p.split("Your step", 1)[1]).group(1)
                vals = dict(re.findall(r"- \[(\w+)\][^:\n]*: (-?\d+)", p))
                for k, v in vals.items():
                    expr = expr.replace("{" + k + "}", v)
                if "{" in expr:  # a needed result is missing from the context
                    out.append("I lack an input. RESULT: 0"); continue
                val = int(eval(expr, {"__builtins__": {}}))  # synthetic integer arithmetic only
                if self.rng.random() < self.err:
                    val += self.rng.choice([-2, -1, 1, 2])
                out.append(f"Computing. RESULT: {val}")
            else:  # chain of thought
                key = p.split("\n")[1].strip() if "\n" in p else p
                gold = next((g for t, g in self.golds.items() if t in p), "0")
                ans = gold if self.rng.random() < self.cot_acc else str(int(gold) + 1)
                out.append(f"Thinking... \\boxed{{{ans}}}")
        return out
