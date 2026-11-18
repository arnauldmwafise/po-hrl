"""M3 LoRA policy-gradient trainer (PyTorch + PEFT). Runs on the Colab GPU.

After each update the adapter is saved and handed to vLLM as a new LoRA request,
so generation always uses the current policy without copying weights.
All batching / masking / normalization logic lives in trainer_core (tested on CPU).
"""
from __future__ import annotations

import os
from typing import List, Optional

from . import trainer_core as TC


class LoRATrainer:
    def __init__(self, model_name: str, out_dir: str, lr: float = 1e-5, rank: int = 64, alpha: int = 128,
                 max_len: int = 3072, micro_bs: int = 2, grad_clip: float = 1.0, resume: Optional[str] = None):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                    attn_implementation="sdpa").cuda()
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
        if resume:
            self.model = PeftModel.from_pretrained(base, resume, is_trainable=True)
        else:
            cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            self.model = get_peft_model(base, cfg)
        self.opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
        self.out_dir, self.max_len, self.micro_bs, self.grad_clip = out_dir, max_len, micro_bs, grad_clip
        os.makedirs(out_dir, exist_ok=True)

    def step(self, examples: List[dict]) -> dict:
        torch = self.torch
        if not examples:
            return {"loss": 0.0, "examples": 0}
        enc = [TC.build_ids(self.tok(e["prompt"], add_special_tokens=False)["input_ids"],
                            self.tok(e["completion"], add_special_tokens=False)["input_ids"] + [self.tok.eos_token_id],
                            self.max_len) for e in examples]
        total = sum(sum(m) for _, m in enc)
        self.model.train(); self.opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for mb in TC.micro_batches(len(enc), self.micro_bs):
            ids, attn, lm = TC.pad_batch([enc[i][0] for i in mb], [enc[i][1] for i in mb], self.tok.pad_token_id)
            ids = torch.tensor(ids, device="cuda"); attn = torch.tensor(attn, device="cuda")
            lm = torch.tensor(lm, device="cuda", dtype=torch.float32)
            adv = torch.tensor([examples[i]["advantage"] for i in mb], device="cuda", dtype=torch.float32)
            logits = self.model(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
            logp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None])[..., 0]
            w = lm[:, 1:] * adv[:, None] / total          # token y_t predicted at position t-1
            loss = -(w * logp).sum()
            loss.backward(); loss_sum += float(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.opt.step()
        return {"loss": loss_sum, "examples": len(examples), "tokens": total}

    def save(self, tag: str) -> str:
        path = os.path.join(self.out_dir, tag)
        self.model.save_pretrained(path)
        return path
