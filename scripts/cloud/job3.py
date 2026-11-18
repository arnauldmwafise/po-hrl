# M3-wide: executor credit on synthetic multi-part problems whose gold plans are wide partial orders
# (k independent two-step parts, k = 3..6; 1.7-4.6 incomparable pairs per step). Steps verified against gold
# sub-answers. Restarted automatically by supervise3.sh if it hangs; re-evaluates its checkpoints at the end.
import os, sys, random, re, signal, threading, gc, time
BASE = '/workspace/pohrl'
os.environ['HF_HOME'] = '/workspace/hf_cache'
sys.path.insert(0, f'{BASE}/code/pohrl')
MODEL = 'Qwen/Qwen3-1.7B'
METHODS = ['poset', 'chain_rand']
SEEDS = [0]
STEPS, PROBLEMS_PER_STEP, G = 150, 16, 8
EVAL_EVERY, N_TRAIN, EVAL_PER_BUCKET, GPU_UTIL = 25, 1200, 50, 0.35
OUT = f'{BASE}/m3w_{MODEL.split("/")[-1]}'
os.makedirs(OUT, exist_ok=True)
log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)

# answer-checker guard (no unbounded sympy work)
from pohrl_llm import checkers as CK
_orig_sym, _orig_num = CK._sympy_equal, CK._to_number
_BAD = re.compile(r"\^\s*\{?\s*-?\d{3,}|\^\s*\{?[^}]*\^|!|\d{40,}|e\d{3,}|E\d{3,}")
class _TO(Exception): pass
def _alarm(*a): raise _TO()
def _safe_sym(a, b):
    if len(a) > 60 or len(b) > 60 or _BAD.search(a) or _BAD.search(b): return False
    if threading.current_thread() is not threading.main_thread(): return False
    old = signal.signal(signal.SIGALRM, _alarm); signal.setitimer(signal.ITIMER_REAL, 2.0)
    try: return _orig_sym(a, b)
    except _TO: return False
    finally: signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old)
def _safe_num(s):
    t = CK.normalize(s).replace(',', '')
    if len(t) > 40 or re.search(r"[eE][+-]?\d{3,}", t): return None
    return _orig_num(s)
CK._sympy_equal, CK._to_number = _safe_sym, _safe_num

if __name__ == '__main__':
    from pohrl_llm import run_m3, wide
    train = wide.make_set(N_TRAIN, [3, 4, 5, 6], seed=11, prefix='wtr')
    ev_pool = wide.make_set(240, [1, 2, 3, 4, 5, 6], seed=12, prefix='wev')
    for name, probs in [('plans_train.jsonl', train), ('plans_eval.jsonl', ev_pool)]:
        if not os.path.exists(f'{OUT}/{name}'):
            wide.write_plan_cache(probs, f'{OUT}/{name}')
    log(len(train), 'train,', len(ev_pool), 'eval pool (gold plans)')
    from pohrl_llm.backends import VLLMBackend
    from pohrl_llm import prompts as PR
    backend = VLLMBackend(MODEL, dtype='bfloat16', gpu_memory_utilization=GPU_UTIL, enable_lora=True,
                          max_lora_rank=64, max_model_len=8192)
    render = lambda msgs: PR.chat(msgs, backend.tokenizer, enable_thinking=False)
    train_all = run_m3.cache_plans(train, backend, render, f'{OUT}/plans_train.jsonl')
    ev_all = run_m3.cache_plans(ev_pool, backend, render, f'{OUT}/plans_eval.jsonl')
    train_planned = run_m3.planned_only(train_all)
    eval_planned = run_m3.stratify(ev_all, EVAL_PER_BUCKET)
    log('planned:', len(train_planned), 'train,', len(eval_planned), 'eval')

    import torch
    from pohrl_llm.trainer import LoRATrainer
    factory = lambda run_dir, resume: LoRATrainer(MODEL, run_dir, lr=1e-5, rank=64, resume=resume, micro_bs=2)
    for seed in SEEDS:
        for m in METHODS:
            run_m3.train(m, seed, train_planned, eval_planned, backend, render, factory, OUT,
                         steps=STEPS, problems_per_step=PROBLEMS_PER_STEP, G=G,
                         eval_every=EVAL_EVERY, verify_mode='gold', log=log)
            gc.collect(); torch.cuda.empty_cache()
    # re-evaluate checkpoints at steps 100/125/150 with 16 samples (same settings as reeval.py)
    import json as _json
    from pohrl_llm import episode as EP, credit_llm as CL
    res_path = f'{OUT}/reeval.jsonl'
    done = {_json.loads(l)['tag'] for l in open(res_path) if l.strip()} if os.path.exists(res_path) else set()
    for k, (m, s) in enumerate([(m, s) for m in METHODS for s in (100, 125, 150)]):
        tag, path = f'{m}@{s}', f'{OUT}/{m}_s0/step{s}'
        if tag in done or not os.path.isdir(path): continue
        rolls = EP.run_episodes(eval_planned, backend, render, 16, temperature=0.6, max_tokens=1024,
                                lora=(f're_{m}{s}', 9_500_000 + k, path))
        rows = CL.episode_metrics(eval_planned, rolls)
        rows = [{kk: r[kk] for kk in ('pid', 'width', 'height', 'n', 'acc', 'rounds', 'words_out')} for r in rows]
        open(res_path, 'a').write(_json.dumps({'tag': tag, 'method': m, 'step': s, 'rows': rows}) + '\n')
        log('reeval', tag, round(sum(r['acc'] for r in rows) / len(rows), 3))
    open(f'{BASE}/DONE3', 'w').write('done\n')
    log('ALL DONE')
    os._exit(0)
