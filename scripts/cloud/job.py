# M3 full run as a standalone job (restarted automatically by supervise.sh if it hangs)
import os, sys, random, re, signal, threading, gc, time
BASE = '/workspace/pohrl'
os.environ['HF_HOME'] = '/workspace/hf_cache'
sys.path.insert(0, f'{BASE}/code/pohrl')
MODEL = 'Qwen/Qwen3-1.7B'
METHODS = ['poset', 'chain', 'flat']
SEEDS = [0]
STEPS, PROBLEMS_PER_STEP, G = 150, 16, 8
EVAL_EVERY, N_TRAIN, EVAL_PER_BUCKET, GPU_UTIL = 25, 1200, 50, 0.35
OUT = f'{BASE}/m3_{MODEL.split("/")[-1]}'
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
    from pohrl_llm import data as D, run_m3
    SUBJECTS = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra',
                'number_theory', 'prealgebra', 'precalculus']
    keep = lambda r: str(r.get('level', '')) in ('Level 3', 'Level 4', 'Level 5')
    train = []
    try:
        for s in SUBJECTS:
            train += D.load_hf('EleutherAI/hendrycks_math', split='train', qkey='problem', akey='solution', config=s, filter_fn=keep)
    except Exception as e:
        log('hendrycks_math failed, trying MATH-lighteval', e)
        train = D.load_hf('DigitalLearningGmbH/MATH-lighteval', split='train', qkey='problem', akey='solution', filter_fn=keep)
    random.Random(0).shuffle(train); train = train[:N_TRAIN]
    for i, r in enumerate(train): r['id'] = f'tr{i}'
    ev_pool = [r for r in D.load_hf('math500') if int(r.get('level', 0)) >= 3]
    log(len(train), 'train,', len(ev_pool), 'eval pool')

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
                         eval_every=EVAL_EVERY, verify_mode='consensus', log=log)
            gc.collect(); torch.cuda.empty_cache()
    open(f'{BASE}/DONE', 'w').write('done\n')
    log('ALL DONE')
    os._exit(0)
