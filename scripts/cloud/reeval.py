# Re-evaluate saved checkpoints with 16 samples per problem; per-problem rows saved for bootstrap CIs.
import os, sys, json, time
BASE = '/workspace/pohrl'
os.environ['HF_HOME'] = '/workspace/hf_cache'
sys.path.insert(0, f'{BASE}/code/pohrl'); sys.path.insert(0, BASE)
import job  # applies the answer-checker guard and the config
from pohrl_llm import data as D, run_m3, episode as EP, credit_llm as CL
from pohrl_llm.backends import VLLMBackend
from pohrl_llm import prompts as PR
OUT, G_EVAL = job.OUT, 16
res_path = f'{OUT}/reeval.jsonl'
done = set()
if os.path.exists(res_path):
    done = {json.loads(l)['tag'] for l in open(res_path) if l.strip()}
ev_pool = [r for r in D.load_hf('math500') if int(r.get('level', 0)) >= 3]
backend = VLLMBackend(job.MODEL, dtype='bfloat16', gpu_memory_utilization=0.85, enable_lora=True,
                      max_lora_rank=64, max_model_len=8192)
render = lambda msgs: PR.chat(msgs, backend.tokenizer, enable_thinking=False)
ev_all = run_m3.cache_plans(ev_pool, backend, render, f'{OUT}/plans_eval.jsonl')
eval_planned = run_m3.stratify(ev_all, job.EVAL_PER_BUCKET)
jobs = [('base', 0, None)] + [(m, s, f'{OUT}/{m}_s0/step{s}') for m in job.METHODS for s in (100, 125, 150)]
for k, (m, s, path) in enumerate(jobs):
    tag = f'{m}@{s}'
    if tag in done: continue
    if path and not os.path.isdir(path): job.log('missing', path); continue
    lora = None if path is None else (f're_{m}{s}', 9_000_000 + k, path)
    t0 = time.time()
    rolls = EP.run_episodes(eval_planned, backend, render, G_EVAL, temperature=0.6, max_tokens=1024, lora=lora)
    rows = CL.episode_metrics(eval_planned, rolls)
    rows = [{kk: r[kk] for kk in ('pid', 'width', 'height', 'n', 'acc', 'rounds', 'words_out')} for r in rows]
    open(res_path, 'a').write(json.dumps({'tag': tag, 'method': m, 'step': s, 'rows': rows}) + '\n')
    acc = sum(r['acc'] for r in rows) / len(rows)
    job.log(tag, 'acc', round(acc, 3), f'({(time.time()-t0)/60:.1f} min)')
job.log('REEVAL DONE')
os._exit(0)
