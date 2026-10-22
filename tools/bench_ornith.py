#!/usr/bin/env python3
"""Repeatable, uncached engine-protocol benchmark; runs one model at a time.

Command JSON contains the exact command array (including --serve) and optional env.
Model is used only for its tokenizer. Artifacts include all inputs, outputs, timings,
resource samples and binary hash. Engine errors are retained and return failure.
"""
import argparse, hashlib, json, os, pathlib, queue, subprocess, threading, time, urllib.request
from jinja2 import Environment
from strata_tokenizer import Tokenizer


def gpu():
    return [int(v.strip()) for v in subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(',')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--command', type=pathlib.Path, required=True)
    ap.add_argument('--model', type=pathlib.Path, required=True)
    ap.add_argument('--template', type=pathlib.Path, required=True)
    ap.add_argument('--output', type=pathlib.Path, required=True)
    ap.add_argument('--lengths', default='512,1024,4096,7168')
    ap.add_argument('--workloads', default='prose,code,retrieval')
    ap.add_argument('--corpus', type=pathlib.Path, help='Optional non-repeating prompt material; otherwise synthetic repeated notes')
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--new-tokens', type=int, default=256)
    ap.add_argument('--prose-words', type=int, default=600, help='Requested essay length; raise for long generation tests')
    ap.add_argument('--llamaswap', default='http://127.0.0.1:9292')
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.command.read_text())
    urllib.request.urlopen(a.llamaswap + '/unload', timeout=15).read()
    assert not json.load(urllib.request.urlopen(a.llamaswap + '/running', timeout=15))['running']
    idle = []
    for _ in range(60):
        sample = gpu()
        idle = idle + [sample] if sample[0] < 100 and sample[1] == 0 else []
        if len(idle) == 3: break
        time.sleep(1)
    assert len(idle) == 3, 'GPU did not become idle within 60 seconds'
    tk = Tokenizer.from_gguf(a.model)
    template = Environment().from_string(a.template.read_text())
    tasks = {
        'prose': f'Write a detailed {a.prose_words}-word explanation of how a forest ecosystem recovers after a wildfire. Discuss soil, plants, animals, and water.',
        'code': 'Write Python code implementing an LRU cache with get and put methods, type annotations, and unit tests for eviction and updating existing keys. Explain its complexity.',
        'retrieval': 'Find the key associated with Cedar Observatory in the reference notes. State the key, then explain how an index could make this lookup efficient.'}
    notes = ('Reference notes: Cedar Observatory has key ORNITH-724. The river station measures rainfall. '
             'Birch Library opens at nine. Field researchers catalog plants, inspect sensors, and record daily observations.\n')
    filler = tk.encode(notes + a.corpus.read_text()) if a.corpus else tk.encode(notes * 1000)
    cases = []
    for length in map(int, a.lengths.split(',')):
        for name in a.workloads.split(','):
            # Token-level insertion preserves the complete chat wrapper and question at exact lengths.
            before, after = template.render(messages=[{'role':'user', 'content':'BENCHMARK_INSERTION\n\n' + tasks[name]}], add_generation_prompt=True, enable_thinking=False).split('BENCHMARK_INSERTION')
            first, last = tk.encode(before, parse_special=True), tk.encode(after, parse_special=True)
            n = length - len(first) - len(last)
            assert 0 < n <= len(filler), 'Corpus is too short for the requested prompt length'
            ids = first + filler[:n] + last
            assert len(ids) == length
            cases.append({'name': name, 'prompt_tokens': length, 'ids': ids})
    (a.output/'inputs.json').write_text(json.dumps(cases))
    unit = 'ornith-perf-' + str(os.getpid())
    cmd = ['systemd-run', '--user', '--scope', '--quiet', '--unit='+unit, '-p', 'MemoryMax=56G', '-p', 'MemorySwapMax=0'] + cfg['command']
    env = os.environ.copy(); env.update(cfg.get('env', {}))
    result = {'output_limit': a.new_tokens, 'prose_words': a.prose_words, 'corpus': str(a.corpus) if a.corpus else 'synthetic repeated notes', 'config': cfg, 'command': cmd, 'idle': idle, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'tests': []}
    exe = pathlib.Path(cfg.get('engine_binary', cfg['command'][0]))
    if exe.is_file(): result['binary_sha256'] = hashlib.sha256(exe.read_bytes()).hexdigest()
    library = exe.parent / 'libstrata_kq.so'
    if library.is_file(): result['library_sha256'] = hashlib.sha256(library.read_bytes()).hexdigest()
    result['template_sha256'] = hashlib.sha256(a.template.read_bytes()).hexdigest()
    stop = threading.Event(); lines = queue.Queue()
    log = open(a.output/'engine.log', 'w')
    started = time.monotonic()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1, env=env)
    def read():
        for line in proc.stdout: lines.put(line.strip())
        lines.put(None)
    def monitor():
        with open(a.output/'resources.jsonl', 'w') as f:
            while not stop.is_set():
                try:
                    fields = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(',')
                    mem, util = map(int, fields[:2])
                    cg = subprocess.check_output(['systemctl','--user','show',unit+'.scope','-p','ControlGroup','--value'], text=True).strip()
                    base = pathlib.Path('/sys/fs/cgroup' + cg)
                    row = {'time': time.time(), 'gpu_mib': mem, 'gpu_util': util, 'gpu_sm_mhz':float(fields[2]), 'gpu_mem_mhz':float(fields[3]), 'gpu_watts':float(fields[4]), 'gpu_celsius':float(fields[5])}
                    row['cpu_khz'] = {}
                    for cpu in (0, 1):
                        freq = pathlib.Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq')
                        if freq.exists(): row['cpu_khz'][str(cpu)] = int(freq.read_text())
                    if cg:
                        for field in ('memory.current','memory.peak','memory.swap.current','memory.stat','cpu.stat'):
                            row[field] = (base/field).read_text().strip()
                        row['process_memory'] = {}
                        for pid in (base/'cgroup.procs').read_text().split():
                            try:
                                row['process_memory'][pid] = '\n'.join(v for v in pathlib.Path('/proc',pid,'status').read_text().splitlines() if v.startswith(('Name:','VmRSS:','RssAnon:','RssFile:','RssShmem:','VmSwap:')))
                            except FileNotFoundError: pass
                    f.write(json.dumps(row)+'\n'); f.flush()
                except Exception as e: f.write(json.dumps({'monitor_error':str(e)})+'\n')
                stop.wait(1)
    threading.Thread(target=read, daemon=True).start()
    mon = threading.Thread(target=monitor, daemon=True); mon.start()
    def run(case, rep):
        ids = case['ids']; started_unix = time.time(); t = time.monotonic(); out = []; ttft = None; token_times = []
        proc.stdin.write('GEN '+str(a.new_tokens)+' '+','.join(map(str, ids))+'\n'); proc.stdin.flush()
        while True:
            line = lines.get(timeout=1200)
            if line is None or line.startswith('ERR'): raise RuntimeError(str(line))
            if line.startswith('T '):
                token_times.append(time.monotonic()-t)
                if ttft is None: ttft = token_times[-1]
                token = int(line.split()[1]); out.append(token)
                if token == tk.ids.get('<|im_end|>'):
                    proc.stdin.write('STOP\n'); proc.stdin.flush()
            if line.startswith('DONE '):
                fields = line.split()
                entry = {'started_unix':started_unix, 'name':case['name'], 'rep':rep, 'prompt_tokens':len(ids), 'output_ids':out, 'output':tk.decode(out), 'wall_s':time.monotonic()-t, 'ttft_s':ttft, 'done':line,
                         'generated':int(fields[1]), 'prompt_ms':float(fields[3]), 'decode_ms':float(fields[4]), 'reused':int(fields[8]), 'drafts_accepted':int(fields[6]), 'drafts_offered':int(fields[7]), 'token_times_s':token_times}
                assert entry['reused'] == 0, entry
                entry['pp_tps'] = len(ids)*1000/entry['prompt_ms']
                entry['tg_tps'] = entry['generated']*1000/entry['decode_ms'] if entry['decode_ms'] else None
                return entry
    try:
        while True:
            line = lines.get(timeout=900)
            if line is None: raise RuntimeError('Exited before READY')
            if line.startswith('READY'): break
        result['startup_s'] = time.monotonic()-started
        print('READY', result['startup_s'], flush=True)
        result['warmup'] = run(cases[0], -1)
        # Identical full requests cannot extend prior generated state, so the engine resets.
        # The reuse counter is asserted on every request; no cached timing can slip into results.
        for rep in range(a.repeats):
            for case in cases:
                entry = run(case, rep); result['tests'].append(entry)
                (a.output/'results.json').write_text(json.dumps(result, indent=2))
                print(json.dumps({k:v for k,v in entry.items() if k not in ('output','output_ids','token_times_s')}), flush=True)
        proc.stdin.close(); proc.wait(timeout=60)
        result['exit_code'] = proc.returncode
        if proc.returncode: raise RuntimeError('Nonzero engine exit')
    except BaseException as e:
        result['error'] = repr(e)
        raise
    finally:
        # Stop the complete scope, including profiler-launched children if its launcher exited first.
        subprocess.run(['systemctl','--user','stop',unit+'.scope'], capture_output=True)
        stop.set(); mon.join(timeout=5); log.close()
        result['wall_s'] = time.monotonic()-started
        (a.output/'results.json').write_text(json.dumps(result, indent=2))

if __name__ == '__main__': main()
