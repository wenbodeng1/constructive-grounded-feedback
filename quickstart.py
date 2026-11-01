#!/usr/bin/env python3
"""Quick start: reproduce one cell of Table II with one command.

Default setting: +CGF on NATURAL PLAN calendar scheduling with Qwen3.5-9B, 82.7 % strict success on 1,000 tasks.

    python quickstart.py                                   # one GPU with >= 24 GB, all 1,000 tasks
    python quickstart.py --tp 2 --vllm-args="--language-model-only --gpu-memory-utilization 0.85"   # two 16 GB GPUs
    python quickstart.py --limit 100                       # the first 100 tasks only
    python quickstart.py --methods direct cgf              # Direct as well (paper: 34.6 %)

What it does:
  1. checks the environment (Python, absl-py, released plans);
  2. fetches NATURAL PLAN at the commit used in the paper and verifies the SHA-256 of the task file and of the
     official evaluators (re-scoring all 48 released NATURAL PLAN runs with these files reproduces every outcome);
  3. starts a vLLM server with the flags from the README, or reuses one that already serves the model;
  4. runs np/np_run.py (greedy decoding, enable_thinking=false, r = 3 repair rounds, official evaluator);
  5. pairs every task with the released plans in results/runs.tar.xz (exact McNemar test, Wilson 95 % CI) and
     writes report.md and manifest.json (code, data, model, server, GPU and cost provenance).

A finished run is reused instead of overwritten (--force reruns it). The exit status is 0 when every method agrees
with the paper (McNemar p >= 0.05), 3 when one differs, and 1 on errors. Needs only the standard library and
absl-py (used by the official evaluator); vLLM may live in another environment (--vllm /path/to/bin/vllm).
"""
import argparse, datetime, hashlib, json, math, os, platform, re, shlex, shutil, signal, subprocess, sys, tarfile, threading, time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / 'analysis'))
from aggregate import sign_p  # noqa: E402  (the exact McNemar / sign test used for the paper)

NP_URL = 'https://github.com/google-deepmind/natural-plan'
NP_COMMIT = 'ca76db336072ff8931db43bc1ca8d381038cf073'
NP_SHA256 = {'data/calendar_scheduling.json': '3f34ff275b22202e16f750560f39805860378ba30f4e15cd80cf37986bcd484d',
             'evaluate_calendar_scheduling.py': '94ac5d86ef051fd54cb3eb765dd3f82900cc2f047a6484c245566d555c7581ea',
             'evaluate_meeting_planning.py': '0227d6ecefe48881e25b073127681c48e0835f837c77bea354f9b3aaa959c5a8',
             'evaluate_trip_planning.py': 'a4544b15abab82caa0dd65ea2706f6df19a93b7128792ecb5ffd5acc0bfccd13'}
N_TASKS = 1000
MODELS = {'q35': ('Qwen/Qwen3.5-9B', 'qwen3.5-9b', 'Qwen3.5-9B'),
          'qvl': ('Qwen/Qwen3-VL-8B-Instruct', 'qwen3-vl-8b', 'Qwen3-VL-8B'),
          'hermes': ('NousResearch/Hermes-3-Llama-3.1-8B', 'hermes-3-8b', 'Hermes-3-8B')}
METHODS = {'direct': 'Direct', 'check': '+Check', 'cgf': '+CGF', 'cgf_unranked': '+CGF, unranked'}
STRUCTURED = '{"backend": "xgrammar", "disable_any_whitespace": true}'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # the server is local: never go through a proxy


class Fail(Exception):
    pass


def say(msg=''): print(msg, flush=True)


def stamp(t=None): return datetime.datetime.fromtimestamp(t or time.time()).astimezone().isoformat(timespec='seconds')


def step(i, msg): say(f'\n[{i}/5] {msg}')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def sh(*cmd, cwd=None):
    try: return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError): return ''


def pval(p): return f'{p:.2f}' if p >= 0.01 else f'{p:.1e}'


def rel(p):
    try: return str(Path(p).relative_to(REPO))
    except ValueError: return str(p)


def http(port, path, timeout=5):
    with OPENER.open(f'http://127.0.0.1:{port}{path}', timeout=timeout) as r: return r.read().decode()


# ---------------------------------------------------------------- 1. environment
def check_env():
    if sys.version_info < (3, 8): raise Fail('Python 3.8+ is required')
    try: import absl  # noqa: F401
    except ImportError: raise Fail(f'absl-py is missing in {sys.executable}: pip install absl-py')
    if not (REPO / 'results' / 'runs.tar.xz').is_file(): raise Fail('results/runs.tar.xz is missing; clone the full repository')
    say(f'      Python {platform.python_version()} ({sys.executable}), absl-py ok, released plans ok')


# ---------------------------------------------------------------- 2. data
def prepare_data(root):
    if not root.exists():
        say(f'      cloning {NP_URL} @ {NP_COMMIT[:7]} into {root}')
        for cmd in (['git', 'clone', '--quiet', NP_URL, str(root)], ['git', '-C', str(root), 'checkout', '--quiet', NP_COMMIT]):
            if subprocess.run(cmd).returncode: raise Fail('could not fetch NATURAL PLAN: ' + ' '.join(cmd))
    bad = [f for f, h in NP_SHA256.items() if not (root / f).is_file() or sha256(root / f) != h]
    if bad: raise Fail(f'{root} differs from NATURAL PLAN @ {NP_COMMIT[:7]} used in the paper ({", ".join(bad)}); '
                       'delete it so that it is fetched again, or point --np-root to a clean copy')
    n = len(json.load(open(root / 'data' / 'calendar_scheduling.json')))
    if n != N_TASKS: raise Fail(f'expected {N_TASKS} calendar tasks, found {n}')
    say(f'      NATURAL PLAN @ {NP_COMMIT[:7]}: {n} calendar tasks; task file and official evaluators match SHA-256')
    return dict(url=NP_URL, pinned_commit=NP_COMMIT, checkout=sh('git', 'rev-parse', 'HEAD', cwd=root) if (root / '.git').exists() else None,
                sha256=NP_SHA256, calendar_tasks=n)


# ---------------------------------------------------------------- 3. server
class Server:
    """Reuses a vLLM server that already serves `served` on `port`, or starts one and stops it on exit."""

    def __init__(self, a, log):
        self.a, self.log, self.proc, self.cmd, self.logf = a, log, None, None, None

    def models(self):
        try: return json.loads(http(self.a.port, '/v1/models'))['data']
        except OSError: return None

    def __enter__(self):
        a, found = self.a, self.models()
        if found is not None:
            m = next((m for m in found if m['id'] == a.served_name), None)
            if m is None: raise Fail(f'port {a.port} already serves {[m["id"] for m in found]}; stop it or use --port')
            if (m.get('max_model_len') or a.max_model_len) < a.max_model_len:
                raise Fail(f'the server on port {a.port} has max_model_len={m["max_model_len"]} < {a.max_model_len}')
            self.mode = 'reused'
            say(f'      reusing the vLLM server on port {a.port} ({a.served_name})')
        else:
            exe = shutil.which(a.vllm) or (a.vllm if os.access(a.vllm, os.X_OK) else None)
            if not exe: raise Fail(f'vLLM executable "{a.vllm}" not found: pip install "vllm==0.17.1" or pass --vllm /path/to/bin/vllm')
            self.cmd = [exe, 'serve', a.model, '--served-model-name', a.served_name, '--host', '127.0.0.1', '--port', str(a.port),
                        '--tensor-parallel-size', str(a.tp), '--dtype', 'bfloat16', '--max-model-len', str(a.max_model_len),
                        '--max-num-seqs', str(a.workers), '--structured-outputs-config', STRUCTURED, *shlex.split(a.vllm_args)]
            say('      starting: ' + shlex.join(self.cmd))
            say(f'      server log: {rel(self.log)} (the first start downloads the model)')
            self.log.parent.mkdir(parents=True, exist_ok=True)
            self.logf = open(self.log, 'w')
            self.proc = subprocess.Popen(self.cmd, stdout=self.logf, stderr=subprocess.STDOUT, start_new_session=True)
            t0 = time.time()
            try:  # __exit__ does not run when __enter__ raises: stop the server here on errors and Ctrl+C while it loads
                while True:
                    try:
                        http(a.port, '/health'); break
                    except OSError: pass
                    if self.proc.poll() is not None:
                        tail = ''.join(open(self.log, errors='replace').readlines()[-25:])
                        hint = ('\nvLLM could not reach the Hugging Face Hub. If the model is already downloaded, rerun with '
                                'HF_HUB_OFFLINE=1 or pass --model /path/to/snapshot.') if 'huggingface.co' in tail else ''
                        raise Fail(f'vLLM exited with code {self.proc.returncode}; last lines of {rel(self.log)}:\n{tail}{hint}')
                    if time.time() - t0 > a.server_timeout: raise Fail(f'vLLM was not ready after {a.server_timeout} s; see {rel(self.log)}')
                    time.sleep(5)
            except BaseException:
                self.stop(force=True); raise
            self.mode = 'started'
            say(f'      server ready after {time.time() - t0:.0f} s')
        m = next(m for m in self.models() if m['id'] == a.served_name)
        try: version = json.loads(http(a.port, '/version'))['version']
        except (OSError, ValueError, KeyError): version = None
        self.info = dict(mode=self.mode, vllm_version=version, root=m.get('root'), max_model_len=m.get('max_model_len'), command=self.cmd)
        say(f'      vLLM {version}, model root {m.get("root")}, max_model_len {m.get("max_model_len")}')
        return self

    def __exit__(self, *exc): self.stop()

    def stop(self, force=False):
        if self.proc and self.proc.poll() is None and (force or not self.a.keep_server):
            say('      stopping the vLLM server')
            os.killpg(self.proc.pid, signal.SIGTERM)
            try: self.proc.wait(60)
            except subprocess.TimeoutExpired: os.killpg(self.proc.pid, signal.SIGKILL)
        if self.logf: self.logf.close()

    def finished_calls(self):
        try: return sum(float(l.rsplit(' ', 1)[1]) for l in http(self.a.port, '/metrics').splitlines() if l.startswith('vllm:request_success_total'))
        except (OSError, ValueError, IndexError): return None


# ---------------------------------------------------------------- 4. run
def finished(out, a, n):
    meta = Path(str(out) + '.meta.json')
    if not (out.is_file() and meta.is_file()): return False
    m = json.loads(meta.read_text())
    return m.get('n') == n == sum(1 for _ in open(out)) and m.get('model') == a.served_name


def run_method(a, method, out, server):
    cmd = [sys.executable, '-u', str(REPO / 'np' / 'np_run.py'), '--task', 'calendar', '--method', method, '--model', a.served_name,
           '--ports', str(a.port), '--workers', str(a.workers), '--out', str(out)] + (['--limit', str(a.limit)] if a.limit else [])
    env = dict(os.environ, NP_ROOT=str(a.np_root))
    t0, base, done, tty, shown = time.time(), server.finished_calls() or 0, threading.Event(), sys.stdout.isatty(), []

    def progress():
        while not done.wait(10 if tty else 60):
            shown.append(1)
            calls = server.finished_calls()
            msg = f'      {METHODS[method]}: {datetime.timedelta(seconds=int(time.time() - t0))} elapsed' + (f', {calls - base:.0f} model calls finished' if calls is not None else '')
            print('\r' + msg if tty else msg, end='' if tty else '\n', flush=True)
    th = threading.Thread(target=progress, daemon=True); th.start()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, text=True)
    try: stdout, _ = proc.communicate()
    finally:
        done.set(); th.join()
        if tty and shown: say()
        if proc.poll() is None: proc.terminate()
    if proc.returncode: raise Fail(f'np/np_run.py failed with exit code {proc.returncode}: {" ".join(cmd)}')
    Path(str(out) + '.provenance.json').write_text(json.dumps(dict(
        server=server.info, model=dict(name=a.model, served_name=a.served_name, revision=hf_revision(a.model, server.info['root'])),
        gpus=gpus(), started=stamp(t0), finished=stamp(time.time()), command=cmd[1:]), indent=1))


# ---------------------------------------------------------------- 5. compare with the paper
def released(tag, name):
    with tarfile.open(REPO / 'results' / 'runs.tar.xz') as t:
        return t.extractfile(f'runs/naturalplan/{tag}/{name}').read().decode()


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, 100 * (c - h) / d), min(100.0, 100 * (c + h) / d)


def compare(a, method, out):
    ours = {r['id']: r['ok'] for r in map(json.loads, open(out))}
    paper = {r['id']: r['ok'] for r in map(json.loads, released(a.tag, f'calendar_{method}.jsonl').splitlines())}
    meta, pmeta = json.loads(Path(str(out) + '.meta.json').read_text()), json.loads(released(a.tag, f'calendar_{method}.jsonl.meta.json'))
    prov = Path(str(out) + '.provenance.json')
    n, k = len(ours), sum(ours.values())
    b = sum(paper[i] and not ours[i] for i in ours); c = sum(ours[i] and not paper[i] for i in ours)
    return dict(method=method, name=METHODS[method], n=n, acc=100 * k / n, ci95=wilson(k, n), paper_acc=100 * sum(paper[i] for i in ours) / n,
                paper_acc_all=pmeta['acc'], same=n - b - c, paper_only=b, ours_only=c, p=sign_p(b, c), calls=meta['calls'] / n,
                paper_calls=pmeta['calls'] / pmeta['n'], seconds=meta['seconds'], errors=meta['errors'],
                tokens=dict(prompt=meta['prompt'], completion=meta['completion']), output=rel(out),
                provenance=json.loads(prov.read_text()) if prov.is_file() else None)


# ---------------------------------------------------------------- provenance and report
def gpus():
    for exe in ('nvidia-smi', '/usr/lib/wsl/lib/nvidia-smi'):
        out = sh(exe, '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader')
        if out: return out.splitlines()
    return []


def hf_revision(model, root):
    m = re.search(r'/snapshots/([0-9a-f]{40})', str(root or '') + '/' + model)
    if m: return m.group(1)
    hub = Path(os.environ.get('HF_HUB_CACHE') or Path(os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface')) / 'hub')
    ref = hub / ('models--' + model.replace('/', '--')) / 'refs' / 'main'
    return ref.read_text().strip() if ref.is_file() else None


def report(a, rows, manifest, path):
    sub = f'the first {a.limit} tasks' if a.limit else f'all {N_TASKS} tasks'
    lines = [f'# Quick start: NATURAL PLAN calendar scheduling, {MODELS[a.tag][2]}, {sub}', '',
             '| Method | This run (95 % CI) | Paper, same tasks | Same outcome | Only paper / only this run | McNemar p | Calls per task (paper) |',
             '|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f'| {r["name"]} | {r["acc"]:.1f} % ({r["ci95"][0]:.1f}–{r["ci95"][1]:.1f}) | {r["paper_acc"]:.1f} % | {r["same"]}/{r["n"]} | '
                     f'{r["paper_only"]} / {r["ours_only"]} | {pval(r["p"])} | {r["calls"]:.2f} ({r["paper_calls"]:.2f}) |')
    lines += ['', f'Verdict: **{manifest["verdict"]}**', '', '## Provenance', '',
              f'- code: `{manifest["code"]["commit"]}`' + (' (with local changes)' if manifest['code']['dirty'] else ''),
              f'- data: NATURAL PLAN, task file and official evaluators identical to commit `{NP_COMMIT}` (SHA-256 verified)',
              f'- report: Python {manifest["client"]["python"]} on {manifest["client"]["platform"]}, {manifest["finished"]}']
    for r in rows:
        p = r['provenance']
        if p is None:
            lines.append(f'- {r["name"]}: `{r["output"]}`, not produced by quickstart.py, so model, server and GPUs were not recorded'); continue
        s, m = p['server'], p['model']
        lines += [f'- {r["name"]}: `{r["output"]}`, run {p["started"]} to {p["finished"]}',
                  f'  - model `{m["name"]}` served as `{m["served_name"]}`, revision `{m["revision"] or "unknown"}`',
                  f'  - vLLM {s["vllm_version"] or "unknown"}, server {s["mode"]}, max_model_len {s["max_model_len"]}'
                  + (f': `vllm {shlex.join(s["command"][1:])}`' if s['command'] else ''),
                  f'  - GPUs: {"; ".join(p["gpus"]) if p["gpus"] else "unknown"}']
    lines.append('')
    path.write_text('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], epilog=__doc__.split('\n\n', 1)[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--methods', nargs='+', default=['cgf'], choices=list(METHODS), help='methods to run (default: cgf)')
    ap.add_argument('--limit', type=int, help=f'run only the first N of the {N_TASKS} tasks')
    ap.add_argument('--tag', default='q35', choices=list(MODELS), help='paper model to reproduce: q35 = Qwen3.5-9B, qvl = Qwen3-VL-8B, hermes = Hermes-3-8B')
    ap.add_argument('--model', help='Hugging Face id or local path to serve (default: the paper model of --tag)')
    ap.add_argument('--served-name', help='API model name (default: qwen3.5-9b / qwen3-vl-8b / hermes-3-8b)')
    ap.add_argument('--tp', type=int, default=1, help='GPUs for tensor parallelism (default: 1)')
    ap.add_argument('--workers', type=int, default=32, help='concurrent requests, also vLLM --max-num-seqs (default: 32)')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--max-model-len', type=int, default=8192, help='context length; calendar prompt, draft, feedback and answer stay below 8k tokens')
    ap.add_argument('--vllm', default='vllm', help='vLLM executable (default: vllm on PATH)')
    ap.add_argument('--vllm-args', default='', help='extra vLLM flags, e.g. "--language-model-only --gpu-memory-utilization 0.85"')
    ap.add_argument('--server-timeout', type=int, default=3600, help='seconds to wait for the server, including the model download')
    ap.add_argument('--keep-server', action='store_true', help='leave a server started here running')
    ap.add_argument('--np-root', type=Path, default=Path(os.environ.get('NP_ROOT', REPO / 'third_party' / 'natural-plan')))
    ap.add_argument('--out-dir', type=Path, default=REPO / 'runs' / 'quickstart')
    ap.add_argument('--force', action='store_true', help='rerun methods that already have a finished run')
    a = ap.parse_args()
    a.model = a.model or MODELS[a.tag][0]; a.served_name = a.served_name or MODELS[a.tag][1]; a.vllm = os.path.expanduser(a.vllm)
    a.np_root, a.out_dir = a.np_root.resolve(), a.out_dir.resolve()
    if a.limit is not None and not 0 < a.limit <= N_TASKS: ap.error(f'--limit must be in 1..{N_TASKS}')
    n = a.limit or N_TASKS; suffix = f'_first{a.limit}' if a.limit else ''
    outdir = a.out_dir / a.tag; outdir.mkdir(parents=True, exist_ok=True)
    outs = {m: outdir / f'calendar_{m}{suffix}.jsonl' for m in dict.fromkeys(a.methods)}
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    started = stamp()
    say(f'Reproducing NATURAL PLAN calendar scheduling, {MODELS[a.tag][2]}, {", ".join(METHODS[m] for m in outs)}, {n} tasks')

    step(1, 'environment'); check_env()
    step(2, 'data'); data = prepare_data(a.np_root)
    todo = [m for m, o in outs.items() if a.force or not finished(o, a, n)]
    for m in outs:
        if m not in todo: say(f'      {METHODS[m]}: finished run found at {rel(outs[m])} (pass --force to rerun it)')
    if todo:
        step(3, 'model server')
        with Server(a, outdir / f'vllm{suffix}.log') as server:
            step(4, f'running {", ".join(METHODS[m] for m in todo)} on {n} tasks (2x RTX 4080: about 4 min per 100 tasks for +CGF)')
            for m in todo: run_method(a, m, outs[m], server)
    else: step(3, 'model server: not needed'); step(4, 'run: not needed')

    step(5, 'comparison with the released plans of the paper')
    rows = [compare(a, m, o) for m, o in outs.items()]
    table = [['method', 'this run (95% CI)', 'paper', 'same outcome', 'McNemar p', 'calls/task (paper)']] + [
        [r['name'], f'{r["acc"]:.1f} % ({r["ci95"][0]:.1f}-{r["ci95"][1]:.1f})', f'{r["paper_acc"]:.1f} %', f'{r["same"]}/{r["n"]}',
         pval(r['p']), f'{r["calls"]:.2f} ({r["paper_calls"]:.2f})'] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
    for row in table: say('      ' + '   '.join(c.ljust(w) for c, w in zip(row, widths)))
    ok = all(r['p'] >= 0.05 for r in rows)
    verdict = ('consistent with the paper: no significant paired difference (exact McNemar, p >= 0.05)' if ok else
               'differs from the paper for ' + ', '.join(r['name'] for r in rows if r['p'] < 0.05) + ' (exact McNemar, p < 0.05)')
    if ok and n < 200: verdict += f'; with {n} tasks the test only detects large differences, run all {N_TASKS} for the full check'
    manifest = dict(setting=dict(benchmark='NATURAL PLAN calendar scheduling', model=MODELS[a.tag][2], tag=a.tag, tasks=n, methods=list(outs)),
                    verdict=verdict, results=rows, started=started, finished=stamp(),
                    code=dict(commit=sh('git', 'rev-parse', 'HEAD', cwd=REPO), dirty=bool(sh('git', 'status', '--porcelain', cwd=REPO))),
                    data=data, client=dict(python=platform.python_version(), platform=platform.platform()), args={k: str(v) for k, v in vars(a).items()})
    (outdir / f'manifest{suffix}.json').write_text(json.dumps(manifest, indent=1))
    report(a, rows, manifest, outdir / f'report{suffix}.md')
    say(f'\n      verdict: {verdict}\n      report:  {rel(outdir / f"report{suffix}.md")}\n      details: {rel(outdir / f"manifest{suffix}.json")}')
    return 0 if ok else 3


if __name__ == '__main__':
    try: sys.exit(main())
    except Fail as e: say(f'\nerror: {e}'); sys.exit(1)
    except KeyboardInterrupt: say('\ninterrupted'); sys.exit(130)
