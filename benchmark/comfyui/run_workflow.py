# Runs inside the container: starts ComfyUI, queues N Z-Image Turbo prompts one at a time
# (different seeds, so nothing comes from ComfyUI's cache) and writes /out/run.json.
import argparse, glob, json, os, re, signal, subprocess, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--backend', required=True, choices=['rocm', 'vulkan'])
ap.add_argument('--runs', type=int, default=10)
ap.add_argument('--res', type=int, default=1024)
ap.add_argument('--steps', type=int, default=8)
ap.add_argument('--unet', default='z_image_turbo-Q6_K.gguf')
ap.add_argument('--text-encoder', default='Qwen3-4B-Instruct-2507-Q8_0.gguf')
ap.add_argument('--vae', default='ae.safetensors')
ap.add_argument('--keep-images', type=int, default=1, help='keep the first N images in /out')
ap.add_argument('--out', default='/out')
ap.add_argument('--port', type=int, default=8188)
ap.add_argument('--stats', action='store_true',
                help='inside ComfyUI, write torchvulkan.dispatch_stats() to <out>/tvstats.jsonl every 2 s (debug builds)')
ap.add_argument('--warn-always', action='store_true',
                help='torch.set_warn_always(True): log every torchvulkan CPU fallback, not just the first')
ap.add_argument('comfy_args', nargs='*', help='extra ComfyUI arguments (after --)')
a = ap.parse_args()

PROMPT = 'a red fox standing in fresh snow, morning light, detailed photo'
base = f'http://127.0.0.1:{a.port}'
te_loader = 'ClipLoaderGGUF' if a.text_encoder.endswith('.gguf') else 'CLIPLoader'


def workflow(seed):
    return {
        '1': {'class_type': 'LoaderGGUF', 'inputs': {'gguf_name': a.unet}},
        '2': {'class_type': te_loader, 'inputs': {'clip_name': a.text_encoder, 'type': 'lumina2'}},
        '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': a.vae}},
        '4': {'class_type': 'ModelSamplingAuraFlow', 'inputs': {'model': ['1', 0], 'shift': 3.0}},
        '5': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['2', 0], 'text': PROMPT}},
        '6': {'class_type': 'ConditioningZeroOut', 'inputs': {'conditioning': ['5', 0]}},
        '7': {'class_type': 'EmptySD3LatentImage', 'inputs': {'width': a.res, 'height': a.res, 'batch_size': 1}},
        '8': {'class_type': 'KSampler', 'inputs': {'model': ['4', 0], 'seed': seed, 'steps': a.steps, 'cfg': 1.0,
              'sampler_name': 'res_multistep', 'scheduler': 'simple', 'positive': ['5', 0],
              'negative': ['6', 0], 'latent_image': ['7', 0], 'denoise': 1.0}},
        '9': {'class_type': 'VAEDecode', 'inputs': {'samples': ['8', 0], 'vae': ['3', 0]}},
        '10': {'class_type': 'SaveImage', 'inputs': {'images': ['9', 0], 'filename_prefix': f'bench-{a.backend}'}},
    }


def get(path, timeout=60):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.load(r)


def post(path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


TQDM_DONE = re.compile(r'100%\|[^|]*\|\s*(\d+)/(\d+) \[(\d+):(\d+)<')


def sampling_seconds(text):
    """Elapsed time of the last finished tqdm bar in a chunk of ComfyUI log."""
    hits = TQDM_DONE.findall(text.replace('\r', '\n'))
    if not hits:
        return None
    _, _, mm, ss = hits[-1]
    return int(mm) * 60 + int(ss)


os.makedirs(a.out, exist_ok=True)
log_path = os.path.join(a.out, 'comfyui.log')
comfy = ['main.py', '--listen', '127.0.0.1', '--port', str(a.port), '--disable-auto-launch',
         '--output-directory', a.out, '--temp-directory', '/tmp']
STATS_THREAD = """
import json, threading, time
def _tvstats(path):
    import torchvulkan
    while not hasattr(torchvulkan, 'dispatch_stats') or not torchvulkan.is_initialized():
        time.sleep(1)
    with open(path, 'a') as f:
        while True:
            row = dict(t=time.time(), **torchvulkan.dispatch_stats())
            if hasattr(torchvulkan, 'gpu_profile'):
                row['gpu_profile'] = torchvulkan.gpu_profile()
            f.write(json.dumps(row) + chr(10))
            f.flush()
            time.sleep(2)
threading.Thread(target=_tvstats, args=(%r,), daemon=True).start()
"""
if a.warn_always or a.stats:
    boot = 'import sys, runpy, torch\n'
    if a.warn_always:
        boot += 'torch.set_warn_always(True)\n'
    if a.stats and a.backend == 'vulkan':
        boot += STATS_THREAD % os.path.join(a.out, 'tvstats.jsonl')
    boot += 'sys.argv = sys.argv[1:]\nrunpy.run_path("main.py", run_name="__main__")\n'
    cmd = [sys.executable] + (['-W', 'always'] if a.warn_always else []) + ['-c', boot] + comfy
else:
    cmd = [sys.executable] + comfy
if a.backend == 'vulkan':
    cmd.append('--vulkan')
cmd += a.comfy_args
log = open(log_path, 'w')
srv = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
res = {'backend': a.backend, 'cmd': cmd[1:], 'runs': a.runs, 'res': a.res, 'steps': a.steps,
       'unet': a.unet, 'text_encoder': a.text_encoder, 'vae': a.vae, 'status': 'ok', 'prompts': [],
       'versions': open('/venv/versions.txt').read().splitlines(), 't_start': time.time()}
try:
    import torch
    if a.backend == 'vulkan':
        import torchvulkan
        res['device'] = torchvulkan.get_device_name()
    else:
        props = torch.cuda.get_device_properties(0)
        res['device'] = f"{props.name} ({getattr(props, 'gcnArchName', '?')})"
except Exception as e:
    res['device'] = f'unknown ({e})'
print(f"[bench] {a.backend}: {res['device']}", flush=True)

t0 = time.time()
try:
    while True:
        if srv.poll() is not None:
            raise RuntimeError(f'ComfyUI exited with {srv.returncode} during startup')
        try:
            get('/system_stats', timeout=5)
            break
        except Exception:
            time.sleep(1)
        if time.time() - t0 > 900:
            raise RuntimeError('ComfyUI not ready after 900 s')
    res['startup_s'] = round(time.time() - t0, 1)
    for i in range(a.runs):
        seed = 1000 + i
        log_pos = os.path.getsize(log_path)
        t = time.time()
        pid = post('/prompt', {'prompt': workflow(seed), 'client_id': 'bench'})['prompt_id']
        while True:
            if srv.poll() is not None:
                raise RuntimeError(f'ComfyUI exited with {srv.returncode} during prompt {i}')
            try:
                h = get(f'/history/{pid}')
            except Exception:
                time.sleep(1)
                continue
            if pid in h:
                st = h[pid].get('status', {})
                if st.get('status_str') != 'success':
                    errs = [m[1] for m in st.get('messages', []) if m and m[0] == 'execution_error']
                    why = (f"{errs[0].get('node_type')}: {errs[0].get('exception_type')}: "
                           f"{errs[0].get('exception_message', '').strip()}") if errs else json.dumps(st)[-400:]
                    raise RuntimeError(f'prompt {i} failed: {why}'[:800])
                break
            time.sleep(0.5)
        end = time.time()
        with open(log_path, errors='replace') as f:
            f.seek(log_pos)
            chunk = f.read()
        samp = sampling_seconds(chunk)
        res['prompts'].append({'i': i, 'seed': seed, 'start': t, 'end': end, 'seconds': round(end - t, 2),
                               'sampling_s': samp,
                               's_per_it': round(samp / a.steps, 2) if samp is not None else None})
        print(f"[bench] {a.backend} prompt {i + 1}/{a.runs}: {end - t:.1f} s (sampling {samp} s)", flush=True)
except Exception as e:
    res['status'] = 'fail'
    res['error'] = str(e)[-800:]
    print(f'[bench] {a.backend} FAILED: {e}', flush=True)
finally:
    srv.send_signal(signal.SIGINT)
    try:
        srv.wait(60)
    except subprocess.TimeoutExpired:
        srv.kill()
        srv.wait()
    res['t_end'] = time.time()
    text = open(log_path, errors='replace').read()
    res['torchvulkan_fallbacks'] = sorted(set(re.findall(r'torchvulkan \[(?:NOT IMPLEMENTED|WARNING)\]: ([^(\n]+)', text)))
    counts = {}
    for op in re.findall(r'Silent fallback detected for operation: ([\w.:]+)', text):
        op = op.rstrip('.')
        counts[op] = counts.get(op, 0) + 1
    for msg in re.findall(r'torchvulkan \[WARNING\]: ([^(\n]+)', text):
        counts[msg.strip()] = counts.get(msg.strip(), 0) + 1
    res['fallback_counts'] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    res['warn_always'] = a.warn_always
    for n, p in enumerate(sorted(glob.glob(os.path.join(a.out, f'bench-{a.backend}_*.png')))):
        if n >= a.keep_images:
            os.remove(p)
    json.dump(res, open(os.path.join(a.out, 'run.json'), 'w'), indent=1)
sys.exit(0 if res['status'] == 'ok' else 1)
