# Summarize a bench.sh result directory: <out>/<backend>/{run.json,monitor.csv} + <out>/host.json
# Writes <out>/results.md and <out>/results.json.
# usage: python3 summarize.py <out_dir>
import csv, json, os, re, statistics, sys

out = sys.argv[1]
host = json.load(open(os.path.join(out, 'host.json')))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def stats(rows, key, fn):
    vals = [v for v in (num(r[key]) for r in rows) if v is not None]
    return round(fn(vals), 1) if vals else None


backends = {}
for b in ('rocm', 'vulkan'):
    rj = os.path.join(out, b, 'run.json')
    if not os.path.exists(rj):
        continue
    run = json.load(open(rj))
    rows = list(csv.DictReader(open(os.path.join(out, b, 'monitor.csv'))))
    p = run['prompts']
    err = run.get('error')
    logf = os.path.join(out, b, 'comfyui.log')
    if err and os.path.exists(logf):  # prefer the exception line from ComfyUI's log over the API dump
        exc = re.findall(r'^\S*(?:Error|Exception): .+$', open(logf, errors='replace').read(), re.M)
        if exc:
            err = f"image {len(run['prompts']) + 1} failed: {exc[-1][:400]}"
    s = {'status': run['status'], 'error': err, 'device': run.get('device'),
         'versions': run.get('versions'), 'startup_s': run.get('startup_s'), 'images_ok': len(p),
         'runs': run['runs'], 'fallbacks': run.get('torchvulkan_fallbacks', []),
         'fallback_counts': run.get('fallback_counts'), 'warn_always': run.get('warn_always', False)}
    if p:
        s['cold_s'] = p[0]['seconds']
        warm = [x['seconds'] for x in p[1:]]
        if warm:
            s['warm_median_s'] = round(statistics.median(warm), 2)
            s['warm_mean_s'] = round(statistics.mean(warm), 2)
            s['warm_min_s'], s['warm_max_s'] = min(warm), max(warm)
        spi = [x['s_per_it'] for x in p[1:] if x.get('s_per_it')] or [x['s_per_it'] for x in p if x.get('s_per_it')]
        s['s_per_it_median'] = round(statistics.median(spi), 2) if spi else None
        s['total_gen_s'] = round(p[-1]['end'] - p[0]['start'], 1)
        win = [r for r in rows if p[0]['start'] <= float(r['t']) <= p[-1]['end']]
    else:
        win = rows
    def first(key):
        return next((num(r[key]) for r in rows if num(r[key]) is not None), None)
    idle = rows[0] if rows else {}
    s['window_samples'] = len(win)
    s['gpu_busy_avg_pct'] = stats(win, 'gpu_busy_pct', statistics.mean)
    s['vram_peak_mb'] = stats(win or rows, 'vram_mb', max)
    s['vram_idle_mb'] = num(idle.get('vram_mb'))
    s['gtt_peak_mb'] = stats(win or rows, 'gtt_mb', max)
    s['rss_peak_mb'] = stats(rows, 'rss_tree_mb', max)
    s['mem_avail_min_mb'] = stats(rows, 'mem_avail_mb', min)
    s['swap_delta_mb'] = (stats(rows, 'swap_used_mb', max) or 0) - (num(idle.get('swap_used_mb')) or 0)
    s['cpu_tree_avg_pct'] = stats(win, 'cpu_tree_pct', statistics.mean)
    s['cpu_tree_max_pct'] = stats(win, 'cpu_tree_pct', max)
    s['cpu_host_avg_pct'] = stats(win, 'cpu_host_pct', statistics.mean)
    s['cpu_temp_max_c'] = stats(win, 'cpu_temp_c', max)
    for k in ('gpu_edge_c', 'gpu_junction_c', 'gpu_mem_c', 'gpu_power_w', 'gpu_sclk_mhz'):
        s[k + '_avg'] = stats(win, k, statistics.mean)
        s[k + '_max'] = stats(win, k, max)
    s['gpu_edge_c_idle'] = first('gpu_edge_c')
    backends[b] = s

json.dump({'host': host, 'backends': backends}, open(os.path.join(out, 'results.json'), 'w'), indent=1)

R = [b for b in ('rocm', 'vulkan') if b in backends]


def row(label, *keys):
    cells = []
    for b in R:
        vals = ['–' if backends[b].get(k) is None else str(backends[b][k]) for k in keys]
        cells.append(' / '.join(vals))
    return f'| {label} | ' + ' | '.join(cells) + ' |'


lines = [f"# torchvulkan vs ROCm in ComfyUI — {host['gpu']}", '',
         f"Host: {host['cpu']}, {host['ram_gb']} GB RAM, kernel {host['kernel']}, podman {host['podman']}. "
         f"{host['runs']} images per backend, {host['res']}x{host['res']}, {host['steps']} steps, "
         f"Z-Image Turbo Q6_K GGUF + Qwen3-4B-Instruct-2507 Q8_0 GGUF text encoder + ae VAE. "
         f"ComfyUI args: `{host['comfy_args']}`.", '',
         '| | ' + ' | '.join(R) + ' |', '|---|' + '---|' * len(R),
         row('status', 'status'), row('images OK', 'images_ok'), row('device', 'device'),
         row('first image (cold), s', 'cold_s'), row('warm image median, s', 'warm_median_s'),
         row('warm image min / max, s', 'warm_min_s', 'warm_max_s'),
         row('sampling, s/it (median)', 's_per_it_median'), row('all images, s', 'total_gen_s'),
         row('GPU busy avg, %', 'gpu_busy_avg_pct'), row('VRAM start / peak, MB', 'vram_idle_mb', 'vram_peak_mb'),
         row('GTT peak, MB', 'gtt_peak_mb'), row('process RSS peak, MB', 'rss_peak_mb'),
         row('host MemAvailable min, MB', 'mem_avail_min_mb'),
         row('ComfyUI CPU avg / max, % of one core', 'cpu_tree_avg_pct', 'cpu_tree_max_pct'),
         row('host CPU avg, %', 'cpu_host_avg_pct'),
         row('GPU edge temp start / avg / max, °C', 'gpu_edge_c_idle', 'gpu_edge_c_avg', 'gpu_edge_c_max'),
         row('GPU junction temp avg / max, °C', 'gpu_junction_c_avg', 'gpu_junction_c_max'), row('GPU mem temp avg / max, °C', 'gpu_mem_c_avg', 'gpu_mem_c_max'),
         row('GPU power avg / max, W', 'gpu_power_w_avg', 'gpu_power_w_max'),
         row('GPU sclk avg, MHz', 'gpu_sclk_mhz_avg'), row('CPU temp max, °C', 'cpu_temp_max_c'), '']
for b in R:
    if backends[b].get('error'):
        lines.append(f"**{b} error:** `{backends[b]['error']}`")
    fb = [x.strip() for x in backends[b].get('fallbacks') or [] if 'Out of memory' not in x]
    if b == 'vulkan':
        if backends[b].get('warn_always') and backends[b].get('fallback_counts') is not None:
            fc = backends[b]['fallback_counts']
            lines.append(f"**{b} CPU fallbacks (every occurrence logged, {backends[b]['images_ok']} images):** "
                         + ('; '.join(f'`{k}` × {v}' for k, v in fc.items()) if fc else 'none'))
        elif fb:
            lines.append(f"**{b} CPU fallbacks logged:** " + '; '.join(f'`{x}`' for x in fb))
lines.append('')
KEEP = ('backend=', 'comfyui=', 'gguf_node=', 'python ', 'vulkan-radeon', 'torch ', 'torchvision ', 'torchvulkan ',
        'triton', 'comfy-kitchen', 'comfy-aimdo')
for b in R:
    v = [' '.join(x.split()) for x in backends[b].get('versions') or [] if x.startswith(KEEP)]
    lines.append(f"{b} versions: " + ', '.join(v) + '  ')
open(os.path.join(out, 'results.md'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
