# Host-side sampler: one CSV row per second until <root_pid> exits.
# GPU (amdgpu sysfs/hwmon): busy %, VRAM, GTT, edge/junction/mem temperature, power, sclk, fan.
# CPU: host-wide busy %, CPU % of the container's process tree, CPU temperature.
# Memory: RSS of the process tree, host MemAvailable, swap in use.
# usage: python3 monitor.py <root_pid> <out.csv>
import csv, glob, os, sys, time

root, out = int(sys.argv[1]), sys.argv[2]
HZ = os.sysconf('SC_CLK_TCK')
PAGE = os.sysconf('SC_PAGE_SIZE')


def rd(p, default=None):
    try:
        return int(open(p).read().split()[0])
    except (OSError, ValueError, IndexError):
        return default


def gpu_card():
    best = None
    for d in glob.glob('/sys/class/drm/card[0-9]*/device'):
        total = rd(d + '/mem_info_vram_total')
        if total and (best is None or total > best[0]):
            best = (total, d)
    if best is None:
        sys.exit('no amdgpu card with mem_info_vram_total found')
    return best[1]


CARD = gpu_card()
HWMON = (glob.glob(CARD + '/hwmon/hwmon*') or [None])[0]


def hwmon_by_label(base, prefix, label):
    for p in glob.glob(f'{base}/{prefix}*_label'):
        if open(p).read().strip().lower() == label:
            return p[:-len('_label')] + '_input'
    return None


GPU_T = {k: hwmon_by_label(HWMON, 'temp', k) if HWMON else None for k in ('edge', 'junction', 'mem')}
GPU_SCLK = hwmon_by_label(HWMON, 'freq', 'sclk') if HWMON else None
GPU_POWER = None
if HWMON:
    for n in ('power1_average', 'power1_input'):
        if os.path.exists(f'{HWMON}/{n}'):
            GPU_POWER = f'{HWMON}/{n}'
            break
CPU_T = None
for h in glob.glob('/sys/class/hwmon/hwmon*'):
    name = open(h + '/name').read().strip()
    if name in ('k10temp', 'zenpower'):
        CPU_T = hwmon_by_label(h, 'temp', 'tctl') or f'{h}/temp1_input'
    elif name == 'coretemp' and CPU_T is None:
        CPU_T = hwmon_by_label(h, 'temp', 'package id 0') or f'{h}/temp1_input'
if CPU_T is None:  # e.g. no k10temp module loaded: use a board sensor labelled "CPU" (asus-ec etc.)
    for h in glob.glob('/sys/class/hwmon/hwmon*'):
        CPU_T = CPU_T or hwmon_by_label(h, 'temp', 'cpu')


def host_cpu():
    f = [int(x) for x in open('/proc/stat').readline().split()[1:]]
    idle = f[3] + f[4]
    return sum(f), idle


def tree(pid):
    kids = {}
    for p in os.listdir('/proc'):
        if p.isdigit():
            try:
                pp = int(open(f'/proc/{p}/stat').read().rsplit(')', 1)[1].split()[1])
                kids.setdefault(pp, []).append(int(p))
            except (OSError, IndexError, ValueError):
                pass
    res, todo = [], [pid]
    while todo:
        x = todo.pop()
        res.append(x)
        todo += kids.get(x, [])
    return res


def proc_stats(pids):
    ticks = rss = 0
    for p in pids:
        try:
            f = open(f'/proc/{p}/stat').read().rsplit(')', 1)[1].split()
            ticks += int(f[11]) + int(f[12])
            rss += int(f[21]) * PAGE
        except (OSError, IndexError, ValueError):
            pass
    return ticks, rss


def meminfo():
    d = {}
    for line in open('/proc/meminfo'):
        k, v = line.split(':')
        d[k] = int(v.split()[0]) // 1024
    return d


cols = ['t', 'gpu_busy_pct', 'vram_mb', 'gtt_mb', 'gpu_edge_c', 'gpu_junction_c', 'gpu_mem_c', 'gpu_power_w',
        'gpu_sclk_mhz', 'gpu_fan_rpm', 'cpu_host_pct', 'cpu_tree_pct', 'cpu_temp_c', 'rss_tree_mb',
        'mem_avail_mb', 'swap_used_mb']
w = csv.writer(open(out, 'w', newline=''))
w.writerow(cols)
prev_host = host_cpu()
prev_tree = 0
prev_t = time.time()
while os.path.exists(f'/proc/{root}'):
    time.sleep(1)
    now = time.time()
    pids = tree(root)
    ticks, rss = proc_stats(pids)
    total_ticks = max(ticks, prev_tree)  # children that exited take their ticks with them
    h = host_cpu()
    dt_all, dt_idle = h[0] - prev_host[0], h[1] - prev_host[1]
    mi = meminfo()
    t = lambda p: None if p is None or rd(p) is None else round(rd(p) / 1000, 1)
    pw = rd(GPU_POWER) if GPU_POWER else None
    w.writerow([round(now, 2), rd(CARD + '/gpu_busy_percent'), (rd(CARD + '/mem_info_vram_used') or 0) >> 20,
                (rd(CARD + '/mem_info_gtt_used') or 0) >> 20, t(GPU_T['edge']), t(GPU_T['junction']),
                t(GPU_T['mem']), None if pw is None else round(pw / 1e6, 1),
                None if GPU_SCLK is None or rd(GPU_SCLK) is None else rd(GPU_SCLK) // 1000000,
                rd(HWMON + '/fan1_input') if HWMON else None,
                round(100 * (dt_all - dt_idle) / dt_all, 1) if dt_all else None,
                round(100 * (total_ticks - prev_tree) / HZ / (now - prev_t), 1),
                t(CPU_T), rss >> 20, mi['MemAvailable'], mi['SwapTotal'] - mi['SwapFree']])
    prev_host, prev_tree, prev_t = h, total_ticks, now
