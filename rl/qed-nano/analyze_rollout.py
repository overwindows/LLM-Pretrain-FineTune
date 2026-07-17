#!/usr/bin/env python3
"""Analyze rollout ablation: group reward std/mean + training curves.
Usage: python analyze_rollout.py <results_dir> <label>
"""
import sys, json, os, re, statistics as st

rdir = sys.argv[1]
label = sys.argv[2]

# ---- 1. group-level reward std/mean from actor stream ----
actor = os.path.join(rdir, "streams/actor/0/0/0.jsonl")
group_sizes, group_means, group_stds, zero_std_groups = [], [], [], 0
n_groups = 0
with open(actor) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            grp = json.loads(line)
        except Exception:
            continue
        if not isinstance(grp, list):
            continue
        rewards = []
        for s in grp:
            r = s.get("reward")
            if r is None and isinstance(s.get("metrics"), dict):
                r = s["metrics"].get("reward")
            if r is not None:
                rewards.append(float(r))
        if len(rewards) < 2:
            continue
        n_groups += 1
        group_sizes.append(len(rewards))
        group_means.append(sum(rewards) / len(rewards))
        sd = st.pstdev(rewards)
        group_stds.append(sd)
        if sd < 1e-9:
            zero_std_groups += 1

def summ(xs):
    if not xs:
        return "n/a"
    return f"mean={sum(xs)/len(xs):.4f} median={st.median(xs):.4f} min={min(xs):.4f} max={max(xs):.4f}"

print(f"\n===== {label} : GROUP REWARD stats (from actor stream) =====")
print(f"  n_groups={n_groups}  avg_group_size={sum(group_sizes)/len(group_sizes):.1f}" if group_sizes else "  no groups")
if group_stds:
    print(f"  group reward STD : {summ(group_stds)}")
    print(f"  group reward MEAN: {summ(group_means)}")
    frac_zero = zero_std_groups / n_groups
    print(f"  DEGENERATE groups (std~0, no learning signal): {zero_std_groups}/{n_groups} = {frac_zero:.1%}")
    print(f"  USABLE groups (std>0): {1-frac_zero:.1%}")

# ---- 2. training curves from finetune info log ----
info = os.path.join(rdir, "finetune/log/info_0.log")
steps = []
pat = re.compile(r"Completed steps (\d+): (\{.*\})")
with open(info, errors="ignore") as f:
    for line in f:
        m = pat.search(line)
        if not m:
            continue
        step = int(m.group(1))
        # dict uses single quotes and string values; eval-safe parse
        raw = m.group(2)
        d = {}
        for km in re.finditer(r"'([^']+)':\s*'([-\d.eE+n]+|nan)'", raw):
            k, v = km.group(1), km.group(2)
            try:
                d[k] = float(v)
            except Exception:
                pass
        steps.append((step, d))

def curve(key):
    return [(s, d[key]) for s, d in steps if key in d]

print(f"\n===== {label} : TRAINING CURVES ({len(steps)} logged steps) =====")
for key in ["rl/reward", "rl/loss", "rl/entropy", "rl/kl", "rl/ess",
            "rl/advantage", "rl/max_advantage", "rl/min_advantage",
            "stats/grad_norm", "stats/lag", "stats/time_waiting_for_data",
            "throughput/tokens_per_sec"]:
    c = curve(key)
    if not c:
        continue
    vals = [v for _, v in c]
    first = vals[0]
    last = vals[-1]
    avg = sum(vals) / len(vals)
    # advantage spread = max-min average
    print(f"  {key:32s} first={first:9.4f} last={last:9.4f} mean={avg:9.4f}")

# advantage spread series
adv_max = dict(curve("rl/max_advantage"))
adv_min = dict(curve("rl/min_advantage"))
spreads = [adv_max[s] - adv_min[s] for s in adv_max if s in adv_min]
if spreads:
    print(f"  {'rl/advantage_SPREAD(max-min)':32s} mean={sum(spreads)/len(spreads):9.4f}")
