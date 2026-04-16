#!/usr/bin/env python3
"""
A100 GPU 간섭 실험 결과 분석 스크립트
Solo baseline 대비 pair 실행 시 성능 저하(slowdown)를 분석합니다.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "results"

def load_all_results():
    solos = {}
    pairs = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        with open(f) as fp:
            d = json.load(fp)
        d["_file"] = f.name
        if d["mode"] == "solo":
            solos[d["model"]] = d
        else:
            pairs.append(d)
    return solos, pairs


def fmt_pct(v):
    """Format percentage with sign and color hint."""
    return f"{v:+.2f}%"


def print_separator(char="=", width=120):
    print(char * width)


def print_section(title):
    print()
    print_separator("=")
    print(f"  {title}")
    print_separator("=")
    print()


def main():
    solos, pairs = load_all_results()

    # ──────────────────────────────────────────────
    # 1. Solo Baselines
    # ──────────────────────────────────────────────
    print_section("1. Solo Baseline 결과 (단독 실행, GPU 4장)")

    solo_header = f"{'Model':<20} {'Mean Iter (s)':>14} {'Std (s)':>10} {'Min (s)':>10} {'Max (s)':>10} {'Throughput (samples/s)':>22} {'GPU Mem Alloc (MB)':>18} {'Avg Loss':>10}"
    print(solo_header)
    print("-" * len(solo_header))

    solo_models = sorted(solos.keys())
    for m in solo_models:
        d = solos[m]
        it = d["iter_times_sec"]
        print(f"{m:<20} {it['mean']:>14.6f} {it['std']:>10.6f} {it['min']:>10.6f} {it['max']:>10.6f} {d['throughput_samples_per_sec']:>22.2f} {d['gpu_memory_allocated_mb']:>18.1f} {d['avg_loss']:>10.4f}")

    # ──────────────────────────────────────────────
    # 2. Pair 결과 & Slowdown 계산
    # ──────────────────────────────────────────────
    print_section("2. Pair 실행 결과 - 각 모델의 Solo 대비 Slowdown (%)")

    # Build a dict: (model, partner) -> data
    pair_map = {}
    for d in pairs:
        pair_map[(d["model"], d["partner"])] = d

    # For each unique unordered pair, show both models' slowdown
    seen_pairs = set()
    pair_rows = []

    for d in pairs:
        m, p = d["model"], d["partner"]
        key = tuple(sorted([m, p]))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        # Model A = key[0], Model B = key[1]
        a, b = key
        data_a = pair_map.get((a, b))
        data_b = pair_map.get((b, a))

        solo_a_mean = solos[a]["iter_times_sec"]["mean"]
        solo_b_mean = solos[b]["iter_times_sec"]["mean"]

        slowdown_a = ((data_a["iter_times_sec"]["mean"] - solo_a_mean) / solo_a_mean * 100) if data_a else None
        slowdown_b = ((data_b["iter_times_sec"]["mean"] - solo_b_mean) / solo_b_mean * 100) if data_b else None

        thru_a = data_a["throughput_samples_per_sec"] if data_a else None
        thru_b = data_b["throughput_samples_per_sec"] if data_b else None

        solo_thru_a = solos[a]["throughput_samples_per_sec"]
        solo_thru_b = solos[b]["throughput_samples_per_sec"]

        thru_drop_a = ((thru_a - solo_thru_a) / solo_thru_a * 100) if thru_a else None
        thru_drop_b = ((thru_b - solo_thru_b) / solo_thru_b * 100) if thru_b else None

        pair_rows.append({
            "a": a, "b": b,
            "slowdown_a": slowdown_a, "slowdown_b": slowdown_b,
            "thru_drop_a": thru_drop_a, "thru_drop_b": thru_drop_b,
            "mean_a": data_a["iter_times_sec"]["mean"] if data_a else None,
            "mean_b": data_b["iter_times_sec"]["mean"] if data_b else None,
            "thru_a": thru_a, "thru_b": thru_b,
            "avg_slowdown": ((slowdown_a or 0) + (slowdown_b or 0)) / 2,
        })

    # Sort by average slowdown descending
    pair_rows.sort(key=lambda r: r["avg_slowdown"], reverse=True)

    hdr = (f"{'Pair (A + B)':<40} "
           f"{'A Iter(s)':>10} {'A Slow%':>9} {'A Thru':>8} "
           f"{'B Iter(s)':>10} {'B Slow%':>9} {'B Thru':>8} "
           f"{'Avg Slow%':>10}")
    print(hdr)
    print("-" * len(hdr))

    for r in pair_rows:
        pair_label = f"{r['a']} + {r['b']}"
        a_iter = f"{r['mean_a']:.6f}" if r['mean_a'] else "N/A"
        b_iter = f"{r['mean_b']:.6f}" if r['mean_b'] else "N/A"
        a_slow = f"{r['slowdown_a']:+.2f}%" if r['slowdown_a'] is not None else "N/A"
        b_slow = f"{r['slowdown_b']:+.2f}%" if r['slowdown_b'] is not None else "N/A"
        a_thru = f"{r['thru_a']:.0f}" if r['thru_a'] else "N/A"
        b_thru = f"{r['thru_b']:.0f}" if r['thru_b'] else "N/A"
        avg_s = f"{r['avg_slowdown']:+.2f}%"
        print(f"{pair_label:<40} {a_iter:>10} {a_slow:>9} {a_thru:>8} {b_iter:>10} {b_slow:>9} {b_thru:>8} {avg_s:>10}")

    # ──────────────────────────────────────────────
    # 3. 모델별 평균 간섭 영향
    # ──────────────────────────────────────────────
    print_section("3. 모델별 평균 간섭 영향 (해당 모델이 pair로 실행될 때 평균 slowdown)")

    model_slowdowns = defaultdict(list)
    for r in pair_rows:
        if r["slowdown_a"] is not None:
            model_slowdowns[r["a"]].append(r["slowdown_a"])
        if r["slowdown_b"] is not None:
            model_slowdowns[r["b"]].append(r["slowdown_b"])

    hdr2 = f"{'Model':<20} {'Avg Slowdown%':>14} {'Min Slowdown%':>14} {'Max Slowdown%':>14} {'# Pairs':>8}"
    print(hdr2)
    print("-" * len(hdr2))

    model_avg = []
    for m in sorted(model_slowdowns.keys()):
        vals = model_slowdowns[m]
        avg_s = sum(vals) / len(vals)
        model_avg.append((m, avg_s, min(vals), max(vals), len(vals)))

    model_avg.sort(key=lambda x: x[1], reverse=True)
    for m, avg_s, mn, mx, cnt in model_avg:
        print(f"{m:<20} {avg_s:>+14.2f}% {mn:>+14.2f}% {mx:>+14.2f}% {cnt:>8}")

    # ──────────────────────────────────────────────
    # 4. 파트너별 간섭 유발 순위 (다른 모델에 미치는 영향)
    # ──────────────────────────────────────────────
    print_section("4. 파트너별 간섭 유발 순위 (이 모델이 파트너일 때 상대방의 평균 slowdown)")

    caused_slowdowns = defaultdict(list)
    for r in pair_rows:
        # A가 받는 slowdown -> B가 유발
        if r["slowdown_a"] is not None:
            caused_slowdowns[r["b"]].append(r["slowdown_a"])
        if r["slowdown_b"] is not None:
            caused_slowdowns[r["a"]].append(r["slowdown_b"])

    hdr3 = f"{'Partner Model':<20} {'Avg Caused Slowdown%':>22} {'Max Caused Slowdown%':>22} {'# Cases':>8}"
    print(hdr3)
    print("-" * len(hdr3))

    caused_avg = []
    for m in sorted(caused_slowdowns.keys()):
        vals = caused_slowdowns[m]
        caused_avg.append((m, sum(vals)/len(vals), max(vals), len(vals)))

    caused_avg.sort(key=lambda x: x[1], reverse=True)
    for m, avg_s, mx, cnt in caused_avg:
        print(f"{m:<20} {avg_s:>+22.2f}% {mx:>+22.2f}% {cnt:>8}")

    # ──────────────────────────────────────────────
    # 5. 최고/최저 간섭 Top 5
    # ──────────────────────────────────────────────
    print_section("5. 간섭이 가장 큰 조합 Top 10 (평균 slowdown 기준)")

    top10_worst = pair_rows[:10]
    hdr4 = f"{'Rank':<6} {'Pair':<40} {'Avg Slowdown%':>14} {'A Slowdown%':>12} {'B Slowdown%':>12}"
    print(hdr4)
    print("-" * len(hdr4))
    for i, r in enumerate(top10_worst, 1):
        pair_label = f"{r['a']} + {r['b']}"
        a_s = f"{r['slowdown_a']:+.2f}%" if r['slowdown_a'] is not None else "N/A"
        b_s = f"{r['slowdown_b']:+.2f}%" if r['slowdown_b'] is not None else "N/A"
        print(f"{i:<6} {pair_label:<40} {r['avg_slowdown']:>+14.2f}% {a_s:>12} {b_s:>12}")

    print_section("6. 간섭이 가장 작은 조합 Top 10 (평균 slowdown 기준)")

    top10_best = pair_rows[-10:][::-1]
    print(hdr4)
    print("-" * len(hdr4))
    for i, r in enumerate(top10_best, 1):
        pair_label = f"{r['a']} + {r['b']}"
        a_s = f"{r['slowdown_a']:+.2f}%" if r['slowdown_a'] is not None else "N/A"
        b_s = f"{r['slowdown_b']:+.2f}%" if r['slowdown_b'] is not None else "N/A"
        print(f"{i:<6} {pair_label:<40} {r['avg_slowdown']:>+14.2f}% {a_s:>12} {b_s:>12}")

    # ──────────────────────────────────────────────
    # 6. Slowdown Heatmap (text)
    # ──────────────────────────────────────────────
    print_section("7. Slowdown Heatmap (행=실행 모델, 열=파트너 모델, 값=실행 모델의 slowdown %)")

    all_models = sorted(solos.keys())
    # Build matrix
    matrix = {}
    for d in pairs:
        m, p = d["model"], d["partner"]
        solo_mean = solos[m]["iter_times_sec"]["mean"]
        slowdown = (d["iter_times_sec"]["mean"] - solo_mean) / solo_mean * 100
        matrix[(m, p)] = slowdown

    # Print header
    col_width = 12
    header = f"{'Model':<20}"
    for m2 in all_models:
        header += f"{m2:>{col_width}}"
    print(header)
    print("-" * len(header))

    for m1 in all_models:
        row = f"{m1:<20}"
        for m2 in all_models:
            if m1 == m2:
                row += f"{'---':>{col_width}}"
            elif (m1, m2) in matrix:
                row += f"{matrix[(m1,m2)]:>{col_width}.2f}%"
            else:
                row += f"{'N/A':>{col_width}}"
        print(row)

    # ──────────────────────────────────────────────
    # Summary stats
    # ──────────────────────────────────────────────
    print_section("8. 전체 요약 통계")

    all_slowdowns = [matrix[k] for k in matrix]
    print(f"  총 pair 실험 수:                 {len(pairs)}")
    print(f"  고유 pair 조합 수:               {len(seen_pairs)}")
    print(f"  전체 평균 slowdown:              {sum(all_slowdowns)/len(all_slowdowns):+.2f}%")
    print(f"  전체 최대 slowdown:              {max(all_slowdowns):+.2f}%")
    print(f"  전체 최소 slowdown:              {min(all_slowdowns):+.2f}%")
    print(f"  Slowdown > 20% 인 케이스 수:     {sum(1 for v in all_slowdowns if v > 20)}")
    print(f"  Slowdown > 10% 인 케이스 수:     {sum(1 for v in all_slowdowns if v > 10)}")
    print(f"  Slowdown <  5% 인 케이스 수:     {sum(1 for v in all_slowdowns if v < 5)}")
    print()


if __name__ == "__main__":
    main()
