# compare_results.py
#
# Parses the results.json produced by the official repo's metrics.py for two
# runs (baseline vs. densified-init) and prints a side-by-side convergence
# table -- this is what becomes your thesis results table/plot.
#
# Usage:
#   python compare_results.py --baseline output/scene_baseline/results.json \
#                              --modified output/scene_densified/results.json
#
# Note: the official repo's metrics.py keys results.json by checkpoint name,
# typically "ours_<iteration>" (e.g. "ours_7000"). Run once with just
# --inspect to print the raw keys and confirm the naming before trusting the
# table, since exact key formatting can vary slightly by repo version.

import argparse
import json


def load_results(path):
    with open(path) as f:
        return json.load(f)


def iteration_from_key(key):
    # keys look like "ours_7000" -> 7000
    digits = "".join(c for c in key if c.isdigit())
    return int(digits) if digits else -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="results.json for the baseline (unmodified init) run")
    parser.add_argument("--modified", required=True, help="results.json for the densified-init run")
    parser.add_argument("--inspect", action="store_true", help="just print raw keys/values and exit")
    args = parser.parse_args()

    baseline = load_results(args.baseline)
    modified = load_results(args.modified)

    if args.inspect:
        print("Baseline keys:", list(baseline.keys()))
        print("Baseline sample:", next(iter(baseline.values())))
        print("Modified keys:", list(modified.keys()))
        raise SystemExit

    common_keys = sorted(
        (k for k in baseline.keys() if k in modified),
        key=iteration_from_key,
    )

    if not common_keys:
        print("No matching checkpoint keys between the two results.json files.")
        print("Run with --inspect on each file to check the key names.")
        raise SystemExit(1)

    header = f"{'Iter':>8} | {'PSNR base':>10} {'PSNR mod':>10} {'Δ':>6} | " \
             f"{'SSIM base':>10} {'SSIM mod':>10} {'Δ':>6} | " \
             f"{'LPIPS base':>11} {'LPIPS mod':>10} {'Δ':>6}"
    print(header)
    print("-" * len(header))

    for key in common_keys:
        it = iteration_from_key(key)
        b, m = baseline[key], modified[key]

        def metric(d, name):
            # metrics.py sometimes stores these under different case; try a
            # couple of common variants defensively.
            for cand in (name, name.upper(), name.lower(), name.capitalize()):
                if cand in d:
                    return d[cand]
            raise KeyError(f"Could not find metric '{name}' in {list(d.keys())}")

        psnr_b, psnr_m = metric(b, "PSNR"), metric(m, "PSNR")
        ssim_b, ssim_m = metric(b, "SSIM"), metric(m, "SSIM")
        lpips_b, lpips_m = metric(b, "LPIPS"), metric(m, "LPIPS")

        print(f"{it:8d} | {psnr_b:10.3f} {psnr_m:10.3f} {psnr_m - psnr_b:+6.3f} | "
              f"{ssim_b:10.4f} {ssim_m:10.4f} {ssim_m - ssim_b:+6.4f} | "
              f"{lpips_b:11.4f} {lpips_m:10.4f} {lpips_m - lpips_b:+6.4f}")

    print("\nPositive Δ PSNR/SSIM = modified is better. Negative Δ LPIPS = modified is better.")
    print("The gap at EARLY iterations (e.g. 1000-3000) is your convergence-speed story;")
    print("the gap at the FINAL iteration is your final-quality story.")
