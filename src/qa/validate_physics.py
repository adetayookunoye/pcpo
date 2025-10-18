
import os, json, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results")
    args = parser.parse_args()
    metric_path = os.path.join(args.results, "divfree_fno_eval_metrics.json")
    if os.path.exists(metric_path):
        with open(metric_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        ok = True
        if m["div"] > 0.02:
            print("WARNING: Divergence too high for target.")
            ok = False
        if m["energy_err"] > 0.01:
            print("WARNING: Energy error above 1%.")
            ok = False
        print("Physics validation:", "PASS" if ok else "CHECK")
    else:
        print("No eval metrics found; run eval first.")

if __name__ == "__main__":
    main()
