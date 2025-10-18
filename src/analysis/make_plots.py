
import os, json, argparse
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results")
    args = parser.parse_args()
    hist_path = os.path.join(args.results, "divfree_fno_train_history.json")
    if not os.path.exists(hist_path):
        print("No history found to plot.")
        return
    with open(hist_path, "r", encoding="utf-8") as f:
        hist = json.load(f)["history"]
    epochs = [h["epoch"] for h in hist]
    l2s = [h["l2"] for h in hist]
    divs = [h["div"] for h in hist]
    plt.figure()
    plt.plot(epochs, l2s, label="L2")
    plt.plot(epochs, divs, label="Div")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    os.makedirs(os.path.join(args.results, "figures"), exist_ok=True)
    out = os.path.join(args.results, "figures", "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
