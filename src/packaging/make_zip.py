
import os, argparse, zipfile, pathlib

EXCLUDES = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in EXCLUDES]
        for file in files:
            if file.endswith(".zip"):
                continue
            p = os.path.join(root, file)
            arc = os.path.relpath(p, path)
            ziph.write(p, arc)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="final_solution.zip")
    args = parser.parse_args()

    out = args.out
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zipdir(".", zf)
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
