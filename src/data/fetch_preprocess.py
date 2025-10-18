import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np

CSV_URL = "https://raw.githubusercontent.com/pdebench/PDEBench/main/pdebench/data_download/pdebench_data_urls.csv"
DEFAULT_DATASET = "ns_incom"
DEFAULT_DOWN_SAMPLE = 8  # 512 -> 64 grid
DEFAULT_FRAME_STRIDE = 20
DEFAULT_TARGET_OFFSET = 4
DEFAULT_MAX_PAIRS = 2048


@dataclass(frozen=True)
class FileSpec:
    filename: str
    url: str
    md5: str
    relative_dir: Path


def log(msg: str) -> None:
    print(msg, file=sys.stdout)


def fetch_metadata(dataset: str, metadata_url: str = CSV_URL) -> list[FileSpec]:
    try:
        with urllib.request.urlopen(metadata_url) as response:
            text = response.read().decode("utf-8")
    except urllib.error.URLError as err:
        raise RuntimeError(f"Failed to download metadata CSV from {metadata_url}") from err

    reader = csv.DictReader(io.StringIO(text))
    entries: list[FileSpec] = []

    for row in reader:
        if row["PDE"].strip().lower() != dataset.lower():
            continue
        entries.append(
            FileSpec(
                filename=row["Filename"].strip(),
                url=row["URL"].strip(),
                md5=row["MD5"].strip(),
                relative_dir=Path(row["Path"].strip()),
            )
        )

    if not entries:
        raise RuntimeError(f"No entries for dataset '{dataset}' were found in metadata.")

    entries.sort(key=lambda spec: spec.filename)
    return entries


def select_specs(
    specs: Sequence[FileSpec],
    shards: Sequence[str] | None,
    max_files: int | None,
) -> list[FileSpec]:
    chosen: list[FileSpec]
    if shards:
        shards_lower = [sh.lower() for sh in shards]
        chosen = [
            spec
            for spec in specs
            if any(fragment in spec.filename.lower() for fragment in shards_lower)
        ]
    else:
        chosen = list(specs)

    if max_files is not None and max_files > 0:
        chosen = chosen[:max_files]

    if not chosen:
        raise RuntimeError("No files left to download after applying filters.")

    return chosen


def compute_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download_file(spec: FileSpec, raw_root: Path, skip_existing: bool) -> Path:
    target_path = raw_root / spec.relative_dir / spec.filename
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and skip_existing:
        log(f"Found existing file {target_path}, skipping download.")
        return target_path

    tmp_path = target_path.with_suffix(target_path.suffix + ".download")
    log(f"Downloading {spec.url} -> {target_path}")
    try:
        with urllib.request.urlopen(spec.url) as response, tmp_path.open("wb") as dst:
            shutil.copyfileobj(response, dst)
    except urllib.error.URLError as err:
        raise RuntimeError(f"Failed to download {spec.url}") from err

    if spec.md5:
        md5_value = compute_md5(tmp_path)
        if md5_value.lower() != spec.md5.lower():
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"MD5 mismatch for {spec.filename}: expected {spec.md5}, got {md5_value}"
            )

    tmp_path.replace(target_path)
    return target_path


def average_pool(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return arr
    n, h, w, c = arr.shape
    if h % factor != 0 or w % factor != 0:
        raise ValueError(f"Spatial dims {(h, w)} are not divisible by factor {factor}.")
    arr = arr.reshape(n, h // factor, factor, w // factor, factor, c)
    return arr.mean(axis=(2, 4))


def frame_pairs(
    frames: np.ndarray,
    target_offset: int,
    stride: int,
    max_pairs: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if target_offset <= 0:
        raise ValueError("target_offset must be > 0.")
    n_frames = frames.shape[0]
    last_start = n_frames - target_offset
    if last_start <= 0:
        return np.empty((0, *frames.shape[1:]), dtype=frames.dtype), np.empty(
            (0, *frames.shape[1:]), dtype=frames.dtype
        )

    indices = list(range(0, last_start, stride))
    if max_pairs is not None and max_pairs > 0:
        indices = indices[:max_pairs]

    if not indices:
        return np.empty((0, *frames.shape[1:]), dtype=frames.dtype), np.empty(
            (0, *frames.shape[1:]), dtype=frames.dtype
        )

    idx_array = np.array(indices, dtype=np.int32)
    src = frames[idx_array]
    dst = frames[idx_array + target_offset]
    return src, dst


def convert_file(
    h5_path: Path,
    processed_root: Path,
    downsample: int,
    frame_stride: int,
    target_offset: int,
    max_pairs: int | None,
) -> tuple[Path, int]:
    processed_root.mkdir(parents=True, exist_ok=True)
    out_path = processed_root / f"{h5_path.stem}_ds{downsample}_stride{frame_stride}_off{target_offset}.npz"
    if out_path.exists():
        log(f"Processed file already exists at {out_path}, skipping conversion.")
        with np.load(out_path) as data:
            return out_path, int(data["x"].shape[0])

    log(f"Converting {h5_path} -> {out_path}")
    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []

    with h5py.File(h5_path, "r") as handle:
        if "velocity" not in handle:
            raise RuntimeError(f"{h5_path} missing 'velocity' dataset.")
        velocity = handle["velocity"]
        n_batch, n_frames, height, width, vec_dim = velocity.shape
        if vec_dim != 2:
            raise RuntimeError(f"Expected velocity vector dim of 2, got {vec_dim}.")

        for batch_idx in range(n_batch):
            # Process batch member in manageable temporal chunks while keeping memory modest.
            downsampled_frames: list[np.ndarray] = []
            chunk = max(1, min(32, n_frames))
            for start in range(0, n_frames, chunk):
                stop = min(start + chunk, n_frames)
                block = velocity[batch_idx, start:stop]  # shape (chunk, H, W, 2)
                block = np.asarray(block, dtype=np.float32)
                block = average_pool(block, downsample)
                downsampled_frames.append(block)

            if not downsampled_frames:
                continue
            frames = np.concatenate(downsampled_frames, axis=0)
            xs, ys = frame_pairs(
                frames=frames,
                target_offset=target_offset,
                stride=frame_stride,
                max_pairs=max_pairs,
            )
            if xs.size == 0:
                continue
            x_chunks.append(xs)
            y_chunks.append(ys)

    if not x_chunks:
        raise RuntimeError(f"No training pairs could be generated from {h5_path}.")

    x = np.concatenate(x_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)
    np.savez_compressed(out_path, x=x.astype(np.float32), y=y.astype(np.float32))
    return out_path, int(x.shape[0])


def write_manifest(
    manifest_path: Path,
    dataset: str,
    specs: Iterable[FileSpec],
    processed_files: Iterable[Path],
    downsample: int,
    frame_stride: int,
    target_offset: int,
    total_pairs: int,
) -> None:
    manifest = {
        "dataset": dataset,
        "source_files": [spec.filename for spec in specs],
        "processed_files": [str(path.name) for path in processed_files],
        "downsample": downsample,
        "frame_stride": frame_stride,
        "target_offset": target_offset,
        "total_pairs": total_pairs,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and preprocess PDEBench datasets into training-ready pairs."
    )
    parser.add_argument("--root", type=str, default="./data_cache", help="Root output directory.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help="Dataset identifier from PDEBench metadata (default: ns_incom).",
    )
    parser.add_argument(
        "--metadata-url",
        type=str,
        default=CSV_URL,
        help="Override metadata CSV URL if needed.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=1,
        help="Maximum number of dataset shards to download.",
    )
    parser.add_argument(
        "--shards",
        nargs="+",
        help="Specific shard identifiers to download (substring match on filename).",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=DEFAULT_DOWN_SAMPLE,
        help="Spatial downsampling factor when converting HDF5 data.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=DEFAULT_FRAME_STRIDE,
        help="Stride between starting frames when forming supervision pairs.",
    )
    parser.add_argument(
        "--target-offset",
        type=int,
        default=DEFAULT_TARGET_OFFSET,
        help="Offset (in frames) between input and target snapshots.",
    )
    parser.add_argument(
        "--pairs-per-file",
        type=int,
        default=DEFAULT_MAX_PAIRS,
        help="Maximum number of (input,target) pairs to keep per shard.",
    )
    parser.add_argument(
        "--delete-raw",
        action="store_true",
        help="Remove raw HDF5 shards after successful conversion to save space.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading raw files and only run conversion on existing data.",
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip conversion step (useful when only downloading raw data).",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    raw_root = root / "raw" / args.dataset
    processed_root = root / "processed" / args.dataset

    specs = fetch_metadata(args.dataset, metadata_url=args.metadata_url)
    selected_specs = select_specs(specs, args.shards, args.max_files)

    processed_paths: list[Path] = []
    total_pairs = 0

    for spec in selected_specs:
        h5_path = raw_root / spec.relative_dir / spec.filename
        if not args.skip_download:
            h5_path = download_file(spec, raw_root, skip_existing=True)

        if args.skip_conversion:
            continue

        processed_path, pairs = convert_file(
            h5_path=h5_path,
            processed_root=processed_root,
            downsample=args.downsample,
            frame_stride=args.frame_stride,
            target_offset=args.target_offset,
            max_pairs=args.pairs_per_file,
        )
        processed_paths.append(processed_path)
        total_pairs += pairs

        if args.delete_raw and h5_path.exists():
            log(f"Removing raw file {h5_path}")
            h5_path.unlink()

    if not args.skip_conversion and processed_paths:
        manifest_path = processed_root / "manifest.json"
        write_manifest(
            manifest_path=manifest_path,
            dataset=args.dataset,
            specs=selected_specs,
            processed_files=processed_paths,
            downsample=args.downsample,
            frame_stride=args.frame_stride,
            target_offset=args.target_offset,
            total_pairs=total_pairs,
        )
        log(f"Stored manifest at {manifest_path}")
        log(f"Prepared {total_pairs} input-target pairs across {len(processed_paths)} files.")


if __name__ == "__main__":
    main()
