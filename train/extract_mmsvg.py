"""Extract FLUX-LoRA training data from MMSVG parquet shards.

Renders each sampled SVG at --resolution (native vector -> no upscale blur),
composited on a white background, and writes a manifest.jsonl compatible with
dataset.py (image paths relative to the manifest directory).

Usage (inside the flux-lora container):
    python extract_mmsvg.py \
        --raw_root /data/shp216/Flux-Lora-train-bundle/train_data/raw \
        --out_dir  /data/shp216/Flux-Lora-train-bundle/train_data \
        --n_icon 150000 --n_illu 150000 --resolution 1024 --workers 32
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cairosvg
import pyarrow.parquet as pq
from tqdm import tqdm


def process_shard(shard_path: str, img_dir: str, indices: list[int],
                  resolution: int) -> tuple[list[dict], int]:
    """Render selected rows of one parquet shard. Returns (manifest rows, fails)."""
    t = pq.read_table(shard_path, columns=["id", "svg", "description", "detail"])
    ids = t.column("id")
    svgs = t.column("svg")
    descs = t.column("description")
    details = t.column("detail")

    out = Path(img_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    fails = 0
    for i in indices:
        rid = ids[i].as_py()
        try:
            png = cairosvg.svg2png(
                bytestring=svgs[i].as_py().encode(),
                output_width=resolution, output_height=resolution,
                background_color="white",
            )
            (out / f"{rid}.png").write_bytes(png)
        except Exception:
            fails += 1
            continue
        rows.append({
            "id": rid,
            "image": str(out / f"{rid}.png"),
            "caption": (descs[i].as_py() or "").strip(),
            "detail": (details[i].as_py() or "").strip(),
        })
    return rows, fails


def plan_dataset(raw_root: Path, name: str, n: int, seed: int):
    """Allocate a per-shard sample plan proportional to shard row counts."""
    shards = sorted((raw_root / name / "data").glob("*.parquet"))
    counts = [pq.ParquetFile(s).metadata.num_rows for s in shards]
    total = sum(counts)
    n = min(n, total)
    rng = random.Random(seed)
    plan = []
    remaining = n
    for k, (shard, cnt) in enumerate(zip(shards, counts)):
        # proportional allocation; last shard takes the remainder
        take = round(n * cnt / total) if k < len(shards) - 1 else remaining
        take = max(0, min(take, cnt, remaining))
        remaining -= take
        if take:
            plan.append((str(shard), rng.sample(range(cnt), take)))
    print(f"[{name}] pool {total:,} -> sampling {n:,} across {len(plan)} shards")
    return plan


def plan_from_manifest(manifest_path: Path, raw_root: Path):
    """Exact reproduction: read dataset_manifest(_compact).json[.gz] and return
    the same (src, shard_path, indices) plan that produced it."""
    opener = gzip.open if str(manifest_path).endswith(".gz") else open
    with opener(manifest_path, "rt", encoding="utf-8") as f:
        d = json.load(f)
    repo = {"icon": "MMSVG-Icon", "illustration": "MMSVG-Illustration", "illu": "MMSVG-Illustration"}
    by_shard: dict[tuple[str, str], list[int]] = {}
    for it in d["items"]:
        src = "icon" if it["src"] == "icon" else "illu"
        by_shard.setdefault((src, it["source_parquet"]), []).append(int(it["source_row"]))
    plan = []
    for (src, shard), rows in sorted(by_shard.items()):
        path = raw_root / repo[src] / "data" / shard
        if not path.exists():
            raise FileNotFoundError(f"shard listed in manifest not found: {path}")
        plan.append((src, str(path), sorted(rows)))
    n = sum(len(r) for _, _, r in plan)
    print(f"[from_manifest] {n:,} items across {len(plan)} shards from {manifest_path.name}")
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_icon", type=int, default=150_000)
    ap.add_argument("--n_illu", type=int, default=150_000)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--trigger", default="<vector>",
                    help="prepended to every caption; '' to disable")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--from_manifest", type=Path, default=None,
                    help="dataset_manifest(_compact).json[.gz]: render exactly the "
                         "listed (source_parquet, source_row) items instead of sampling")
    args = ap.parse_args()

    jobs = []  # (src, shard_path, img_dir, indices)
    if args.from_manifest:
        plan = plan_from_manifest(args.from_manifest, args.raw_root)
    else:
        plan = []
        for src, repo, n in [("icon", "MMSVG-Icon", args.n_icon),
                             ("illu", "MMSVG-Illustration", args.n_illu)]:
            plan += [(src, shard, idx) for shard, idx in
                     plan_dataset(args.raw_root, repo, n, args.seed)]
    for src, shard, idx in plan:
        shard_tag = Path(shard).stem.split("-")[1]  # e.g. 00042
        img_dir = args.out_dir / "images" / src / shard_tag
        jobs.append((src, shard, str(img_dir), idx))

    trigger = args.trigger.strip()
    all_rows: list[dict] = []
    total_fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_shard, sp, d, idx, args.resolution): (src, sp)
                for src, sp, d, idx in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="shards"):
            src, sp = futs[fut]
            rows, fails = fut.result()
            total_fail += fails
            for r in rows:
                r["src"] = src
                if trigger:
                    r["caption"] = f"{trigger} {r['caption']}"
            all_rows.extend(rows)

    random.Random(args.seed).shuffle(all_rows)
    manifest = args.out_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for r in all_rows:
            # store image path relative to the manifest dir (dataset.py resolves it)
            r["image"] = str(Path(r["image"]).relative_to(args.out_dir))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"[done] {len(all_rows):,} rows -> {manifest} | render fails: {total_fail}")
    print("       by src:", dict(Counter(r["src"] for r in all_rows)))


if __name__ == "__main__":
    main()
