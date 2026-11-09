"""Prepare offline ALFWorld placeholder parquet files in a fresh directory."""

import argparse
import hashlib
import json
from pathlib import Path


def prepare(output_dir, train_size=8, val_size=32):
    # These placeholders carry modality and sample count. The environment
    # supplies live observations; no external dataset or network is needed.
    import pyarrow as pa
    import pyarrow.parquet as pq

    if train_size < 1 or val_size < 1:
        raise ValueError("sample counts must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    metadata = {}
    for split, size in (("train", train_size), ("test", val_size)):
        rows = [
            {"data_source": "text", "prompt": [{"role": "user", "content": ""}], "ability": "agent", "extra_info": {"split": split, "index": i}}
            for i in range(size)
        ]
        path = output / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        metadata[split] = {"rows": size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-size", default=8, type=int)
    parser.add_argument("--val-size", default=32, type=int)
    args = parser.parse_args()
    prepare(args.output_dir, args.train_size, args.val_size)


if __name__ == "__main__":
    main()
