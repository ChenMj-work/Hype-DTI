import argparse
import subprocess
import sys
from pathlib import Path

from download_kge_checkpoint import FILENAME


DATASETS = [
    f"{target}_10shots_{fold}"
    for target in ("ago", "blocker", "e-")
    for fold in range(5)
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--filter-protected-pseudo-candidates", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    entrypoint = root / "kge" / "std_main.py"
    for index, dataset in enumerate(DATASETS, start=1):
        command = [
            sys.executable,
            str(entrypoint),
            "--dataset",
            dataset,
            "--device",
            str(args.device),
            "--seed",
            str(args.seed),
            "--load_kge_model",
            FILENAME,
        ]
        if args.filter_protected_pseudo_candidates:
            command.append("--filter-protected-pseudo-candidates")
        print(f"[run {index:02d}/{len(DATASETS)}] {dataset}", flush=True)
        subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
