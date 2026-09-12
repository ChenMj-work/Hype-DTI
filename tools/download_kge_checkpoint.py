import argparse
import hashlib
from pathlib import Path


FILE_ID = "1_RCRrHJBosWycpqzxXxzrmJYCGWaJfc8"
FILENAME = (
    "2024-04-28_10_01_40.24__kgeSLHstd_main.py--save--dataset__a-10--"
    "device__6--gate__kge.pth"
)
SHA256 = "A2FC0281D2399A0B67D9554CF89AB3AE09BEE4059EE2A6FF308C37F375DC9F12"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="var_models")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / FILENAME
    if output_path.exists() and not args.force:
        if file_sha256(output_path) == SHA256:
            print(f"[download] verified existing checkpoint: {output_path}")
            return
        raise RuntimeError(f"Existing checkpoint has the wrong checksum: {output_path}")

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown is required to download the KG checkpoint; install the "
            "provided environment.yml."
        ) from exc
    result = gdown.download(id=FILE_ID, output=str(output_path), quiet=False)
    if result is None or not output_path.exists():
        raise RuntimeError("KG checkpoint download failed.")
    actual = file_sha256(output_path)
    if actual != SHA256:
        output_path.unlink()
        raise RuntimeError(f"KG checkpoint checksum mismatch: {actual}")
    print(f"[download] checkpoint verified: {output_path}")


if __name__ == "__main__":
    main()
