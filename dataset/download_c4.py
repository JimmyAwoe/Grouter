import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import hf_hub_download

repo_id = "allenai/c4"
repo_type = "dataset"
local_dir = "dataset/c4_data"


parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_files",
    type=int,
    help="The number of files to download, range 0~1023",
    default=256,
)
parser.add_argument(
    "--start_index",
    type=int,
    help="Start shard index, range 0~1023",
    default=0,
)
parser.add_argument(
    "--max_workers",
    type=int,
    help="Parallel download workers (defaults to HF_MAX_WORKERS or 16)",
    default=int(os.getenv("HF_MAX_WORKERS", "16")),
)
parser.add_argument(
    "--retries",
    type=int,
    help="Retry times per file when download fails",
    default=3,
)
args = parser.parse_args()
n = args.num_files
start = args.start_index
max_workers = max(1, args.max_workers)
retries = max(0, args.retries)

if start < 0 or start > 1023:
    raise ValueError("--start_index must be in range [0, 1023]")
if n < 0 or start + n > 1024:
    raise ValueError("--num_files must satisfy start_index + num_files <= 1024")

print(f"[download] repo_id={repo_id}")
print(f"[download] local_dir={local_dir}")
print(f"[download] start_index={start}, num_files={n}")
print(f"[download] max_workers={max_workers}, retries={retries}")
print(f"[download] HF_ENDPOINT={os.getenv('HF_ENDPOINT')}")
print(f"[download] HF_HUB_ENABLE_HF_TRANSFER={os.getenv('HF_HUB_ENABLE_HF_TRANSFER')}")


def download_one(i: int) -> str:
    filename = f"en/c4-train.{i:05d}-of-01024.json.gz"
    last_err = None
    for attempt in range(retries + 1):
        try:
            hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                force_download=False,
                resume_download=True,
            )
            return filename
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                print(f"[retry {attempt + 1}/{retries}] {filename}: {exc}")
    raise RuntimeError(f"Failed to download {filename}") from last_err


indices = list(range(start, start + n))
ok = 0
failed = []
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    future_to_idx = {executor.submit(download_one, i): i for i in indices}
    for future in as_completed(future_to_idx):
        idx = future_to_idx[future]
        try:
            filename = future.result()
            ok += 1
            print(f"[ok {ok}/{n}] {filename}")
        except Exception as exc:
            failed.append(idx)
            print(f"[failed] en/c4-train.{idx:05d}-of-01024.json.gz: {exc}")

if failed:
    print(f"Done with failures: success={ok}, failed={len(failed)}, failed_indices={failed}")
else:
    print(f"Done, downloaded {ok} files.")


