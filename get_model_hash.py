import hashlib
import json
from pathlib import Path
from datetime import datetime

HF_CACHE = Path(r"C:\Users\ddm\.cache\huggingface\hub")

OUTPUT_JSON = "huggingface_model_hashes.json"


def sha256_file(path: Path, chunk_size=1024 * 1024):
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


results = []

model_dirs = [p for p in HF_CACHE.iterdir() if p.is_dir() and p.name.startswith("models--")]

for model_dir in model_dirs:
    print(f"\n[SCAN] {model_dir.name}")

    snapshots_dir = model_dir / "snapshots"

    if not snapshots_dir.exists():
        print("  -> snapshots 폴더 없음")
        continue

    for snapshot in snapshots_dir.iterdir():

        if not snapshot.is_dir():
            continue

        commit_hash = snapshot.name

        print(f"  [COMMIT] {commit_hash}")

        for file_path in snapshot.rglob("*"):

            if not file_path.is_file():
                continue

            try:
                relative_path = file_path.relative_to(snapshot)

                size = file_path.stat().st_size

                print(f"    hashing: {relative_path}")

                sha256 = sha256_file(file_path)

                results.append({
                    "repo": model_dir.name.replace("models--", "").replace("--", "/"),
                    "commit_hash": commit_hash,
                    "relative_path": str(relative_path).replace("\\", "/"),
                    "absolute_path": str(file_path),
                    "size_bytes": size,
                    "sha256": sha256,
                    "hashed_at": datetime.utcnow().isoformat() + "Z"
                })

            except Exception as e:
                print(f"    ERROR: {file_path}")
                print(f"    {e}")

output = {
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "huggingface_cache": str(HF_CACHE),
    "files": results
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n완료: {OUTPUT_JSON}")
print(f"총 파일 수: {len(results)}")