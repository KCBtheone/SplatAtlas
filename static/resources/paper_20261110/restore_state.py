"""Restore the listed model and camera assets for one SplatAtlas state."""
import argparse
import json
from pathlib import Path, PurePosixPath
import urllib.request

def restore(asset, output):
    relative = PurePosixPath(asset["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe asset path: {relative}")
    target = output.joinpath(*relative.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    request = urllib.request.Request(asset["url"])
    size = asset["bytes"]
    if "offset" in asset:
        first = asset["offset"]
        request.add_header("Range", f"bytes={first}-{first + size - 1}")
    temporary = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(request, timeout=60) as response:
        if "offset" in asset and (response.status != 206 or not response.headers.get(
                "Content-Range", "").startswith(f"bytes {first}-{first + size - 1}/")):
            raise RuntimeError("The archive server did not return the indexed byte range")
        with temporary.open("xb") as file:
            remaining = size
            while remaining:
                chunk = response.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise OSError(f"Truncated response: {relative}")
                file.write(chunk)
                remaining -= len(chunk)
    temporary.rename(target)
    print(target)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state_id")
    parser.add_argument("output", type=Path)
    parser.add_argument("--index", type=Path, default=Path(__file__).with_name("states.jsonl"))
    args = parser.parse_args()
    with args.index.open(encoding="utf-8") as source:
        state = next((row for row in map(json.loads, source) if row["state_uid"] == args.state_id), None)
    if state is None:
        parser.error("State identifier is absent from the index")
    if state["public_native_asset_status"] != "listed":
        parser.error("The selected state has no complete indexed public native asset group")
    for asset in state["asset_links"]:
        if asset["kind"] in {"native_model", "auxiliary_weight", "camera", "saved_config"}:
            restore(asset, args.output)

if __name__ == "__main__":
    main()
