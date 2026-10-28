"""Collect OWL RGB + segmentation locally, then upload to Drive in batches.

Writes to repo temp/. A separate --upload-only process packs UPLOAD_EVERY
frames into one zip, uploads to MyDrive/PhysicalAI_adv/data/OWL, then
removes the local files. Collect can be killed and resumed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from tqdm import tqdm

from envs.test_env import ARCHES, BALLS, TestEnv
from obj_gen.fetch_ground_textures import drive_service, ensure_path
from utils.lights import LIGHTING_PRESETS

N_ENVS = 40
N_SCENES = 20
N_VIEWS = 5
START_SEED = 0
RES = (1920, 1080)
UPLOAD_EVERY = 500
ALL_OBJ = BALLS + ARCHES
ROOT = Path(__file__).resolve().parent
TEMP_DIR = ROOT / "temp"
PROGRESS_PATH = TEMP_DIR / "progress.jsonl"
UPLOAD_PID = TEMP_DIR / "upload.pid"
REPACK_DIR = TEMP_DIR / "repack"
DRIVE_DATA = ("PhysicalAI_adv", "data", "OWL")
DRIVE_TEXTURE = Path("PhysicalAI_adv") / "assets" / "texture"
STEM_RE = re.compile(r"e(\d+)_s(\d+)_v(\d+)_")


def parse_res(value: str) -> tuple[int, int]:
    width, height = value.lower().split("x", 1)
    return int(width), int(height)


def find_mydrive() -> Path | None:
    colab = Path("/content/drive/MyDrive")
    if colab.is_dir():
        return colab
    cloud = Path.home() / "Library" / "CloudStorage"
    if cloud.is_dir():
        for drive in sorted(cloud.glob("GoogleDrive*")):
            for name in ("My Drive", "マイドライブ"):
                candidate = drive / name
                if candidate.is_dir():
                    return candidate
    return None


def default_texture_dir() -> Path:
    mydrive = find_mydrive()
    if mydrive is not None:
        drive_tex = mydrive / DRIVE_TEXTURE
        if drive_tex.is_dir():
            return drive_tex
    cache = ROOT / ".cache" / "polyhaven_textures"
    if cache.is_dir():
        return cache
    return ROOT / "assets" / "texture"


def list_textures(root: Path) -> list[Path]:
    textures = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        if all((path / name).exists() for name in ("diffuse.jpg", "roughness.jpg", "normal.jpg")):
            textures.append(path)
    if not textures:
        raise FileNotFoundError(f"no complete textures under {root}")
    return textures


def as_image(arr):
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    image = np.ascontiguousarray(arr)
    if image.ndim == 4:
        image = image[0]
    return image


def sample_view(rng: random.Random):
    lookat = (
        rng.uniform(3.0, 8.0),
        rng.uniform(-2.0, 2.0),
        rng.uniform(0.1, 0.5),
    )
    radius = rng.uniform(2.5, 6.0)
    yaw = rng.uniform(0.0, math.tau)
    pitch = rng.uniform(math.radians(12.0), math.radians(50.0))
    pos = (
        lookat[0] + radius * math.cos(pitch) * math.cos(yaw),
        lookat[1] + radius * math.cos(pitch) * math.sin(yaw),
        max(0.35, lookat[2] + radius * math.sin(pitch)),
    )
    return pos, lookat


def listed_objects(env):
    objects = []
    for entity in env.scene.entities:
        spec = next((obj for obj in ALL_OBJ if entity.name.startswith(obj["name"])), None)
        if spec is None:
            continue
        objects.append({"idx": entity.idx, "name": entity.name, "query": spec["query"]})
    return objects


def save_frame(stem: Path, rgb, seg) -> list[Path]:
    written = []
    rgb = as_image(rgb)
    rgb_path = Path(f"{stem}.png")
    iio.imwrite(rgb_path, rgb)
    written.append(rgb_path)
    if seg is None:
        return written
    seg = as_image(seg)
    npy_path = Path(f"{stem}_seg.npy")
    np.save(npy_path, seg)
    written.append(npy_path)
    return written


def parse_key(name: str) -> tuple[int, int, int] | None:
    match = STEM_RE.match(name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def keys_from_names(names) -> set[tuple[int, int, int]]:
    keys = set()
    for name in names:
        key = parse_key(Path(name).name)
        if key is not None:
            keys.add(key)
    return keys


def keys_from_dir(root: Path) -> set[tuple[int, int, int]]:
    if not root.is_dir():
        return set()
    return keys_from_names(path.name for path in root.iterdir() if path.is_file())


def keys_from_progress(path: Path) -> set[tuple[int, int, int]]:
    keys = set()
    if not path.is_file():
        return keys
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        keys.add((int(row["env"]), int(row["scene"]), int(row["view"])))
    return keys


def list_drive_files(service, folder_id: str) -> list[dict]:
    files = []
    token = None
    while True:
        listed = (
            service.files()
            .list(
                q=f"'{folder_id}' in parents and trashed = false",
                spaces="drive",
                fields="nextPageToken, files(id, name)",
                pageSize=1000,
                pageToken=token,
            )
            .execute()
        )
        files.extend(listed.get("files") or [])
        token = listed.get("nextPageToken")
        if not token:
            return files


def keys_from_drive(service, folder_id: str) -> set[tuple[int, int, int]]:
    return keys_from_names(item["name"] for item in list_drive_files(service, folder_id))


def drive_call(fn, retries: int = 6):
    for attempt in range(retries):
        try:
            return fn()
        except HttpError as exc:
            if exc.resp.status not in {403, 429} or attempt == retries - 1:
                raise
            wait = 30 * (attempt + 1)
            tqdm.write(f"drive quota, sleep {wait}s")
            time.sleep(wait)
    raise RuntimeError("drive call failed")


def download_drive_file(service, file_id: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=file_id)

    def _download():
        with dest.open("wb") as handle:
            downloader = MediaIoBaseDownload(handle, request, chunksize=8 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()

    drive_call(_download)


def delete_drive_ids(service, file_ids: list[str]) -> None:
    def callback(_request_id, _response, exception):
        if exception is not None:
            tqdm.write(f"delete fail {exception}")

    for start in range(0, len(file_ids), 20):
        chunk = file_ids[start : start + 20]
        batch = service.new_batch_http_request(callback=callback)
        for file_id in chunk:
            batch.add(service.files().delete(fileId=file_id))
        drive_call(batch.execute)
        time.sleep(1)


def stem_from_loose_name(name: str) -> str | None:
    if name.endswith(".zip") or parse_key(name) is None:
        return None
    if name.endswith("_seg.npy") or name.endswith("_seg.png"):
        return name[:-8]
    if name.endswith(".json") or name.endswith(".png"):
        return Path(name).stem
    return None


def group_loose_stems(files: list[dict]) -> dict[str, dict[str, dict]]:
    grouped: dict[str, dict[str, dict]] = {}
    for item in files:
        stem = stem_from_loose_name(item["name"])
        if stem is None:
            continue
        grouped.setdefault(stem, {})[item["name"]] = item
    return grouped


def complete_loose_stems(grouped: dict[str, dict[str, dict]]) -> list[str]:
    complete = []
    for stem in sorted(grouped):
        names = grouped[stem]
        if f"{stem}.json" in names and f"{stem}.png" in names and f"{stem}_seg.npy" in names:
            complete.append(stem)
    return complete


def repack_drive_main(args) -> None:
    service = drive_service()
    folder_id = ensure_path(service, DRIVE_DATA)
    files = list_drive_files(service, folder_id)
    zips = [item for item in files if item["name"].endswith(".zip")]
    grouped = group_loose_stems(files)
    stems = complete_loose_stems(grouped)
    tqdm.write(f"drive files={len(files)} zips={len(zips)} loose_frames={len(stems)}")
    if not stems:
        tqdm.write("no loose frames to pack")
        return
    REPACK_DIR.mkdir(parents=True, exist_ok=True)
    needed = {f"{stem}.json" for stem in stems}
    needed |= {f"{stem}.png" for stem in stems}
    needed |= {f"{stem}_seg.npy" for stem in stems}
    to_download = [item for item in files if item["name"] in needed]
    for item in tqdm(to_download, desc="download", unit="file"):
        dest = REPACK_DIR / item["name"]
        if dest.exists() and dest.stat().st_size > 0:
            continue
        download_drive_file(service, item["id"], dest)
        time.sleep(0.15)
    local = [REPACK_DIR / stem for stem in stems]
    every = args.upload_every
    for start in range(0, len(local), every):
        batch = local[start : start + every]
        flush_stems(service, folder_id, batch)
        ids = []
        for stem in batch:
            for item in grouped[stem.name].values():
                ids.append(item["id"])
        tqdm.write(f"delete {len(ids)} loose files")
        delete_drive_ids(service, ids)
    tqdm.write("repack done")


def complete_scenes(keys: set[tuple[int, int, int]], n_views: int) -> set[tuple[int, int]]:
    views: dict[tuple[int, int], set[int]] = {}
    for env_i, scene_i, view_i in keys:
        views.setdefault((env_i, scene_i), set()).add(view_i)
    needed = set(range(n_views))
    return {pair for pair, have in views.items() if needed <= have}


def record_progress(env_i: int, scene_i: int, view_i: int, stem: Path) -> None:
    with PROGRESS_PATH.open("a") as handle:
        handle.write(
            json.dumps(
                {"env": env_i, "scene": scene_i, "view": view_i, "stem": stem.name}
            )
            + "\n"
        )


def load_done_scenes(n_views: int, folder_id: str | None) -> set[tuple[int, int]]:
    keys = keys_from_dir(TEMP_DIR) | keys_from_progress(PROGRESS_PATH)
    if folder_id is not None:
        try:
            keys |= keys_from_drive(drive_service(), folder_id)
        except Exception:
            traceback.print_exc()
            tqdm.write("resume: drive listing failed, using local files only")
    done = complete_scenes(keys, n_views)
    tqdm.write(f"resume: {len(keys)} frames, {len(done)} complete scenes")
    return done


def stem_files(stem: Path) -> list[Path]:
    return [
        path
        for path in stem.parent.glob(stem.name + "*")
        if path.is_file() and path.suffix != ".zip"
    ]


def zip_paths(stem: Path) -> list[Path]:
    return [path for path in stem_files(stem) if not path.name.endswith("_seg.png")]


def pack_stems(stems: list[Path]) -> Path:
    zip_path = stems[0].parent / f"{stems[0].name}_n{len(stems)}.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for stem in tqdm(stems, desc="zip", unit="frame", leave=False):
            for path in zip_paths(stem):
                compress = zipfile.ZIP_STORED if path.suffix == ".png" else zipfile.ZIP_DEFLATED
                archive.write(path, arcname=path.name, compress_type=compress)
    return zip_path


def upload_zip(service, folder_id: str, zip_path: Path) -> None:
    media = MediaFileUpload(str(zip_path), mimetype="application/zip", resumable=True)
    request = service.files().create(
        body={"name": zip_path.name, "parents": [folder_id]},
        media_body=media,
        fields="id",
    )
    response = None
    done = 0
    with tqdm(total=100, desc=f"upload {zip_path.name}", unit="%", leave=True) as bar:
        while response is None:
            status, response = request.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                bar.update(max(0, pct - done))
                done = pct
        if done < 100:
            bar.update(100 - done)


def flush_stems(service, folder_id: str, stems: list[Path]) -> None:
    if not stems:
        return
    zip_path = pack_stems(stems)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    tqdm.write(f"zip ready {zip_path.name}  frames={len(stems)}  {size_mb:.1f}MB")
    try:
        upload_zip(service, folder_id, zip_path)
    except Exception:
        zip_path.unlink(missing_ok=True)
        raise
    zip_path.unlink(missing_ok=True)
    for stem in stems:
        for path in stem_files(stem):
            path.unlink(missing_ok=True)


def complete_stems(root: Path) -> list[Path]:
    stems = []
    for meta in sorted(root.glob("e*.json")):
        stem = meta.with_suffix("")
        if stem.with_suffix(".png").exists() and Path(f"{stem}_seg.npy").exists():
            stems.append(stem)
    return stems


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def spawn_uploader(upload_every: int) -> subprocess.Popen | None:
    if UPLOAD_PID.is_file():
        try:
            old = int(UPLOAD_PID.read_text().strip())
        except ValueError:
            old = None
        if old and pid_alive(old):
            tqdm.write(f"upload already running pid={old}")
            return None
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--upload-only", "--upload-every", str(upload_every)],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    UPLOAD_PID.write_text(str(proc.pid))
    tqdm.write(f"upload subprocess pid={proc.pid}")
    return proc


def upload_main(args) -> None:
    folder_id = ensure_path(drive_service(), DRIVE_DATA)
    service = drive_service()
    tqdm.write(f"upload-only temp={TEMP_DIR} every={args.upload_every}")
    pending = tqdm(total=args.upload_every, desc="pending", unit="frame")
    idle = 0
    last_n = -1
    try:
        while True:
            stems = complete_stems(TEMP_DIR)
            n = len(stems)
            pending.n = min(n, args.upload_every)
            pending.refresh()
            while len(stems) >= args.upload_every:
                flush_stems(service, folder_id, stems[: args.upload_every])
                stems = complete_stems(TEMP_DIR)
                n = len(stems)
                pending.n = min(n, args.upload_every)
                pending.refresh()
                idle = 0
            if n == 0:
                idle = 0
            elif n == last_n:
                idle += 1
                if idle >= 5:
                    tqdm.write(f"no new frames, flushing remainder {n}")
                    flush_stems(service, folder_id, stems)
                    break
            else:
                idle = 0
            last_n = n
            time.sleep(2)
    except KeyboardInterrupt:
        tqdm.write("upload stopping, flushing remainder")
        flush_stems(service, folder_id, complete_stems(TEMP_DIR))
    finally:
        pending.close()
        if UPLOAD_PID.is_file():
            try:
                if int(UPLOAD_PID.read_text().strip()) == os.getpid():
                    UPLOAD_PID.unlink(missing_ok=True)
            except ValueError:
                pass


def collect_main(args) -> None:
    import genesis as gs

    texture_dir = args.texture_dir or default_texture_dir()
    textures = list_textures(texture_dir)
    folder_id = ensure_path(drive_service(), DRIVE_DATA)
    done = set() if args.no_resume else load_done_scenes(args.views, folder_id)
    if not args.collect_only:
        spawn_uploader(args.upload_every)

    total = args.envs * args.scenes * args.views
    remaining_scenes = args.envs * args.scenes - len(done)
    tqdm.write(f"{len(textures)} textures, {len(LIGHTING_PRESETS)} lights")
    tqdm.write(f"temp={TEMP_DIR}")
    tqdm.write(f"drive={'/'.join(DRIVE_DATA)}  zip every {args.upload_every} frames")
    tqdm.write(f"collect {args.envs}x{args.scenes}x{args.views}={total}  remaining_scenes={remaining_scenes}")

    scene_bar = tqdm(total=remaining_scenes, desc="collect", unit="scene")
    for env_i in range(args.envs):
        env_rng = random.Random(args.seed + env_i)
        texture = env_rng.choice(textures)
        lighting = env_rng.choice(LIGHTING_PRESETS)
        for scene_i in range(args.scenes):
            if (env_i, scene_i) in done:
                continue
            seed = args.seed + env_i * args.scenes + scene_i
            rng = random.Random(seed)
            scene_bar.set_postfix(env=env_i, scene=scene_i, light=lighting["name"])
            gs.init(backend=gs.gpu)
            try:
                env = TestEnv(
                    seed=seed,
                    res=args.res,
                    texture_dir=texture,
                    lighting=lighting,
                    follow_robot=False,
                )
                objects = listed_objects(env)
                for view_i in range(args.views):
                    pos, lookat = sample_view(rng)
                    env.set_camera(pos, lookat)
                    rgb, _, seg, _ = env.render(depth=False, normal=False)
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    stem = TEMP_DIR / f"e{env_i:02d}_s{scene_i:04d}_v{view_i:02d}_{stamp}"
                    save_frame(stem, rgb, seg)
                    (stem.with_name(stem.name + ".json")).write_text(
                        json.dumps(
                            {
                                "env": env_i,
                                "scene": scene_i,
                                "view": view_i,
                                "seed": seed,
                                "texture": texture.name,
                                "light": lighting["name"],
                                "camera": {"pos": list(pos), "lookat": list(lookat)},
                                "res": list(args.res),
                                "objects": objects,
                            },
                            indent=2,
                        )
                    )
                    record_progress(env_i, scene_i, view_i, stem)
            finally:
                gs.destroy()
            done.add((env_i, scene_i))
            scene_bar.update(1)
    scene_bar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=N_ENVS)
    parser.add_argument("--scenes", type=int, default=N_SCENES)
    parser.add_argument("--views", type=int, default=N_VIEWS)
    parser.add_argument("--seed", type=int, default=START_SEED)
    parser.add_argument("--res", type=parse_res, default=RES)
    parser.add_argument("--upload-every", type=int, default=UPLOAD_EVERY)
    parser.add_argument("--texture-dir", type=Path, default=None)
    parser.add_argument("--upload-only", action="store_true")
    parser.add_argument("--repack-drive", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    if args.upload_only:
        upload_main(args)
        return
    if args.repack_drive:
        repack_drive_main(args)
        return
    collect_main(args)


if __name__ == "__main__":
    main()
