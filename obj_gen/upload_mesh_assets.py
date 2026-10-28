"""Upload ball and arch glbs to MyDrive/PhysicalAI_adv/assets/{balls,arches}."""

from __future__ import annotations

import sys
from pathlib import Path

from googleapiclient.http import MediaFileUpload

from obj_gen.fetch_ground_textures import drive_service, ensure_folder, ensure_path, find_child

ROOT = Path(__file__).resolve().parents[1]
DRIVE_ASSETS = ("PhysicalAI_adv", "assets")
FOLDERS = ("balls", "arches")


def upload_folder(service, assets_id: str, name: str) -> None:
    local = ROOT / "assets" / name
    folder_id = ensure_folder(service, assets_id, name)
    for path in sorted(local.glob("*.glb")):
        if find_child(service, folder_id, path.name, folder=False):
            print(f"  skip  {name}/{path.name}")
            continue
        print(f"  put   {name}/{path.name}")
        media = MediaFileUpload(str(path), mimetype="model/gltf-binary", resumable=False)
        service.files().create(
            body={"name": path.name, "parents": [folder_id]},
            media_body=media,
            fields="id",
        ).execute()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    service = drive_service()
    assets_id = ensure_path(service, DRIVE_ASSETS)
    for name in FOLDERS:
        print(name)
        upload_folder(service, assets_id, name)
    print("done")


if __name__ == "__main__":
    main()
