"""Download Poly Haven ground maps and store them on Google Drive.

Each texture becomes MyDrive/PhysicalAI_adv/assets/texture/<slug>/ with
2K JPG files named diffuse.jpg, roughness.jpg, and normal.jpg. The normal
map is the OpenGL (nor_gl) variant. Existing files are left in place.

Maps are cached under .cache/polyhaven_textures so a later Drive upload
can reuse the same files.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import google.auth
from google.auth.exceptions import DefaultCredentialsError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

API = "https://api.polyhaven.com"
RESOLUTION = "2k"
FORMAT = "jpg"
USER_AGENT = "robot-rollouts-ground-textures"
SCOPES = ["https://www.googleapis.com/auth/drive"]
DRIVE_PATH = ("PhysicalAI_adv", "assets", "texture")
ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / ".cache" / "polyhaven_textures"
CREDENTIALS_PATH = ROOT / "credentials.json"
TOKEN_PATH = ROOT / "token.json"

MAPS = (
    ("Diffuse", "diffuse.jpg"),
    ("Rough", "roughness.jpg"),
    ("nor_gl", "normal.jpg"),
)

TEXTURES = (
    "leafy_grass",
    "withered_grass",
    "forest_floor",
    "forest_ground_04",
    "forest_leaves_02",
    "forrest_ground_03",
    "coast_sand_01",
    "red_sand",
    "dirt",
    "brown_mud_03",
    "mud_cracked_dry_03",
    "gravel_floor",
    "ganges_river_pebbles",
    "rocks_ground_08",
    "snow_02",
    "asphalt_01",
    "concrete_floor",
    "cobblestone_floor_01",
    "brown_planks_03",
    "rubberized_track",
)

FOLDER_MIME = "application/vnd.google-apps.folder"


def _http_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _http_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def map_url(files: dict, map_key: str) -> str:
    node = files.get(map_key, {}).get(RESOLUTION, {}).get(FORMAT)
    if not node or "url" not in node:
        raise KeyError(f"{map_key}/{RESOLUTION}/{FORMAT}")
    return node["url"]


def cache_texture(slug: str) -> Path:
    dest = CACHE_ROOT / slug
    dest.mkdir(parents=True, exist_ok=True)
    missing = [name for _, name in MAPS if not (dest / name).exists()]
    if not missing:
        print(f"  cache {slug}")
        return dest
    files = _http_json(f"{API}/files/{slug}")
    for map_key, filename in MAPS:
        path = dest / filename
        if path.exists():
            print(f"  cache {slug}/{filename}")
            continue
        url = map_url(files, map_key)
        print(f"  get   {slug}/{filename}")
        path.write_bytes(_http_bytes(url))
    return dest


def drive_service():
    try:
        creds, _ = google.auth.default(scopes=SCOPES)
        if creds and not creds.valid:
            creds.refresh(Request())
        if creds and creds.valid:
            print("Using gcloud application-default credentials for Drive.", flush=True)
            return build("drive", "v3", credentials=creds)
    except DefaultCredentialsError as exc:
        print(f"gcloud credentials unavailable ({exc}).", flush=True)

    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_PATH.exists():
                raise FileNotFoundError(f"missing {CREDENTIALS_PATH}")
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            print("Authorize Google Drive in the browser window.", flush=True)
            creds = flow.run_local_server(port=0, open_browser=True)
        TOKEN_PATH.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds)


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def find_child(service, parent_id: str, name: str, folder: bool) -> str | None:
    mime = f" and mimeType = '{FOLDER_MIME}'" if folder else ""
    query = (
        f"'{parent_id}' in parents and name = '{_escape(name)}' "
        f"and trashed = false{mime}"
    )
    listed = (
        service.files()
        .list(q=query, spaces="drive", fields="files(id, name)", pageSize=1)
        .execute()
    )
    files = listed.get("files") or []
    return files[0]["id"] if files else None


def ensure_folder(service, parent_id: str, name: str) -> str:
    existing = find_child(service, parent_id, name, folder=True)
    if existing:
        return existing
    created = (
        service.files()
        .create(
            body={"name": name, "mimeType": FOLDER_MIME, "parents": [parent_id]},
            fields="id",
        )
        .execute()
    )
    return created["id"]


def ensure_path(service, parts: tuple[str, ...]) -> str:
    parent_id = "root"
    for name in parts:
        parent_id = ensure_folder(service, parent_id, name)
    return parent_id


def upload_if_missing(service, folder_id: str, name: str, data: bytes) -> str:
    existing = find_child(service, folder_id, name, folder=False)
    if existing:
        return "skip"
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype="image/jpeg", resumable=False)
    service.files().create(
        body={"name": name, "parents": [folder_id]},
        media_body=media,
        fields="id",
    ).execute()
    return "upload"


def upload_texture(service, texture_root: str, slug: str, local: Path) -> None:
    folder_id = ensure_folder(service, texture_root, slug)
    for _, filename in MAPS:
        if find_child(service, folder_id, filename, folder=False):
            print(f"  skip  {slug}/{filename}")
            continue
        print(f"  put   {slug}/{filename}")
        upload_if_missing(service, folder_id, filename, (local / filename).read_bytes())


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    cached: list[tuple[str, Path]] = []
    for slug in TEXTURES:
        print(slug)
        try:
            cached.append((slug, cache_texture(slug)))
        except (KeyError, urllib.error.HTTPError) as exc:
            print(f"  fail {slug}: {exc}")
            raise
    skip_upload = "--cache-only" in sys.argv
    if skip_upload:
        print(f"cached {len(cached)} textures under {CACHE_ROOT}")
        return
    service = drive_service()
    texture_root = ensure_path(service, DRIVE_PATH)
    for slug, local in cached:
        print(f"upload {slug}")
        upload_texture(service, texture_root, slug, local)
    print(f"done {len(TEXTURES)} textures")


if __name__ == "__main__":
    main()
