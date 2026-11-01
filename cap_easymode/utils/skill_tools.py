from pathlib import Path
import re

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"
_SKILL_FILES = {
    "skill": "SKILL.md",
    "reference": "REFERENCE.md",
}
_WRITE_FILES = ("skill", "reference", "script")
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_YAML_PLAIN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9 _-]*[A-Za-z0-9])?$")


def _check_name(name: str, kind: str = "name") -> str:
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValueError(f"invalid {kind}: {name}")
    return name


def _skill_dir(skill_name: str) -> Path:
    _check_name(skill_name, "skill name")
    path = SKILLS_DIR / skill_name
    if not path.is_dir():
        raise ValueError(f"skill not found: {skill_name}")
    return path


def _yaml_quote(value: str) -> str:
    if "\n" in value or "\r" in value:
        raise ValueError("description cannot contain newlines")
    if _YAML_PLAIN.fullmatch(value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _yaml_unquote(raw: str) -> str:
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        inner = raw[1:-1]
        if raw[0] == '"':
            inner = inner.replace('\\"', '"').replace("\\\\", "\\")
        return inner
    return raw


def _front_matter_field(text: str, field: str) -> str | None:
    front = _front_matter(text)
    if not front:
        return None
    prefix = f"{field}:"
    for line in front.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return _yaml_unquote(stripped[len(prefix):].strip())
    return None


def _require_skill_md(text: str, skill_name: str) -> None:
    if not text.startswith("---"):
        raise ValueError("SKILL.md must start with YAML front matter")
    name = _front_matter_field(text, "name")
    if name != skill_name:
        raise ValueError(f"front matter name must be {skill_name}")


def _rel(path: Path) -> str:
    return path.relative_to(SKILLS_DIR.parent).as_posix()


def _report(action: str, path: Path) -> str:
    return f"{action} {_rel(path)}"


def _resolve_write_path(skill_name: str, file_name: str, script_name: str | None = None) -> Path:
    skill_dir = _skill_dir(skill_name)
    if file_name not in _WRITE_FILES:
        raise ValueError(f"unknown file: {file_name}")
    if file_name == "script":
        if not script_name:
            raise ValueError("script_name is required")
        _check_name(script_name, "script name")
        return skill_dir / "scripts" / f"{script_name}.py"
    return skill_dir / _SKILL_FILES[file_name]


def read_skills(skill_name: str, file_name: str = "skill") -> str:
    """Load a skill markdown file. file_name is 'skill' or 'reference'."""
    rel = _SKILL_FILES.get(file_name)
    target = _skill_dir(skill_name) / rel if rel else None
    if target is None or not target.is_file():
        raise ValueError(f"unknown file: {file_name}")
    return target.read_text()


def exec_skill_scripts(skill_name: str, script_name: str, *args, **kwargs):
    """Run a skill script's main() and return its result."""
    script_path = _skill_dir(skill_name) / "scripts" / f"{script_name}.py"
    if not script_path.is_file():
        raise ValueError(f"unknown script: {script_name}")
    namespace = {}
    try:
        exec(compile(script_path.read_text(), str(script_path), "exec"), namespace)
        main = namespace.get("main")
        if not callable(main):
            raise ValueError(f"script has no main(): {script_name}")
        return main(*args, **kwargs)
    except Exception as e:
        raise RuntimeError(f"script failed: {script_name}") from e


def create_skill(name: str, description: str, body: str) -> str:
    """Create skills/<name>/SKILL.md. Fails if that skill already exists."""
    _check_name(name, "skill name")
    if "\n" in description or "\r" in description:
        raise ValueError("description cannot contain newlines")
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    skill_dir = SKILLS_DIR / name
    if skill_dir.exists():
        raise ValueError(f"skill already exists: {name}")
    skill_dir.mkdir()
    text = (
        f"---\nname: {name}\ndescription: {_yaml_quote(description)}\n---\n\n"
        f"{body.strip()}\n"
    )
    path = skill_dir / "SKILL.md"
    path.write_text(text)
    return _report("created", path)


def write_skill_file(
    skill_name: str, file_name: str, content: str, script_name: str | None = None
) -> str:
    """Create or replace a file in an existing skill."""
    path = _resolve_write_path(skill_name, file_name, script_name)
    if file_name == "skill":
        _require_skill_md(content, skill_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return _report("wrote", path)


def replace_in_skill_file(
    skill_name: str,
    file_name: str,
    old: str,
    new: str,
    script_name: str | None = None,
) -> str:
    """Replace one occurrence of old with new in an existing skill file."""
    path = _resolve_write_path(skill_name, file_name, script_name)
    if not path.is_file():
        raise ValueError(f"unknown file: {_rel(path)}")
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise ValueError(f"old must occur exactly once, found {count}")
    updated = text.replace(old, new, 1)
    if file_name == "skill":
        _require_skill_md(updated, skill_name)
    path.write_text(updated)
    return _report("replaced", path)

def _front_matter(text: str) -> str:
    if not text.startswith("---"):
        return ""
    parts = text.split("---", 2)
    if len(parts) < 3:
        return ""
    return parts[1].strip()

def skill_list() -> str:
    if not SKILLS_DIR.is_dir():
        return ""
    blocks = []
    for path in sorted(p for p in SKILLS_DIR.iterdir()):
        skill_md = path / "SKILL.md"
        if not skill_md.is_file():
            continue
        front = _front_matter(skill_md.read_text())
        if front:
            blocks.append(front)
    return "\n---\n".join(blocks)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_skills",
            "description": "Load a skill markdown file. file_name is 'skill' or 'reference'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Skill directory name under skills/.",
                    },
                    "file_name": {
                        "type": "string",
                        "enum": ["skill", "reference"],
                        "description": "Which file to load. Defaults to skill.",
                    },
                },
                "required": ["skill_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exec_skill_scripts",
            "description": "Run a skill script's main() and return its result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Skill directory name under skills/.",
                    },
                    "script_name": {
                        "type": "string",
                        "description": "Script stem under skills/<name>/scripts/, without .py.",
                    },
                },
                "required": ["skill_name", "script_name"],
            },
        },
    },
]

WRITE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_skill",
            "description": "Create skills/<name>/SKILL.md. Fails if that skill already exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "New skill directory name under skills/.",
                    },
                    "description": {
                        "type": "string",
                        "description": "One-line description for YAML front matter. No newlines.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Markdown body after the front matter.",
                    },
                },
                "required": ["name", "description", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_skill_file",
            "description": "Create or replace a file in an existing skill. file_name is 'skill', 'reference', or 'script'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Skill directory name under skills/.",
                    },
                    "file_name": {
                        "type": "string",
                        "enum": ["skill", "reference", "script"],
                        "description": "Which file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file contents.",
                    },
                    "script_name": {
                        "type": "string",
                        "description": "Script stem under skills/<name>/scripts/, without .py. Required when file_name is script.",
                    },
                },
                "required": ["skill_name", "file_name", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_skill_file",
            "description": "Replace exactly one occurrence of old with new in an existing skill file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Skill directory name under skills/.",
                    },
                    "file_name": {
                        "type": "string",
                        "enum": ["skill", "reference", "script"],
                        "description": "Which file to edit.",
                    },
                    "old": {
                        "type": "string",
                        "description": "Exact text that must occur once.",
                    },
                    "new": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                    "script_name": {
                        "type": "string",
                        "description": "Script stem under skills/<name>/scripts/, without .py. Required when file_name is script.",
                    },
                },
                "required": ["skill_name", "file_name", "old", "new"],
            },
        },
    },
]
