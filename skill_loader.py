import re

from config import SKILLS_DIR, WORKDIR

SKILL_REGISTRY: dict[str, dict] = {}

def _parse_frontmatter(raw: str):
    pattern = "^---\n(.*?)\n---\n(.*)"
    result = re.match(pattern,raw,re.DOTALL)
    if not result:
        return None
    meta = result.group(1).splitlines()
    body = result.group(2)
    metaDict = {}
    for s in meta:
        ss = s.split(":",1)
        if len(ss) == 2:
            metaDict[ss[0]] = ss[1]
    return metaDict,body

def _scan_skills():
    if not SKILLS_DIR.exists():
        return
    for d in SKILLS_DIR.iterdir():
        if not d.is_dir():
            continue
        manifest = d / "SKILL.md"
        if manifest.exists():
            raw = manifest.read_text(encoding="utf-8")
            meta,body = _parse_frontmatter(raw)
            if not meta or not body:
                continue
            name = meta.get("name",d.name)
            description = meta.get("description","")
            enabled = meta.get("enabled",True)
            SKILL_REGISTRY[name] = {
                "name":name,
                "description":description,
                "enabled":enabled,
                "content":body
            }
_scan_skills()

def list_skills() -> str:
    return "\n".join(f"- **{s['name']}**: {s['description']}" for s in SKILL_REGISTRY.values())

def build_skill_system() -> str:
    catalog = list_skills()
    return (
        f"Skills available:\n{catalog}\n"
        "Use load_skill to get full details when needed."
    )

SKILL_SYSTEM = build_skill_system()