import json
from collections import deque
from typing import Optional, Iterable, List, Dict, Tuple
from planning_structs.domain import Domain, Type
from planning_structs.instance import Instance
from pathlib import Path


def load_domain_from_json(path: Path) -> Domain:
    """
    Build a Domain from a JSON file with keys:
      {
        "name": "blocksworld",       # optional
        "types":      [{"name": "...", "parent": "..." | null}, ...],
        "predicates": [{"name": "...", "types": ["T1","T2",...]}, ...],
        "actions":    [{"name": "...", "types": ["T1","T2",...]}, ...]
      }
    """
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    # Basic shape checks (lightweight)
    for key in ("types", "predicates", "action_schemas"):
        if key not in spec or not isinstance(spec[key], list):
            raise ValueError(f"JSON must contain a list '{key}'")

    types_spec = [(t["name"], t.get("parent")) for t in spec["types"]]
    preds_spec = [(p["name"], p.get("types", [])) for p in spec["predicates"]]
    actions_spec = [(a["name"], a.get("types", [])) for a in spec["action_schemas"]]

    # Optional domain name
    domain_name = spec.get("name", None)

    return Domain(types_spec, preds_spec, actions_spec, name=domain_name)


def dump_domain_to_json(domain: Domain, path: Optional[Path] = None, compact: bool = True) -> str:
    """
    Serialize a Domain to a JSON string (and optionally write to `path`).
    The JSON matches the loader’s expected shape.
    """
    # Construct payload parts explicitly so we can format them
    types = [
        {"name": t.name, "parent": (t.parent.name if t.parent else None)}
        for t in collect_types_hierarchy(domain.types_set)
    ]
    predicates = [
        {"name": p.name, "types": [t.name for t in p.types]}
        for p in domain.predicates
    ]
    actions = [
        {"name": a.name, "types": [t.name for t in a.types]}
        for a in domain.action_schemas
    ]

    if compact:
        s = compact_json_str(getattr(domain, "name", None), types, predicates, actions)
    else:
        payload = {
            "name": getattr(domain, "name", None),
            "types": types,
            "predicates": predicates,
            "action_schemas": actions,
        }
        s = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)

    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(s)
    return s


def collect_types_hierarchy(types: Iterable["Type"]) -> List["Type"]:
    """
    Return types in root→leaf (BFS order).
    - Deterministic: children at each level sorted by name.
    - Supports multiple roots.
    - Works whether or not 'object' exists (Domain ensures it).
    """
    # Build lookups
    name_to_type: Dict[str, "Type"] = {t.name: t for t in types}
    parent_to_children: Dict[str | None, List[str]] = {}
    for t in types:
        p = t.parent.name if t.parent is not None else None
        parent_to_children.setdefault(p, []).append(t.name)

    # Sort children deterministically
    for p in parent_to_children:
        parent_to_children[p].sort()

    # Roots: explicit None-parents; fall back to "object" if missing
    roots = parent_to_children.get(None, [])
    if not roots and "object" in name_to_type:
        roots = ["object"]

    ordered: List["Type"] = []
    q = deque(roots)
    seen = set()

    while q:
        cur = q.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        ordered.append(name_to_type[cur])
        for child in parent_to_children.get(cur, []):
            q.append(child)

    # If any types were disconnected, append them too
    for orphan in sorted(name_to_type.keys()):
        if orphan not in seen:
            ordered.append(name_to_type[orphan])

    return ordered


def compact_json_str(name, types, predicates, actions) -> str:
    def one_line(obj) -> str:
        """Dump obj as compact JSON (no line breaks)."""
        return json.dumps(obj, ensure_ascii=False, separators=(",", ": "))

    lines = []
    lines.append("{")
    lines.append(f'  "name": {json.dumps(name)},')

    # types
    lines.append('  "types": [')
    for i, t in enumerate(types):
        comma = "," if i < len(types) - 1 else ""
        lines.append(f"    {one_line(t)}{comma}")
    lines.append("  ],")

    # predicates
    lines.append('  "predicates": [')
    for i, p in enumerate(predicates):
        comma = "," if i < len(predicates) - 1 else ""
        lines.append(f"    {one_line(p)}{comma}")
    lines.append("  ],")

    # actions
    lines.append('  "action_schemas": [')
    for i, a in enumerate(actions):
        comma = "," if i < len(actions) - 1 else ""
        lines.append(f"    {one_line(a)}{comma}")
    lines.append("  ]")

    lines.append("}")

    text = "\n".join(lines)

    return text


def load_instance_from_json(
    domain: Domain,
    path: Path,
    require_domain_match: bool = False
) -> Instance:
    """
    Build an Instance from a JSON file:
      {
        "domain": "optional-domain-name",
        "objects": [{"name": "...", "type": "..."}, ...]
      }

    - Validates object types against the provided `domain`.
    - If `require_domain_match` is True and the JSON contains "domain",
      raises if it doesn't match `domain.name`.
    """
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    # Basic validation
    if "objects" not in spec or not isinstance(spec["objects"], list):
        raise ValueError("Instance JSON must contain a list 'objects'.")

    # Optional domain-name consistency check
    file_domain_name = spec.get("domain", None)
    if require_domain_match and file_domain_name is not None:
        if getattr(domain, "name", None) != file_domain_name:
            raise ValueError(
                f"Instance domain '{file_domain_name}' does not match provided Domain.name '{domain.name}'."
            )

    # Normalize objects into the (name, type_name) tuples expected by your Instance ctor
    objects_spec: List[Tuple[str, str]] = []
    for obj in spec["objects"]:
        oname = obj["name"]
        tname = obj["type"]
        if domain.get_type(tname) is None:
            raise ValueError(f"Unknown type '{tname}' for object '{oname}'.")
        objects_spec.append((oname, tname))

    # Construct instance
    inst = Instance(domain=domain, objects=objects_spec)
    return inst


def dump_instance_to_json(
    instance: Instance,
    path: Optional[Path] = None,
    include_domain_name: bool = True,
    compact: bool = True
) -> str:
    """
    Serialize an Instance to JSON string (and optionally write to `path`).
    Outputs one line per object entry for readability.

    Fields:
      version: 1
      name: optional (from instance.name if present)
      domain: optional (from instance.domain.name if include_domain_name=True)
      objects: list of {"name","type"}
    """
    # Collect objects in deterministic order (by name)
    objs = instance.objects

    def one_line(obj) -> str:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ": "))

    lines = []
    lines.append("{")

    # domain
    if include_domain_name:
        dname = getattr(instance.domain, "name", None)
        lines.append(f'  "domain": {json.dumps(dname)},')

    if compact:
        # objects (one line per object)
        lines.append('  "objects": [')
        for i, o in enumerate(objs):
            rec = {"name": o.name, "type": o.type.name}
            comma = "," if i < len(objs) - 1 else ""
            lines.append(f"    {one_line(rec)}{comma}")
        lines.append("  ]")
        lines.append("}")

        text = "\n".join(lines)
    else:
        payload = {
            "domain": dname if include_domain_name else None,
            "objects": [{"name": o.name, "type": o.type.name} for o in objs],
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)

    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return text