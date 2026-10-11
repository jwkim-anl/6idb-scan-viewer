"""Pull the public API out of the source, for docs/api.md and docs/api.html.

One AST walk feeds both, so the two pages cannot drift from each other or
from the code. Run from the repository root:

    python docs/make_api.py > docs/api.md        # the list of one-liners
    python docs/make_api_html.py                 # the full reference
"""

import ast, pathlib, sys


def docstring(node):
    """The whole docstring, blank lines and all."""
    return (ast.get_docstring(node) or "").strip()


def summary(node):
    """Just the first paragraph, on one line."""
    first = docstring(node).split("\n\n")[0].replace("\n", " ").strip()
    return " ".join(first.split())


def sig(fn):
    a = fn.args
    parts = [x.arg for x in a.posonlyargs + a.args]
    defaults = [None] * (len(parts) - len(a.defaults)) + list(a.defaults)
    out = []
    for name, d in zip(parts, defaults):
        out.append(name if d is None else f"{name}={ast.unparse(d)}")
    if a.vararg: out.append("*" + a.vararg.arg)
    for k, d in zip(a.kwonlyargs, a.kw_defaults):
        out.append(k.arg if d is None else f"{k.arg}={ast.unparse(d)}")
    if a.kwarg: out.append("**" + a.kwarg.arg)
    return "(" + ", ".join(out) + ")"


def collect(root="da6idb"):
    """[{module, doc, summary, items}] for every module with public names."""
    found = []
    for path in sorted(pathlib.Path(root).rglob("*.py")):
        if path.name == "__init__.py" and path.parent.name == "hkl":
            continue
        tree = ast.parse(path.read_text())
        items = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                methods = [m for m in node.body if isinstance(m, ast.FunctionDef)
                           and not m.name.startswith("_")]
                props = [m.name for m in methods
                         if any(getattr(d, "id", "") == "property"
                                for d in m.decorator_list)]
                items.append({
                    "kind": "class", "name": node.name, "sig": "",
                    "summary": summary(node), "doc": docstring(node),
                    "props": props,
                    "methods": [{"name": m.name, "sig": sig(m),
                                 "summary": summary(m), "doc": docstring(m)}
                                for m in methods if m.name not in props],
                })
            elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                items.append({
                    "kind": "func", "name": node.name, "sig": sig(node),
                    "summary": summary(node), "doc": docstring(node),
                    "props": [], "methods": [],
                })
        if items:
            found.append({
                "module": str(path.with_suffix("")).replace("/", "."),
                "summary": summary(tree), "doc": docstring(tree), "items": items,
            })
    return found


HEADER = """\
# `da6idb` — every public function

Generated from the source: `python docs/make_api.py > docs/api.md` for this
list, `python docs/make_api_html.py` for `api.html`, which is the same thing
with every docstring in full. Regenerate rather than edit, so neither can
drift from the code.

Nothing here needs Qt. The reading modules need only numpy, h5py and scipy;
`da6idb.session` adds pandas and xarray; `da6idb.hkl` adds xrayutilities and
pyvista.

| module | what it is for |
|---|---|
| `da6idb.files` | finding a scan's files and pairing master with detector |
| `da6idb.scan` | the master: metadata, streams, rasters, peak statistics |
| `da6idb.fit` | peak shapes fitted to a curve, one or several |
| `da6idb.combine` | which scans are repeats, and averaging them |
| `da6idb.images` | the detector stack, its channels and its stored ROIs |
| `da6idb.rois` | rectangles, and what they reduce each frame to |
| `da6idb.hkl.geometry` | the instrument description, overrides, attenuator |
| `da6idb.hkl.convert` | frames to a point cloud in reciprocal space |
| `da6idb.hkl.detector` | the scattering angle a detector pixel sees |
| `da6idb.hkl.grid` | binning onto a regular lattice, cached as `.vti` |
| `da6idb.hkl.slice` | planes and slabs through a grid |
| `da6idb.batch` | gridding a run of scans onto one lattice |
| `da6idb.export` | writing slices and cuts to text, CSV, npz or HDF5 |
| `da6idb.session` | the objects a notebook starts from |
"""


if __name__ == "__main__":
    total_f = total_c = 0
    print(HEADER, end="")
    for module in collect():
        print(f"\n### `{module['module']}`\n")
        print(f"{module['summary']}\n")
        for item in module["items"]:
            if item["kind"] == "func":
                total_f += 1
                print(f"- `{item['name']}{item['sig']}` — {item['summary']}")
            else:
                total_c += 1
                print(f"- **`{item['name']}`** — {item['summary']}")
                if item["props"]:
                    print("  - properties: "
                          + ", ".join("`" + p + "`" for p in item["props"]))
                for m in item["methods"]:
                    total_f += 1
                    print(f"  - `.{m['name']}{m['sig']}` — {m['summary']}")
    print(f"\n<!-- {total_f} callables, {total_c} classes -->")
