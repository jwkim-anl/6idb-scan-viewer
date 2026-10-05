import ast, pathlib, sys

def summary(node):
    doc = ast.get_docstring(node) or ""
    first = doc.strip().split("\n\n")[0].replace("\n", " ").strip()
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

rows = []
for path in sorted(pathlib.Path("da6idb").rglob("*.py")):
    if path.name == "__init__.py" and path.parent.name == "hkl":
        continue
    mod = str(path.with_suffix("")).replace("/", ".")
    tree = ast.parse(path.read_text())
    items = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            methods = [m for m in node.body if isinstance(m, ast.FunctionDef)
                       and not m.name.startswith("_")]
            props = [m.name for m in methods
                     if any(getattr(d, "id", "") == "property" for d in m.decorator_list)]
            items.append(("class", node.name, "", summary(node),
                          [(m.name, sig(m), summary(m)) for m in methods
                           if m.name not in props], props))
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            items.append(("func", node.name, sig(node), summary(node), [], []))
    if items:
        rows.append((mod, summary(tree), items))
    sys.modules.setdefault("_", None)

if __name__ == "__main__":
    total_f = total_c = 0
    for mod, doc, items in rows:
        print(f"\n### `{mod}`\n")
        print(f"{doc}\n")
        for kind, name, s, d, methods, props in items:
            if kind == "func":
                total_f += 1
                print(f"- `{name}{s}` — {d}")
            else:
                total_c += 1
                print(f"- **`{name}`** — {d}")
                if props:
                    print(f"  - properties: {', '.join('`'+p+'`' for p in props)}")
                for m, ms, md in methods:
                    total_f += 1
                    print(f"  - `.{m}{ms}` — {md}")
    print(f"\n<!-- {total_f} callables, {total_c} classes -->")
