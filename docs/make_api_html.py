"""Write docs/api.html: every public name in da6idb, with its whole docstring.

The one-line list in ``api.md`` says what exists; this says what each thing
does, because the paragraph after the first is usually the one that matters --
why the sign is derived rather than assumed, which argument is in pixels,
what comes back NaN.

Shares its reading of the source with ``make_api.py``, so the two cannot
disagree. Self-contained: no fonts, scripts or styles are fetched, since the
machine reading it at a beamline may have no network. Run from the
repository root.
"""

import ast
import html
import pathlib
import re
import sys
from datetime import date

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_api import collect                                   # noqa: E402

OUT = pathlib.Path("docs/api.html")

# ``x`` is a literal, :func:`x` and friends are cross-references we have no
# index for, and *x* is an argument name. All three end up as plain code.
_LITERAL = re.compile(r"``(.+?)``", re.S)
_ROLE = re.compile(r":[a-z]+:`~?([^`]+)`")
_EMPHASIS = re.compile(r"(?<![\w*])\*([^\s*][^*\n]*?)\*(?![\w*])")


def version():
    tree = ast.parse(pathlib.Path("da6idb/__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(
                node.targets[0], "id", "") == "__version__":
            return ast.literal_eval(node.value)
    return ""


def inline(text):
    """RST-ish markup in one paragraph, as HTML."""
    out = html.escape(text)
    out = _LITERAL.sub(lambda m: f"<code>{m.group(1)}</code>", out)
    out = _ROLE.sub(lambda m: f"<code>{m.group(1).rsplit('.', 1)[-1]}</code>", out)
    return _EMPHASIS.sub(lambda m: f"<em>{m.group(1)}</em>", out)


def prose(doc):
    """A whole docstring as HTML: paragraphs, and indented blocks verbatim."""
    if not doc:
        return '<p class="none">No description.</p>'
    blocks = []
    for chunk in re.split(r"\n\s*\n", doc.strip()):
        lines = chunk.split("\n")
        # An indented run is an example or a table; print it as it was typed.
        if all(line.startswith("    ") or not line.strip() for line in lines):
            body = "\n".join(line[4:] for line in lines)
            blocks.append(f"<pre>{html.escape(body)}</pre>")
        else:
            joined = " ".join(line.strip() for line in lines)
            blocks.append(f"<p>{inline(joined)}</p>")
    return "\n".join(blocks)


def anchor(*parts):
    return "-".join(parts).replace(".", "-").replace("_", "-")


def render():
    modules = collect()
    functions = sum(
        len([i for i in m["items"] if i["kind"] == "func"])
        + sum(len(i["methods"]) for i in m["items"] if i["kind"] == "class")
        for m in modules
    )
    classes = sum(len([i for i in m["items"] if i["kind"] == "class"])
                  for m in modules)

    nav, body = [], []
    for module in modules:
        mid = anchor(module["module"])
        names = []
        for item in module["items"]:
            names.append(
                f'<a class="n" href="#{anchor(module["module"], item["name"])}">'
                f'{html.escape(item["name"])}</a>'
            )
        nav.append(
            f'<li><a class="m" href="#{mid}">{html.escape(module["module"])}</a>'
            f'<div class="sub">{"".join(names)}</div></li>'
        )

        entries = []
        for item in module["items"]:
            eid = anchor(module["module"], item["name"])
            if item["kind"] == "func":
                entries.append(f"""
      <article class="entry" id="{eid}" data-name="{html.escape(item['name'])} {html.escape(item['summary'].lower())}">
        <h3><a href="#{eid}">{html.escape(item['name'])}</a><span class="sig">{html.escape(item['sig'])}</span></h3>
        <div class="doc">{prose(item['doc'])}</div>
      </article>""")
                continue

            # A class is one filterable block, so its methods' names have to
            # be searchable through it -- otherwise looking for "two_theta"
            # hides the class that has it.
            haystack = " ".join([item["name"], item["summary"].lower()]
                                + item["props"]
                                + [m["name"] for m in item["methods"]]
                                + [m["summary"].lower() for m in item["methods"]])
            inner = []
            if item["props"]:
                tags = " ".join(f"<code>{html.escape(p)}</code>"
                                for p in item["props"])
                inner.append(f'<p class="props"><span>properties</span> {tags}</p>')
            for m in item["methods"]:
                mid2 = anchor(module["module"], item["name"], m["name"])
                inner.append(f"""
        <div class="method" id="{mid2}">
          <h4><a href="#{mid2}">.{html.escape(m['name'])}</a><span class="sig">{html.escape(m['sig'])}</span></h4>
          <div class="doc">{prose(m['doc'])}</div>
        </div>""")
            entries.append(f"""
      <article class="entry cls" id="{eid}" data-name="{html.escape(haystack)}">
        <h3><span class="kind">class</span><a href="#{eid}">{html.escape(item['name'])}</a></h3>
        <div class="doc">{prose(item['doc'])}</div>
        {"".join(inner)}
      </article>""")

        body.append(f"""
    <section class="module" id="{mid}">
      <h2><a href="#{mid}">{html.escape(module['module'])}</a></h2>
      <p class="lead">{inline(module['summary'])}</p>
      {"".join(entries)}
    </section>""")

    return TEMPLATE.format(
        version=html.escape(version()),
        generated=date.today().isoformat(),
        functions=functions,
        classes=classes,
        modules=len(modules),
        nav="\n".join(nav),
        body="\n".join(body),
    )


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>da6idb API</title>
<style>
  :root {{
    --bg: #fbfbfa; --panel: #fff; --ink: #1b1b1a; --dim: #6a6a66;
    --line: #e3e2de; --accent: #8a5a2b; --code-bg: #f3f2ef;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg: #17181a; --panel: #1e1f22; --ink: #e6e5e2; --dim: #9b9a95;
      --line: #2e2f33; --accent: #d79b63; --code-bg: #26272b;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #17181a; --panel: #1e1f22; --ink: #e6e5e2; --dim: #9b9a95;
    --line: #2e2f33; --accent: #d79b63; --code-bg: #26272b;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  code, pre, .sig {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }}
  header {{
    padding: 26px 16px 20px; border-bottom: 1px solid var(--line);
    background: var(--panel);
  }}
  .wrap {{ max-width: 1180px; margin: 0 auto; }}
  h1 {{ margin: 0 0 6px; font-size: 25px; letter-spacing: -0.01em; }}
  h1 code {{ font-size: 25px; background: none; padding: 0; color: var(--accent); }}
  .meta {{ color: var(--dim); font-size: 13.5px; margin: 0; }}
  #find {{
    margin-top: 16px; width: 100%; padding: 9px 12px; font-size: 14px;
    border: 1px solid var(--line); border-radius: 7px;
    background: var(--bg); color: var(--ink);
  }}
  #find:focus {{ outline: 2px solid var(--accent); outline-offset: -1px; }}
  .layout {{ display: flex; gap: 30px; align-items: flex-start;
             max-width: 1180px; margin: 0 auto; padding: 0 16px 70px; }}
  nav {{
    width: 248px; flex: none; position: sticky; top: 16px;
    max-height: calc(100vh - 32px); overflow-y: auto; padding-top: 22px;
    font-size: 13.5px;
  }}
  nav ul {{ list-style: none; margin: 0; padding: 0; }}
  nav li {{ margin-bottom: 12px; }}
  nav a {{ text-decoration: none; color: var(--ink); display: block; }}
  nav a.m {{ font-weight: 600; color: var(--accent);
             font-family: ui-monospace, Menlo, monospace; font-size: 13px; }}
  nav .sub {{ display: flex; flex-direction: column; margin: 3px 0 0 9px;
              border-left: 1px solid var(--line); }}
  nav a.n {{ color: var(--dim); padding: 1px 0 1px 9px; }}
  nav a.n:hover, nav a.m:hover {{ color: var(--accent); }}
  main {{ flex: 1; min-width: 0; padding-top: 22px; }}
  section.module {{ margin-bottom: 40px; }}
  h2 {{ font-size: 19px; margin: 0 0 4px;
        font-family: ui-monospace, Menlo, monospace; }}
  h2 a {{ color: var(--accent); text-decoration: none; }}
  .lead {{ margin: 0 0 16px; color: var(--dim); }}
  .entry {{
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 9px; padding: 13px 16px; margin-bottom: 11px;
  }}
  .entry h3 {{ margin: 0 0 7px; font-size: 15px; font-weight: 600; }}
  .entry h3 a {{ color: var(--ink); text-decoration: none;
                 font-family: ui-monospace, Menlo, monospace; }}
  .entry h3 a:hover {{ color: var(--accent); }}
  .kind {{ color: var(--accent); font-weight: 500; font-size: 12px;
           margin-right: 7px; text-transform: uppercase; letter-spacing: .05em; }}
  .sig {{ color: var(--dim); font-weight: 400; font-size: 13px;
          word-break: break-word; }}
  .doc p {{ margin: 0 0 9px; }}
  .doc p:last-child {{ margin-bottom: 0; }}
  .doc .none {{ color: var(--dim); font-style: italic; }}
  pre {{ background: var(--code-bg); padding: 9px 11px; border-radius: 6px;
         overflow-x: auto; font-size: 13px; margin: 0 0 9px; }}
  code {{ background: var(--code-bg); padding: 1px 4px; border-radius: 4px;
          font-size: 12.8px; }}
  .props {{ margin: 9px 0; font-size: 13.5px; }}
  .props span {{ color: var(--dim); margin-right: 5px; }}
  .method {{ border-top: 1px solid var(--line); margin-top: 11px;
             padding-top: 11px; }}
  .method h4 {{ margin: 0 0 6px; font-size: 14px; font-weight: 600; }}
  .method h4 a {{ color: var(--ink); text-decoration: none;
                  font-family: ui-monospace, Menlo, monospace; }}
  .method h4 a:hover {{ color: var(--accent); }}
  .hidden {{ display: none; }}
  #empty {{ color: var(--dim); padding: 24px 2px; }}
  @media (max-width: 820px) {{
    .layout {{ flex-direction: column; gap: 0; }}
    nav {{ position: static; width: 100%; max-height: none; }}
  }}
</style>
</head>
<body>
<header><div class="wrap">
  <h1><code>da6idb</code> — every public function</h1>
  <p class="meta">
    Version {version} · {functions} callables, {classes} classes across
    {modules} modules · generated from the source on {generated} by
    <code>python docs/make_api_html.py</code>
  </p>
  <p class="meta">
    Nothing here needs Qt. The reading modules need only numpy, h5py and
    scipy; <code>da6idb.session</code> adds pandas and xarray;
    <code>da6idb.hkl</code> adds xrayutilities and pyvista.
  </p>
  <input id="find" type="search" placeholder="Filter by name or description…"
         autocomplete="off" spellcheck="false">
</div></header>
<div class="layout">
  <nav><ul>
{nav}
  </ul></nav>
  <main>
{body}
    <p id="empty" class="hidden">Nothing matches that.</p>
  </main>
</div>
<script>
  const box = document.getElementById('find');
  const entries = [...document.querySelectorAll('.entry')];
  const sections = [...document.querySelectorAll('section.module')];
  const empty = document.getElementById('empty');
  box.addEventListener('input', () => {{
    const q = box.value.trim().toLowerCase();
    let shown = 0;
    for (const entry of entries) {{
      const hit = !q || entry.dataset.name.includes(q);
      entry.classList.toggle('hidden', !hit);
      if (hit) shown++;
    }}
    for (const section of sections) {{
      const any = section.querySelector('.entry:not(.hidden)');
      section.classList.toggle('hidden', !any);
    }}
    empty.classList.toggle('hidden', shown > 0);
  }});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    OUT.write_text(render())
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} kB)")
