"""Static HTML preview of generated cells: N items per (task, tier).

    <out>/index.html            one row per task, links to task pages
    <out>/<task>.html           five tiers x N items: image, question, answer
    <out>/img/<id>.png          copies of the shown images
"""
from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

from visro.api import RELEASE_VERSION, load_adapter, load_verifier
from visro.registry import TASKS, TIERS, category

CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#6e6e73;--line:#d2d2d7;--card:#f5f5f7;--ok:#1a7f37;--bad:#cf222e}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#f2f2f2;--muted:#a1a1a6;--line:#3a3a3c;--card:#232326}}
body{background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;padding:24px 16px}
main{max-width:1280px;margin:0 auto}a{color:inherit}
h1{font-size:24px;margin:0 0 4px}h2{font-size:19px;margin:28px 0 8px}.muted{color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px;overflow:hidden}
.card img{width:100%;height:auto;border-radius:6px;background:#fff;image-rendering:auto}
.q{white-space:pre-wrap;margin:8px 0 6px;font-size:13.5px}.a{font-weight:600}.a code{font-size:14px}
.tag{display:inline-block;font-size:12px;padding:1px 7px;border-radius:999px;border:1px solid var(--line);margin-right:4px}
.ok{color:var(--ok)}.bad{color:var(--bad)}code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-all}
"""


def _page(title: str, body: str) -> str:
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
            f"<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>{body}</main></body></html>\n")


def build_preview(cells_root: Path, out: Path, *, per_tier: int = 3, tasks: list[str] | None = None,
                  verify: bool = True) -> Path:
    from PIL import Image
    from visro.checks import rebuild
    from visro.driver import load_cell

    cells_root, out = Path(cells_root), Path(out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    tasks = tasks or [t for t in TASKS if any((cells_root / f"{t}_{tier}").is_dir() for tier in TIERS)]
    rows = []
    for task in tasks:
        adapter = load_adapter(task)
        verifier = load_verifier(task) if verify else None
        parts = [f"<p><a href='index.html'>&larr; all tasks</a></p><h1>{html.escape(task)}</h1>"
                 f"<p class=muted>category {category(task)} &middot; release {RELEASE_VERSION}</p>"]
        if verifier is not None:
            parts.append(f"<p class=muted><b>Correctness check:</b> {html.escape(verifier.CORRECTNESS)}<br>"
                         f"<b>Uniqueness check:</b> {html.escape(verifier.UNIQUENESS)}</p>")
        n_ok = n_all = 0
        for tier in TIERS:
            cell = cells_root / f"{task}_{tier}"
            parts.append(f"<h2>{tier}</h2><p class=muted>{html.escape(adapter.TIER_DEFINITIONS[tier])}</p><div class=grid>")
            recs = []
            if (cell / "records.jsonl").exists():
                with open(cell / "records.jsonl") as fh:
                    for line in fh:
                        recs.append(json.loads(line))
                        if len(recs) >= per_tier:
                            break
            for k, rec in enumerate(recs):
                name = Path(rec["image"]).name
                shutil.copyfile(cell / rec["image"], out / "img" / name)
                status = ""
                if verifier is not None:
                    # the verifier needs the in-memory record: rebuild it from cell.json and check it is this one
                    full, again, png = rebuild(task, adapter, load_cell(cell), k)
                    probs = [] if again == rec and png == (cell / rec["image"]).read_bytes() else \
                        ["rebuild differs from the written record"]
                    probs += verifier.verify(full, Image.open(cell / rec["image"]).convert("RGB"))
                    n_all += 1
                    n_ok += not probs
                    status = ("<span class='tag ok'>verified</span>" if not probs else
                              f"<span class='tag bad'>FAILED: {html.escape(probs[0])}</span>")
                choices = ""
                if rec.get("options"):
                    choices = f"<div class=muted>options: {html.escape(', '.join(map(str, rec['options'])))}</div>"
                parts.append(
                    f"<div class=card><img loading=lazy src='img/{html.escape(name)}' alt=''>"
                    f"<div class=q>{html.escape(rec['question'])}</div>{choices}"
                    f"<div class=a>answer: <code>{html.escape(rec['answer'])}</code></div>"
                    f"<div class=muted><span class=tag>{html.escape(rec['answer_type'])}</span>{status}"
                    f" <code>{html.escape(rec['id'])}</code></div></div>")
            if not recs:
                parts.append("<p class=bad>no records</p>")
            parts.append("</div>")
        (out / f"{task}.html").write_text(_page(f"{task} preview", "".join(parts)))
        rows.append(f"<tr><td><a href='{task}.html'>{html.escape(task)}</a></td><td>{category(task)}</td>"
                    f"<td>{RELEASE_VERSION}</td>"
                    f"<td>{'' if verifier is None else f'{n_ok}/{n_all}'}</td></tr>")
    body = ("<h1>VisRO preview</h1>"
            f"<p class=muted>{per_tier} items per tier per task, five tiers: {', '.join(TIERS)}.</p>"
            "<table><tr><th>task</th><th>category</th><th>version</th><th>verified</th></tr>" + "".join(rows) + "</table>")
    (out / "index.html").write_text(_page("VisRO preview", body))
    return out / "index.html"
