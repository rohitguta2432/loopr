"""A tiny local web UI for Loopr - watch the loop optimize a prompt live.

Zero extra dependencies: a stdlib ThreadingHTTPServer streams each iteration to
the browser over Server-Sent Events as the loop runs against your local model.

    loopr serve            # then open http://localhost:8077
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from loopr.llm import LLMClient, LLMError
from loopr.loop import optimize
from loopr.task import Task

TASKS_DIR = os.path.join(os.path.dirname(__file__), "tasks")


def _bundled_tasks() -> list[str]:
    if not os.path.isdir(TASKS_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(TASKS_DIR) if f.endswith((".yaml", ".yml")))


def _load_task(name: str) -> Task:
    safe = os.path.basename(name)
    for ext in (".yaml", ".yml"):
        path = os.path.join(TASKS_DIR, safe + ext)
        if os.path.isfile(path):
            return Task.from_file(path)
    raise FileNotFoundError(f"no bundled task named {name!r}")


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Loopr - self-improving prompt loop</title>
<style>
  :root{--bg:#0b0e14;--panel:#121723;--line:#222a3a;--ink:#e6ebf5;--dim:#8b97ad;
        --accent:#6ee7b7;--accent2:#60a5fa;--bad:#f87171;--chip:#1b2233;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:15px/1.5 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif}
  .wrap{max-width:1080px;margin:0 auto;padding:28px 22px 60px}
  header{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:6px}
  h1{font-size:26px;margin:0;letter-spacing:-.4px}
  h1 .lp{color:var(--accent)}
  .tag{color:var(--dim);font-size:14px}
  .sub{color:var(--dim);font-size:13px;margin:2px 0 22px}
  .grid{display:grid;grid-template-columns:300px 1fr;gap:20px}
  @media(max-width:780px){.grid{grid-template-columns:1fr}}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
  .lbl{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);margin:0 0 8px}
  .tasks{display:flex;flex-wrap:wrap;gap:8px}
  .task{background:var(--chip);border:1px solid var(--line);color:var(--ink);border-radius:8px;
        padding:7px 11px;font-size:13px;cursor:pointer;font-family:ui-monospace,monospace}
  .task.on{border-color:var(--accent);color:var(--accent)}
  pre{background:#0a0d15;border:1px solid var(--line);border-radius:8px;padding:11px;margin:0;
      white-space:pre-wrap;word-break:break-word;font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:#cdd6e6}
  .run{margin-top:14px;width:100%;background:var(--accent);color:#06281d;border:0;border-radius:9px;
       padding:11px;font-weight:700;font-size:14px;cursor:pointer}
  .run:disabled{opacity:.5;cursor:default}
  .meta{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 14px}
  .pill{font-size:12px;color:var(--dim);background:var(--chip);border:1px solid var(--line);
        border-radius:999px;padding:3px 10px}
  .iter{border:1px solid var(--line);border-radius:11px;padding:14px;margin-bottom:12px;background:#0f1420}
  .iter.best{border-color:var(--accent)}
  .ihead{display:flex;align-items:center;gap:12px;margin-bottom:9px}
  .iname{font-family:ui-monospace,monospace;font-size:13px;color:var(--dim);min-width:54px}
  .bar{flex:1;height:9px;background:#0a0d15;border-radius:6px;overflow:hidden}
  .fill{height:100%;background:linear-gradient(90deg,var(--accent2),var(--accent));
        width:0;transition:width .5s ease}
  .pct{font-variant-numeric:tabular-nums;font-weight:700;min-width:62px;text-align:right;font-size:13px}
  .tag-best{color:var(--accent);font-size:11px;font-weight:700}
  .reflect{margin-top:10px;border-left:2px solid var(--accent2);padding-left:11px}
  .reflect .lbl{margin-bottom:5px}
  .done{border:1px solid var(--accent);background:rgba(110,231,183,.07);border-radius:11px;
        padding:14px;margin-bottom:14px}
  .done b{color:var(--accent)}
  .empty{color:var(--dim);text-align:center;padding:40px 0}
  a{color:var(--accent2)}
</style></head><body><div class="wrap">
  <header>
    <h1><span class="lp">loop</span>r</h1>
    <span class="tag">self-improving prompt-optimization loop</span>
  </header>
  <p class="sub">Pick a task. The loop runs your prompt, scores it, reflects on the failures, rewrites it, and repeats - live, against your local model. &nbsp;·&nbsp; <a href="https://github.com/rohitguta2432/loopr" target="_blank">source</a></p>
  <div class="grid">
    <div class="panel">
      <p class="lbl">Task</p>
      <div class="tasks" id="tasks"></div>
      <p class="lbl" style="margin-top:16px">Seed prompt</p>
      <pre id="seed">-</pre>
      <button class="run" id="run">Optimize ▸</button>
    </div>
    <div class="panel">
      <div class="meta" id="meta"></div>
      <div id="stream"><div class="empty">Pick a task and hit Optimize to watch the loop work.</div></div>
    </div>
  </div>
</div>
<script>
let TASKS=[], current=null, seeds={}, running=false;
const $=s=>document.querySelector(s);
function esc(t){return (t||"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
async function boot(){
  const r=await fetch("/api/tasks"); const d=await r.json();
  TASKS=d.tasks; seeds=d.seeds;
  const box=$("#tasks"); box.innerHTML="";
  TASKS.forEach(t=>{const b=document.createElement("button");b.className="task";b.textContent=t;
    b.onclick=()=>pick(t);box.appendChild(b);});
  if(TASKS.length) pick(TASKS[0]);
}
function pick(t){current=t;$("#seed").textContent=seeds[t]||"-";
  document.querySelectorAll(".task").forEach(b=>b.classList.toggle("on",b.textContent===t));}
function iterCard(it){
  const pct=Math.round(it.mean_score*100);
  const div=document.createElement("div");div.className="iter"+(it.is_best?" best":"");
  div.innerHTML=`<div class="ihead"><span class="iname">iter ${it.n}</span>
    <div class="bar"><div class="fill"></div></div>
    <span class="pct">${pct}% · ${it.passed}/${it.total}</span>
    ${it.is_best?'<span class="tag-best">▲ best</span>':''}</div>`+
    (it.proposed_prompt?`<div class="reflect"><p class="lbl">reflection → rewrite</p><pre>${esc(it.proposed_prompt)}</pre></div>`:'');
  $("#stream").appendChild(div);
  requestAnimationFrame(()=>{div.querySelector(".fill").style.width=pct+"%";});
}
function run(){
  if(running||!current)return; running=true;
  $("#run").disabled=true; $("#stream").innerHTML=""; $("#meta").innerHTML='<span class="pill">running…</span>';
  const es=new EventSource("/api/optimize?task="+encodeURIComponent(current));
  es.addEventListener("iteration",e=>iterCard(JSON.parse(e.data)));
  es.addEventListener("done",e=>{const d=JSON.parse(e.data);
    const banner=document.createElement("div");banner.className="done";
    banner.innerHTML=`<b>✅ ${d.stop_reason}</b> — seed ${Math.round(d.seed_score*100)}% → best <b>${Math.round(d.best_score*100)}%</b> (+${Math.round(d.improvement*100)}%, iter ${d.best_iteration})
      <p class="lbl" style="margin:12px 0 5px">best prompt</p><pre>${esc(d.best_prompt)}</pre>`;
    $("#stream").prepend(banner);
    $("#meta").innerHTML=`<span class="pill">${d.stop_reason}</span><span class="pill">${d.iterations} iterations</span><span class="pill">+${Math.round(d.improvement*100)}%</span>`;
    es.close();running=false;$("#run").disabled=false;});
  es.addEventListener("error",e=>{let m="LLM backend unavailable - start Ollama or set an API key.";
    try{m=JSON.parse(e.data).error||m;}catch(_){}
    $("#meta").innerHTML=`<span class="pill" style="color:var(--bad);border-color:var(--bad)">${esc(m)}</span>`;
    es.close();running=false;$("#run").disabled=false;});
}
$("#run").onclick=run; boot();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif parsed.path == "/api/tasks":
            names = _bundled_tasks()
            seeds = {n: _load_task(n).seed_prompt for n in names}
            self._send(200, json.dumps({"tasks": names, "seeds": seeds}).encode(), "application/json")
        elif parsed.path == "/api/optimize":
            self._optimize_sse(parse_qs(parsed.query).get("task", [""])[0])
        else:
            self._send(404, b"not found", "text/plain")

    def _sse_event(self, event: str, data: dict) -> None:
        chunk = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
        self.wfile.write(chunk)
        self.wfile.flush()

    def _optimize_sse(self, task_name: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            task = _load_task(task_name)
        except FileNotFoundError as exc:
            self._sse_event("error", {"error": str(exc)})
            return
        client = LLMClient(temperature=task.config.temperature)
        try:
            result = optimize(
                task,
                client=client,
                on_iteration=lambda it: self._sse_event(
                    "iteration",
                    {
                        "n": it.n,
                        "mean_score": it.mean_score,
                        "passed": it.passed(),
                        "total": len(it.case_results),
                        "is_best": it.is_best,
                        "proposed_prompt": it.proposed_prompt,
                    },
                ),
            )
        except LLMError as exc:
            self._sse_event("error", {"error": f"LLM backend unavailable: {exc}"})
            return
        except (BrokenPipeError, ConnectionResetError):
            return
        payload = result.to_dict()
        payload["iterations"] = len(result.iterations)
        self._sse_event("done", payload)


def serve(host: str = "127.0.0.1", port: int = 8077) -> None:
    """Start the local Loopr web UI."""
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"🔁 loopr web UI → {url}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
        server.shutdown()


if __name__ == "__main__":
    serve(port=int(os.getenv("LOOPR_PORT", "8077")))
