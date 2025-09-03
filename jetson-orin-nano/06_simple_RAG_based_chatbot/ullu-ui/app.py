import os, httpx, gradio as gr

API = os.getenv("ULLU_API_BASE", "http://localhost:34081")

def index_kb(paths):
    kb_paths = [p.strip() for p in paths.split(",") if p.strip()]
    r = httpx.post(f"{API}/v1/index", json={"kb_paths": kb_paths}, timeout=300)
    r.raise_for_status()
    return r.json()

def ask(prompt, mode, kb_paths, top_k, min_local, web_links, web_chars, web_iters,
        hybrid_w, max_tokens, temperature, thread_id, persist_memory, mem_turns):
    kb_list = [p.strip() for p in kb_paths.split(",") if p.strip()]
    payload = {
        "prompt": prompt, "mode": mode, "kb_paths": kb_list,
        "top_k": top_k, "min_local_score": min_local, "web_max_links": web_links,
        "web_max_chars": web_chars, "web_iterations": web_iters,
        "hybrid_local_weight": hybrid_w, "max_tokens": max_tokens,
        "temperature": temperature, "thread_id": thread_id or None,
        "persist_memory": persist_memory, "memory_max_turns": mem_turns
    }
    r = httpx.post(f"{API}/v1/query", json=payload, timeout=600)
    r.raise_for_status()
    out = r.json()
    return f"**Mode used:** {out['mode_used']}  |  **Best KB score:** {out['best_local_score']:.3f}\n\n" + out["answer"], out["thread_id"]

def start_research(topic, kb_paths, time_min, web_links, web_iters, web_chars, hybrid_w):
    kb_list = [p.strip() for p in kb_paths.split(",") if p.strip()]
    payload = {
        "topic": topic, "kb_paths": kb_list, "time_budget_min": time_min,
        "web_max_links": web_links, "web_iterations": web_iters,
        "max_chars_per_link": web_chars, "hybrid_local_weight": hybrid_w
    }
    r = httpx.post(f"{API}/v1/research", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["job_id"]

def poll(job_id):
    r = httpx.get(f"{API}/v1/research/{job_id}", timeout=30)
    r.raise_for_status()
    j = r.json()
    if j["status"] == "done":
        return f"### Research Report\n\n{j['result']}"
    elif j["status"] == "error":
        return f"**Error:** {j['result']}"
    return f"Status: {j['status']} (updated {int(j['updated'])})"

with gr.Blocks(title="ullu", css="footer{visibility:hidden}") as demo:
    gr.Markdown("# 🦉 ullu — Local+Web RAG with Ollama")
    with gr.Tab("Chat"):
        with gr.Row():
            prompt = gr.Textbox(label="Your question", lines=4)
        with gr.Row():
            mode = gr.Dropdown(choices=["auto","local","web","hybrid"], value="auto", label="Mode")
            kb_paths = gr.Textbox(label="KB folders (comma-separated absolute paths)", value="/ssd/kb/science_eg")
            top_k = gr.Slider(1, 10, value=6, step=1, label="Top-K (KB)")
            min_local = gr.Slider(0.0, 1.0, value=0.55, step=0.01, label="Min KB score for 'auto'")
        with gr.Accordion("Web / Hybrid controls", open=False):
            with gr.Row():
                web_links = gr.Slider(1, 50, value=8, step=1, label="Max web links")
                web_iters = gr.Slider(1, 5, value=2, step=1, label="Iterations (breadth)")
                web_chars = gr.Slider(2000, 80000, value=20000, step=1000, label="Max chars per link")
                hybrid_w = gr.Slider(0.0,1.0,value=0.7,step=0.05,label="Hybrid: weight for KB (0..1)")
        with gr.Accordion("LLM params & Memory", open=False):
            with gr.Row():
                max_tokens = gr.Slider(64, 2048, value=512, step=64, label="Max tokens")
                temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Temperature")
                thread_id = gr.Textbox(label="Thread ID (blank = auto)")
                persist_memory = gr.Checkbox(value=True, label="Persist memory")
                mem_turns = gr.Slider(10, 400, value=100, step=10, label="Max memory turns")
        run = gr.Button("Ask (⌘/Ctrl+Enter)")
        out = gr.Markdown()
        tid_out = gr.Textbox(label="Thread ID (persist this for the same chat)")
        run.click(ask, [prompt,mode,kb_paths,top_k,min_local,web_links,web_chars,web_iters,
                        hybrid_w,max_tokens,temperature,thread_id,persist_memory,mem_turns],
                  [out, tid_out])

        gr.Markdown("### Re/Index Knowledge Base")
        idx_paths = gr.Textbox(label="KB paths to (re)index", value="/ssd/kb/science_eg")
        idx_btn = gr.Button("Index / Reindex now")
        idx_out = gr.JSON()
        idx_btn.click(index_kb, [idx_paths], [idx_out])

    with gr.Tab("Deep Research"):
        topic = gr.Textbox(label="Topic / problem to investigate", lines=3)
        dr_kb = gr.Textbox(label="Optional KB folders", value="")
        time_min = gr.Slider(5, 120, value=20, step=5, label="Time budget (minutes)")
        dr_links = gr.Slider(5, 100, value=30, step=1, label="Max links")
        dr_iters = gr.Slider(1, 6, value=3, step=1, label="Iterations")
        dr_chars = gr.Slider(5000, 120000, value=40000, step=5000, label="Max chars per link")
        dr_weight = gr.Slider(0.0,1.0,value=0.5,step=0.05,label="Hybrid weight for KB")
        start = gr.Button("Start Deep Research")
        job = gr.Textbox(label="Job ID")
        start.click(start_research, [topic, dr_kb, time_min, dr_links, dr_iters, dr_chars, dr_weight], [job])
        poll_btn = gr.Button("Poll Status/Result")
        report = gr.Markdown()
        poll_btn.click(poll, [job], [report])

demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)

