(function(){
  const $ = (sel) => document.querySelector(sel);
  const statsDocs = $("#docs");
  const statsChunks = $("#chunks");
  const indexStatus = $("#indexStatus");
  const btnIndex = $("#btnIndex");
  const btnReset = $("#btnReset");
  const btnAsk = $("#btnAsk");
  const files = $("#files");
  const query = $("#query");
  const rag = $("#rag");
  const topk = $("#topk");
  const model = $("#model");
  const temp = $("#temp");
  const maxtok = $("#maxtok");
  const system = $("#system");
  const answer = $("#answer");
  const citations = $("#citations");

  // Compute API base from host user is hitting
  const API_BASE = `${location.protocol}//${location.hostname}:9150`;

  async function refreshStats(){
    try {
      const res = await fetch(`${API_BASE}/health`);
      const data = await res.json();
      statsDocs.textContent = data.docs ?? 0;
      statsChunks.textContent = data.chunks ?? 0;
    } catch(e){
      console.error(e);
    }
  }

  btnIndex.addEventListener("click", async () => {
    if (!files.files.length){
      indexStatus.textContent = "Select one or more files first.";
      return;
    }
    indexStatus.textContent = "Indexing...";
    const fd = new FormData();
    for (const f of files.files) fd.append("files", f, f.name);
    try {
      const res = await fetch(`${API_BASE}/rag/index`, { method: "POST", body: fd });
      const data = await res.json();
      indexStatus.textContent = `Indexed files=${data.indexed_files} chunks=${data.indexed_chunks}`;
      await refreshStats();
    } catch(e){
      indexStatus.textContent = "Index failed. See console.";
      console.error(e);
    }
  });

  btnReset.addEventListener("click", async () => {
    if (!confirm("Reset the index?")) return;
    indexStatus.textContent = "Resetting...";
    try {
      await fetch(`${API_BASE}/rag/reset`, { method: "POST" });
      indexStatus.textContent = "Index reset.";
      await refreshStats();
    } catch(e){
      indexStatus.textContent = "Reset failed.";
      console.error(e);
    }
  });

  btnAsk.addEventListener("click", async () => {
    const body = {
      query: query.value || "",
      rag: rag.value === "true",
      top_k: parseInt(topk.value || "3"),
      model: model.value || "llama3",
      temperature: parseFloat(temp.value || "0.2"),
      max_tokens: parseInt(maxtok.value || "512"),
      system_prompt: system.value || null
    };
    if (!body.query.trim()){
      answer.textContent = "Enter a query.";
      return;
    }
    answer.textContent = "Thinking...";
    citations.innerHTML = "";
    try {
      const res = await fetch(`${API_BASE}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      const data = await res.json();
      answer.textContent = data.answer || "(no answer)";
      (data.citations || []).forEach((c) => {
        const el = document.createElement("div");
        el.className = "cite";
        el.innerHTML = `<div class="src">${c.source} <span class="score">score=${(c.score||0).toFixed(3)}</span></div>
                        <div class="preview">${(c.preview||"").replace(/[<>&]/g, s=>({"<":"&lt;",">":"&gt;","&":"&amp;"}[s]))}</div>`;
        citations.appendChild(el);
      });
    } catch(e){
      answer.textContent = "Request failed. See console.";
      console.error(e);
    }
  });

  refreshStats();
})();

