const SERVIDOR = "http://127.0.0.1:8000";
const $ = (id) => document.getElementById(id);

function mostrarEstado(e) {
  if (!e) return;
  const textos = {
    gravando: "Gravando o áudio da aba (" + (e.segundos || "") + " s)… deixe o vídeo tocando.",
    enviando: "Transcrevendo e traduzindo… a primeira vez demora mais.",
    pronto: e.dados ? e.dados.results.length + " frase(s) em " + e.dados.segundos + " s." : "",
    erro: "Erro: " + (e.mensagem || "falha desconhecida"),
  };
  $("status").textContent = textos[e.estado] || "";
  $("capturar").disabled = e.estado === "gravando" || e.estado === "enviando";
  if (e.estado === "pronto" && e.dados) render(e.dados);
}

function render(dados) {
  const saida = $("saida");
  saida.textContent = "";
  if (!dados.results.length) { $("status").textContent = "Nenhuma fala reconhecida no trecho."; return; }
  for (const item of dados.results) {
    const div = document.createElement("div"); div.className = "item";
    const pt = document.createElement("div"); pt.className = "pt"; pt.textContent = item.pt; div.appendChild(pt);
    for (const m of dados.modelos) {
      const rot = document.createElement("div"); rot.className = "rot"; rot.textContent = m.rotulo;
      const g = document.createElement("div"); g.className = "glosa";
      if (item[m.id]) { g.textContent = item[m.id]; }
      else { g.textContent = "modelo indisponível"; g.classList.add("na"); }
      div.append(rot, g);
    }
    saida.appendChild(div);
  }
}

$("capturar").addEventListener("click", async () => {
  try {
    const [aba] = await chrome.tabs.query({ active: true, currentWindow: true });
    const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: aba.id });
    const segundos = Number($("duracao").value);
    await chrome.storage.session.set({ estado: { estado: "gravando", segundos } });
    chrome.runtime.sendMessage({ tipo: "iniciar", streamId, segundos }).catch(() => {});
  } catch (e) {
    mostrarEstado({ estado: "erro", mensagem: e.message });
  }
});

async function traduzirTexto() {
  const texto = $("texto").value.trim();
  if (!texto) return;
  $("status").textContent = "Traduzindo…";
  try {
    const r = await fetch(SERVIDOR + "/translate", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: texto }),
    });
    if (!r.ok) throw new Error("HTTP " + r.status);
    await chrome.storage.session.set({ estado: { estado: "pronto", dados: await r.json() } });
  } catch (e) {
    mostrarEstado({ estado: "erro", mensagem: "servidor local não respondeu (" + e.message + ")" });
  }
}
$("traduzir").addEventListener("click", traduzirTexto);
$("texto").addEventListener("keydown", (ev) => { if (ev.key === "Enter") traduzirTexto(); });

chrome.storage.session.onChanged.addListener((mudancas) => {
  if (mudancas.estado) mostrarEstado(mudancas.estado.newValue);
});
chrome.storage.session.get("estado").then((v) => mostrarEstado(v.estado));

fetch(SERVIDOR + "/health").then((r) => r.json()).then((h) => {
  const falhas = Object.keys(h.indisponiveis);
  $("servidor").textContent = "Servidor pronto. Modelos: " + (h.disponiveis.join(", ") || "nenhum") +
    (falhas.length ? " | indisponíveis: " + falhas.join(", ") : "");
}).catch(() => {
  $("servidor").textContent = "Servidor local não encontrado em 127.0.0.1:8000. Rode ./run.sh no WSL.";
});
