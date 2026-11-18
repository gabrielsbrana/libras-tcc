const SERVIDOR = "http://127.0.0.1:8000";
let gravando = false;

function avisar(estado) {
  return chrome.runtime.sendMessage({ tipo: "estado", estado }).catch(() => {});
}

async function gravar(streamId, segundos) {
  if (gravando) return;
  gravando = true;
  let stream, contexto;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { mandatory: { chromeMediaSource: "tab", chromeMediaSourceId: streamId } },
    });
    // A captura silencia a aba; devolve o som para os alto-falantes.
    contexto = new AudioContext();
    contexto.createMediaStreamSource(stream).connect(contexto.destination);

    const partes = [];
    const gravador = new MediaRecorder(stream, { mimeType: "audio/webm" });
    gravador.ondataavailable = (ev) => { if (ev.data.size) partes.push(ev.data); };
    const fim = new Promise((resolve) => { gravador.onstop = resolve; });
    gravador.start();
    await avisar({ estado: "gravando", segundos });
    await new Promise((r) => setTimeout(r, segundos * 1000));
    gravador.stop();
    await fim;

    await avisar({ estado: "enviando" });
    const corpo = new FormData();
    corpo.append("file", new Blob(partes, { type: "audio/webm" }), "trecho.webm");
    const r = await fetch(SERVIDOR + "/transcribe_and_translate", { method: "POST", body: corpo });
    if (!r.ok) throw new Error("servidor respondeu HTTP " + r.status);
    await avisar({ estado: "pronto", dados: await r.json() });
  } catch (e) {
    await avisar({ estado: "erro", mensagem: e.message });
  } finally {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if (contexto) contexto.close();
    gravando = false;
  }
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.tipo === "gravar") gravar(msg.streamId, msg.segundos);
});
