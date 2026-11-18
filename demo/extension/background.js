// Service worker: cria o documento offscreen (único contexto do Manifest V3
// que pode gravar o áudio da aba) e guarda o estado para o popup.
async function garantirOffscreen() {
  if (await chrome.offscreen.hasDocument()) return;
  await chrome.offscreen.createDocument({
    url: "offscreen.html",
    reasons: ["USER_MEDIA"],
    justification: "Gravar o áudio da aba para transcrição e tradução em glosa.",
  });
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.tipo === "iniciar") {
    garantirOffscreen()
      .then(() => chrome.runtime.sendMessage({ tipo: "gravar", streamId: msg.streamId, segundos: msg.segundos }))
      .catch((e) => chrome.storage.session.set({ estado: { estado: "erro", mensagem: e.message } }));
  } else if (msg.tipo === "estado") {
    chrome.storage.session.set({ estado: msg.estado });
  }
});
