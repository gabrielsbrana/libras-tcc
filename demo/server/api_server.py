"""
Servidor local da prova de conceito do TCC: português -> glosa em Libras.

Recebe texto ou áudio (capturado pela extensão do Chrome), transcreve com
Whisper e traduz cada frase pelos modelos disponíveis:

    regras  -> pacote vlibras-translate (precisa de Java 8)
    lora    -> PTT5-small + adaptador LoRA em models/lora_adapter

Cada modelo é opcional. Se um não carregar, o servidor continua de pé e
informa o motivo em /health.

Uso (a partir da pasta libras-demo):
    ./run.sh
Teste sem áudio:
    curl -X POST http://127.0.0.1:8000/translate \
         -H "Content-Type: application/json" \
         -d '{"text": "Você sabe dirigir?"}'
"""

import os
import re
import tempfile
import time

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Procura o adaptador em demo/models/lora_adapter ou, no repositório, em models/lora_adapter.
_CANDIDATOS = [os.path.join(RAIZ, "models", "lora_adapter"),
               os.path.join(os.path.dirname(RAIZ), "models", "lora_adapter")]
ADAPTER_DIR = os.environ.get("ADAPTER_DIR") or next(
    (c for c in _CANDIDATOS if os.path.exists(os.path.join(c, "adapter_model.safetensors"))), _CANDIDATOS[0])
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

MODEL_NAME = "unicamp-dl/ptt5-small-portuguese-vocab"
TASK_PREFIX = "traduzir para glosa em libras: "
MAX_LEN = 64
WHISPER_SIZE = os.environ.get("WHISPER_SIZE", "small")

ROTULOS = {"regras": "Regras (vlibras-translate)", "lora": "PTT5-small + LoRA"}
MODELOS = list(ROTULOS)

app = FastAPI(title="Português -> glosa em Libras (demonstração local)")
# Uso local apenas: aceita chamadas da extensão (chrome-extension://...).
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_cache = {}
_erros = {}


def normalizar_glosa(texto):
    s = str(texto or "").strip().upper()
    s = re.sub(r"\s*\[\s*", " [", s)
    s = re.sub(r"\s*\]\s*", "] ", s)
    return re.sub(r"\s+", " ", s).strip()


# A ponte Java do vlibras-translate só aceita Latin-1.
_TROCAS = {"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-",
           "—": "-", "…": "...", " ": " "}


def sanitizar(texto):
    for a, b in _TROCAS.items():
        texto = texto.replace(a, b)
    return texto.encode("latin-1", "ignore").decode("latin-1")


def _carregar_regras():
    from vlibras_translate import translation

    tradutor = translation.Translation()
    return lambda frase: normalizar_glosa(tradutor.rule_translation(sanitizar(frase)))


def _carregar_lora():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if not os.path.exists(os.path.join(ADAPTER_DIR, "adapter_model.safetensors")):
        raise FileNotFoundError(f"adaptador não encontrado em {ADAPTER_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    modelo = PeftModel.from_pretrained(AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME), ADAPTER_DIR).eval()

    def traduzir(frase):
        entrada = tokenizer(TASK_PREFIX + frase, return_tensors="pt", truncation=True, max_length=MAX_LEN)
        with torch.no_grad():
            # mesma decodificação usada na avaliação do TCC
            saida = modelo.generate(**entrada, max_length=MAX_LEN, num_beams=4)
        return normalizar_glosa(tokenizer.decode(saida[0], skip_special_tokens=True))

    return traduzir


_CARREGADORES = {"regras": _carregar_regras, "lora": _carregar_lora}


def _obter(nome):
    if nome in _cache:
        return _cache[nome]
    if nome in _erros:
        return None
    try:
        t0 = time.time()
        _cache[nome] = _CARREGADORES[nome]()
        print(f"[ok] modelo '{nome}' carregado em {time.time() - t0:.1f}s")
        return _cache[nome]
    except Exception as e:  # dependência ou arquivo ausente
        _erros[nome] = f"{type(e).__name__}: {e}"
        print(f"[aviso] modelo '{nome}' indisponível: {_erros[nome]}")
        return None


def _obter_whisper():
    if "whisper" not in _cache:
        from faster_whisper import WhisperModel

        print(f"Carregando Whisper ({WHISPER_SIZE})...")
        _cache["whisper"] = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
    return _cache["whisper"]


def _carregar_audio(caminho):
    """Decodifica qualquer áudio/vídeo com o ffmpeg para 16 kHz mono.

    Feito aqui, e não pelo faster-whisper, porque versões recentes da
    biblioteca PyAV quebram a leitura interna dele.
    """
    import subprocess

    import numpy as np

    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", caminho,
           "-f", "f32le", "-ac", "1", "-ar", "16000", "-"]
    bruto = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(bruto, dtype=np.float32)


def dividir_frases(texto):
    partes = re.split(r"(?<=[.!?])\s+", texto.strip())
    return [p.strip() for p in partes if p.strip()]


def traduzir_frase(frase):
    item = {"pt": frase}
    for nome in MODELOS:
        fn = _obter(nome)
        if fn is None:
            item[nome] = None
            continue
        try:
            item[nome] = fn(frase)
        except Exception as e:
            item[nome] = None
            print(f"[aviso] '{nome}' falhou em '{frase[:40]}': {e}")
    return item


def resposta(frases, transcricao=None, tempo=None):
    return {
        "results": [traduzir_frase(f) for f in frases],
        "modelos": [{"id": m, "rotulo": ROTULOS[m], "disponivel": m in _cache} for m in MODELOS],
        "indisponiveis": dict(_erros),
        "transcricao": transcricao,
        "segundos": tempo,
    }


class TranslateRequest(BaseModel):
    text: str


# Endpoints async de propósito: rodam todos na mesma thread, o que evita
# problemas da ponte Java (regras) quando chamada de threads diferentes.

@app.get("/")
async def pagina():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
async def health():
    for nome in MODELOS:
        _obter(nome)
    return {"status": "ok", "disponiveis": [m for m in MODELOS if m in _cache], "indisponiveis": dict(_erros)}


@app.post("/translate")
async def translate(req: TranslateRequest):
    t0 = time.time()
    return _com_tempo(dividir_frases(req.text), None, t0)


def _com_tempo(frases, transcricao, t0):
    r = resposta(frases, transcricao)
    r["segundos"] = round(time.time() - t0, 1)
    return r


@app.post("/transcribe_and_translate")
async def transcribe_and_translate(file: UploadFile = File(...)):
    t0 = time.time()
    conteudo = await file.read()
    sufixo = os.path.splitext(file.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=sufixo, delete=False) as tmp:
        tmp.write(conteudo)
        caminho = tmp.name
    try:
        segmentos, _ = _obter_whisper().transcribe(_carregar_audio(caminho), language="pt", vad_filter=True)
        texto = " ".join(s.text.strip() for s in segmentos).strip()
    finally:
        os.remove(caminho)
    return _com_tempo(dividir_frases(texto), texto, t0)
