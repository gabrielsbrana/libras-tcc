#!/usr/bin/env bash
# Prepara o ambiente da demonstração no WSL (Ubuntu). Rode uma vez:
#   bash setup_wsl.sh
set -euo pipefail

echo "== 1/4 Pacotes do sistema (pede a senha do sudo) =="
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev build-essential openjdk-8-jdk libhunspell-dev ffmpeg

export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
# O venv fica no disco do Linux; em /mnt/c a criação falha.
VENV="$HOME/libras-demo-venv"
# /tmp do WSL é pequeno para instalar o PyTorch.
export TMPDIR="$HOME/pip-tmp"; mkdir -p "$TMPDIR"

echo "== 2/4 Ambiente Python em $VENV =="
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip wheel Cython

echo "== 3/4 Bibliotecas (alguns minutos) =="
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers peft sentencepiece protobuf faster-whisper fastapi uvicorn python-multipart
# Regras: precisa de Java 8 e de NLTK anterior à 3.8.2.
pip install "nltk<3.8.2" vlibras-translate || echo "AVISO: vlibras-translate não instalou; a demo roda só com o LoRA."

echo "== 4/4 Baixando os modelos (PTT5-small e Whisper small) =="
python - <<'PY'
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
AutoTokenizer.from_pretrained("unicamp-dl/ptt5-small-portuguese-vocab")
AutoModelForSeq2SeqLM.from_pretrained("unicamp-dl/ptt5-small-portuguese-vocab")
from faster_whisper import WhisperModel
WhisperModel("small", device="cpu", compute_type="int8")
print("modelos em cache")
PY
echo
echo "Pronto. Para subir o servidor: ./run.sh"
