# Demonstração: português → glosa em Libras

Prova de conceito do TCC. Um servidor local traduz frases em português para glosa
pelas abordagens disponíveis, e uma extensão do Chrome captura o áudio de um vídeo
(por exemplo, no YouTube), transcreve com Whisper e mostra as glosas lado a lado.

| Modelo | De onde vem | Observação |
| --- | --- | --- |
| Regras | pacote `vlibras-translate` | precisa de Java 8 (o `setup_wsl.sh` instala) |
| PTT5-small + LoRA | `../models/lora_adapter/` (na raiz do repositório) | mesmo adaptador avaliado no TCC |

O seq2seq não está na demonstração: o arquivo do modelo treinado não foi preservado.

## 1. Instalar (uma vez, no WSL/Ubuntu)

Clone o repositório dentro do Linux (por exemplo em `~/libras-tcc`) e rode:

```bash
cd ~/libras-tcc/demo
bash setup_wsl.sh
```

Leva alguns minutos e baixa cerca de 1 GB (PyTorch, PTT5-small e Whisper small).

## 2. Subir o servidor

```bash
./run.sh
```

Abra http://127.0.0.1:8000 no navegador. A página mostra quais modelos carregaram e
permite traduzir um texto digitado ou um arquivo de áudio/vídeo. Ela é o plano B da
defesa, caso a captura pela extensão falhe.

## 3. Instalar a extensão

1. No Chrome, abra `chrome://extensions`.
2. Ative "Modo do desenvolvedor".
3. Clique em "Carregar sem compactação" e selecione a pasta `extension`
   (no Windows: `\\wsl$\Ubuntu\home\SEU_USUARIO\libras-tcc\demo\extension`).

## 4. Usar

1. Abra um vídeo em português e dê play.
2. Clique no ícone da extensão e em "Capturar áudio da aba e traduzir".
3. Aguarde a gravação (10 a 30 s) e a tradução. O resultado fica guardado:
   se o popup fechar, basta abri-lo de novo.

A caixa de texto do popup traduz uma frase digitada, sem áudio.

## Problemas comuns

- **"Servidor local não encontrado"**: o `./run.sh` não está rodando.
- **Regras "indisponível"**: veja o motivo em http://127.0.0.1:8000/health. Em geral é o
  Java: confirme com `/usr/lib/jvm/java-8-openjdk-amd64/bin/java -version`.
- **Primeira tradução lenta**: os modelos são carregados na primeira chamada. Antes da
  defesa, abra a página e traduza uma frase para aquecer.
- **Transcrição lenta**: rode `WHISPER_SIZE=base ./run.sh` (menos preciso, mais rápido).
