# Tradução automática de português para glosa em Libras

TCC do MBA em Ciência de Dados da Universidade de Fortaleza (UNIFOR), 2026.
Autor: Gabriel Sbrana. Orientador: Prof. Dr. Jorge Luiz Bezerra de Araujo.

O trabalho compara três abordagens para traduzir português escrito em glosa de
Libras, a representação textual usada antes da sinalização por avatar:

1. **Regras linguísticas**: pacote `vlibras-translate` (linha de base).
2. **Seq2seq treinado do zero**: LSTM bidirecional com atenção de Bahdanau.
3. **PTT5-small + LoRA**: modelo pré-treinado em português, ajustado com LoRA.

## Resultados (12.732 pares de teste do VLibrasBD)

| Modelo | Pares de treino | BLEU | chrF |
| --- | --- | --- | --- |
| Regras (`vlibras-translate`) | sem treino | 33,98 | 69,71 |
| Seq2seq do zero | 40.000 | 28,08 | 50,67 |
| PTT5-small + LoRA | 101.850 | 9,19 | 24,05 |

As regras lideram no agregado. O seq2seq já as supera em interrogação (BLEU
médio por sentença de 57,14 contra 45,60) e em desambiguação de sentido (30,23
contra 28,52). O LoRA degenera em frases longas: 20,1% das saídas repetem a mesma
glosa três ou mais vezes. Detalhes por categoria, domínio e comprimento em
[`results/RESULTADOS.md`](results/RESULTADOS.md).

BLEU global e por domínio são de conjunto (corpus BLEU do SacreBLEU). BLEU por
categoria é a média do BLEU por sentença. As duas medidas não são comparáveis
entre si; a Seção 2.6 do TCC explica a diferença.

## Estrutura

| Pasta | Conteúdo |
| --- | --- |
| `tcc/` | Texto do TCC em Word e PDF |
| `results/` | Métricas consolidadas, predições das três abordagens no teste, histórico de treino, figuras e saídas originais do `evaluate.py` (`originais/`) |
| `src/` | Normalização e categorias (`text_utils.py`), avaliação (`avaliar_predicoes.py`) e figuras (`gerar_figuras.py`) |
| `notebooks/` | Treino do LoRA no Colab (com as saídas da execução) e avaliação no teste |
| `models/lora_adapter/` | Adaptador LoRA treinado (4,6 MB), sobre `unicamp-dl/ptt5-small-portuguese-vocab` |
| `demo/` | Prova de conceito: servidor local, página web e extensão do Chrome |
| `data/` | Como obter o corpus (não versionado) |
| `apresentacao/` | Link para os slides da defesa |

## Reproduzir

Requisitos: Python 3.10 ou superior. A linha de base por regras exige também
Linux (ou WSL), Java 8 e `nltk<3.8.2`; detalhes em [`demo/README.md`](demo/README.md).

```bash
git clone https://github.com/gabrielsbrana/libras-tcc.git
cd libras-tcc
pip install -r requirements.txt
```

**1. Todas as métricas a partir das predições salvas** (sem GPU, cerca de 1 minuto
cada). As predições das três abordagens sobre o conjunto de teste estão em `results/`
e reproduzem os números do TCC (global, por categoria, por domínio e por comprimento):

```bash
python src/avaliar_predicoes.py results/predictions_completas.csv --coluna glosa_pred_regras   # BLEU 33.98, chrF 69.71
python src/avaliar_predicoes.py results/predictions_completas.csv --coluna glosa_pred_seq2seq  # BLEU 28.08, chrF 50.67
python src/avaliar_predicoes.py results/predictions_lora_test.csv --coluna glosa_pred_lora     # BLEU 9.19,  chrF 24.05
```

**2. Gerar de novo as predições do LoRA** com o adaptador de `models/lora_adapter/`:
abra `notebooks/avaliacao_lora_colab.ipynb` no Google Colab (GPU T4) e envie
`results/predictions_lora_test.csv`, que contém as frases e referências do teste.

**3. Treinar o LoRA do zero**: `notebooks/finetune_lora_colab.ipynb` (Colab, GPU T4,
cerca de 2 h 33 min). Precisa da divisão em `data/processed/` (ver
[`data/README.md`](data/README.md)).

**4. Figuras**: `python src/gerar_figuras.py` (precisa de `data/processed/`).

**5. Prova de conceito**: [`demo/README.md`](demo/README.md).

## O que não está aqui

O código original de preparação dos dados, da linha de base por regras, do
treino do seq2seq e da avaliação conjunta (`data_prep.py`, `baseline_regras.py`,
`seq2seq_model.py`, `evaluate.py`) e o checkpoint do seq2seq não foram
preservados. As predições dessas abordagens foram preservadas e reproduzem todas
as métricas. A configuração do seq2seq está descrita na Seção 3.5 do
texto, e a da linha de base, na Seção 3.4.

## Licenças

- Código e documentação deste repositório: MIT ([`LICENSE`](LICENSE)).
- Adaptador LoRA: derivado de `unicamp-dl/ptt5-small-portuguese-vocab`; vale também
  a licença do modelo base.
- Corpus VLibrasBD: CC BY-SA 4.0, distribuído pelos autores no Mendeley Data. Não
  está versionado aqui; os arquivos de predições em `results/` contêm frases e glosas do
  conjunto de teste e seguem a mesma licença.
- `vlibras-translate`: LGPLv3, instalado do PyPI.

## Como citar

```
G. Sbrana, "libras-tcc: tradução automática de português para glosa em Libras,"
GitHub, versão v1.1-tcc, 2026. [Online]. Disponível em:
https://github.com/gabrielsbrana/libras-tcc
```

Os metadados também estão em [`CITATION.cff`](CITATION.cff).

## Referências principais

- [1] M. A. Lima et al., "VLibrasBD: a Brazilian Portuguese–Brazilian Sign
  Language (Libras) bilingual text dataset designed to support neural machine
  translation," *Data in Brief*, vol. 62, art. 111911, 2025.
- [2] M. Lima et al., "pt-br2libras-gloss," Mendeley Data, v1, 2025.
  doi: 10.17632/ryj88ckjww.1.
- [3] E. J. Hu et al., "LoRA: low-rank adaptation of large language models," em
  *Proc. ICLR*, 2022.
- [4] D. Carmo et al., "PTT5: pretraining and validating the T5 model on
  Brazilian Portuguese data," arXiv:2008.09144, 2020.
