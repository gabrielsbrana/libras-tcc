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
57,14 contra 45,60) e em desambiguação de sentido (30,23 contra 28,52). O LoRA
degenera em frases longas: 20,1% das saídas repetem a mesma glosa três ou mais
vezes. Detalhes por categoria, domínio e comprimento em
[`results/RESULTADOS.md`](results/RESULTADOS.md).

## Estrutura

| Pasta | Conteúdo |
| --- | --- |
| `tcc/` | Texto do TCC em Word e PDF |
| `results/` | Métricas consolidadas, predições do LoRA no teste, histórico de treino e figuras |
| `src/` | Normalização e categorias (`text_utils.py`), avaliação (`avaliar_predicoes.py`) e figuras (`gerar_figuras.py`) |
| `notebooks/` | Treino do LoRA no Colab (com as saídas da execução) e avaliação no teste |
| `models/lora_adapter/` | Adaptador LoRA treinado (4,6 MB), sobre `unicamp-dl/ptt5-small-portuguese-vocab` |
| `demo/` | Prova de conceito: servidor local, página web e extensão do Chrome |
| `data/` | Como obter o corpus (não versionado) |
| `apresentacao/` | Link para os slides da defesa |

## Reproduzir

```bash
pip install -r requirements.txt

# Métricas do LoRA a partir das predições salvas (não precisa de GPU)
python src/avaliar_predicoes.py results/predictions_lora_test.csv --coluna glosa_pred_lora

# Gerar novas predições do LoRA: notebooks/avaliacao_lora_colab.ipynb (Colab, GPU T4)
# Figuras: precisa de data/processed/ (ver data/README.md)
python src/gerar_figuras.py
```

A prova de conceito tem instruções próprias em [`demo/README.md`](demo/README.md).

## O que não está aqui

O código original de preparação dos dados, da linha de base por regras, do
treino do seq2seq e da avaliação conjunta (`data_prep.py`, `baseline_regras.py`,
`seq2seq_model.py`, `evaluate.py`), o checkpoint do seq2seq e as predições
individuais de regras e seq2seq não foram preservados. Dessas duas abordagens
restam as métricas, registradas no TCC e em `results/RESULTADOS.md`. A
configuração do seq2seq está descrita na Seção 3.5 do texto.

## Referências principais

- Lima, M. A. et al. VLibrasBD: a Brazilian Portuguese–Brazilian Sign Language
  (Libras) bilingual text dataset designed to support neural machine
  translation. *Data in Brief*, v. 62, 111911, 2025.
- Hu, E. J. et al. LoRA: low-rank adaptation of large language models. ICLR, 2022.
- Carmo, D. et al. PTT5: pretraining and validating the T5 model on Brazilian
  Portuguese data. arXiv:2008.09144, 2020.
