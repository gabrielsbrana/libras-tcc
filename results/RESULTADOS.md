# Resultados

Conjunto de teste: 12.732 pares (7.230 do domínio geral, 5.502 do governamental).
BLEU e chrF calculados com SacreBLEU após `normalizar_glosa()` (`src/text_utils.py`)
aplicada à referência e à hipótese.

Fontes: seq2seq, saída do `evaluate.py` original (agosto de 2026; predições e modelo
não preservados). Regras, `predictions_regras.csv` (predições preservadas, mesma ordem do
teste). LoRA, `results/predictions_lora_test.csv`, geradas em 16/11/2026. As tabelas por
fenômeno e por comprimento são geradas por `tcc/latex/gerar_figuras_artigo.py`.

## Global

| Modelo | BLEU | chrF |
| --- | --- | --- |
| Regras (`vlibras-translate`) | 33,98 | 69,71 |
| Seq2seq do zero (40.000 pares) | 28,08 | 50,67 |
| PTT5-small + LoRA (101.850 pares) | 9,19 | 24,05 |

## BLEU por categoria linguística

Cada linha usa só os pares cuja glosa de referência contém o fenômeno. Métrica: média do
BLEU por sentença (mesmo cálculo do `evaluate.py` original para regras e seq2seq; para o LoRA,
reproduzível com `src/avaliar_predicoes.py`). Versões anteriores deste arquivo traziam, na coluna
do LoRA, o BLEU de conjunto (8,69; 6,50; ...), que não é comparável com as outras colunas.

| Categoria | Pares | Regras | Seq2seq | LoRA |
| --- | --- | --- | --- | --- |
| Marcador de ponto final | 9.310 | 39,41 | 38,37 | 18,78 |
| Desambiguação de sentido | 7.197 | 28,52 | 30,23 | 11,55 |
| Sinal composto | 2.562 | 35,47 | 30,18 | 15,70 |
| Numeral | 2.005 | 27,18 | 18,48 | 9,72 |
| Negação fundida ao verbo | 1.316 | 38,85 | 31,24 | 19,02 |
| Interrogação | 848 | 45,60 | 57,14 | 31,54 |
| Verbo direcional | 700 | 20,27 | 18,57 | 11,18 |
| Negação isolada | 379 | 30,86 | 29,74 | 16,75 |
| Exclamação | 100 | 48,55 | 47,06 | 32,76 |

A categoria Numeral inclui os 700 pares com verbo direcional, porque o padrão
`\d` também casa com os marcadores de pessoa (`1S_`, `2S_`); 615 dos 2.005 pares
não têm nenhum outro número.

## Por domínio

| Domínio | Pares | BLEU regras | chrF regras | BLEU seq2seq | chrF seq2seq | BLEU LoRA | chrF LoRA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Geral | 7.230 | 36,99 | 68,61 | 26,86 | 44,46 | 11,58 | 25,11 |
| Governamental | 5.502 | 30,49 | 70,64 | 29,26 | 55,81 | 6,68 | 23,16 |

## Regras por comprimento e domínio

| Palavras | BLEU geral | chrF geral | BLEU gov. | chrF gov. | BLEU total | chrF total |
| --- | --- | --- | --- | --- | --- | --- |
| 1 a 5 | 50,65 | 71,54 | 39,90 | 74,40 | 49,39 | 72,25 |
| 6 a 10 | 42,63 | 66,99 | 31,65 | 70,51 | 37,92 | 69,13 |
| 11 a 20 | 34,22 | 68,69 | 32,33 | 71,20 | 33,06 | 70,39 |
| 21 a 40 | 29,87 | 67,91 | 29,35 | 70,01 | 29,68 | 69,08 |
| 41 ou mais | 26,85 | 67,61 | 27,32 | 69,75 | 27,22 | 68,62 |

Pós-processamento das regras (remover SER, ESTAR e o "1" quando o português não tem o
algarismo): BLEU 35,57 / chrF 70,63 (geral 39,46; governamental 31,31). Estimativa
otimista: as palavras foram escolhidas olhando o teste.

## LoRA por comprimento da frase em português

| Palavras | Pares | BLEU | chrF | Saídas com repetição |
| --- | --- | --- | --- | --- |
| 1 a 5 | 4.828 | 23,16 | 40,05 | 0,9% |
| 6 a 10 | 2.918 | 16,31 | 32,70 | 12,7% |
| 11 a 20 | 2.411 | 7,89 | 24,64 | 39,8% |
| 21 a 40 | 2.019 | 2,88 | 18,43 | 46,2% |
| 41 ou mais | 556 | 0,86 | 12,82 | 44,6% |

Repetição: a mesma glosa três ou mais vezes seguidas. No total, 2.555 saídas
(20,1%) e nenhuma vazia; 107 (0,84%) idênticas à referência.

## Treinamento

| | Seq2seq | PTT5-small + LoRA |
| --- | --- | --- |
| Ambiente | CPU (WSL) | GPU T4 (Google Colab) |
| Pares de treino | 40.000 | 101.850 |
| Arquitetura | LSTM bidirecional + atenção de Bahdanau, embedding 256, oculto 256, dropout 0,3, BPE 8.000 | `ptt5-small-portuguese-vocab`, r = 16, alfa = 32, dropout 0,05, módulos q e v |
| Parâmetros treinados | todos | 589.824 de 61.096.448 (0,97%) |
| Épocas | 10 (parada antecipada; melhor na 7ª, perda de validação 3,22) | 15 (limite; perda de validação 2,21 → 1,70) |
| Tempo | cerca de 40 min por época | 9.189 s |

O histórico completo do LoRA está em `lora_training_history.csv`.
