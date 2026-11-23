"""
Calcula BLEU e chrF de um arquivo de predições, nos mesmos recortes do TCC:
global, por categoria linguística, por domínio e por comprimento da frase.

Uso:
    python src/avaliar_predicoes.py results/predictions_lora_test.csv --coluna glosa_pred_lora

O CSV precisa das colunas pt, glosa (referência), is_government_source e a
coluna de predição indicada. Com results/predictions_lora_test.csv, reproduz
os números do PTT5-small + LoRA no texto (BLEU 9,19 / chrF 24,05).
"""

import argparse
import os
import sys

import pandas as pd
import sacrebleu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_utils import CATEGORIAS, ROTULOS, categorias_presentes, normalizar_glosa, tem_repeticao  # noqa: E402

FAIXAS = [(1, 5), (6, 10), (11, 20), (21, 40), (41, 10_000)]


def metricas(refs, hips):
    return (round(sacrebleu.corpus_bleu(hips, [refs]).score, 2),
            round(sacrebleu.corpus_chrf(hips, [refs]).score, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--coluna", default="glosa_pred_lora")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    refs = [normalizar_glosa(x) for x in df["glosa"].astype(str)]
    hips = [normalizar_glosa(x) for x in df[args.coluna].fillna("").astype(str)]

    b, c = metricas(refs, hips)
    print(f"GLOBAL  n={len(refs)}  BLEU={b}  chrF={c}")

    # No TCC (Tabela 6), o BLEU por categoria é a MÉDIA DO BLEU POR SENTENÇA, mesmo critério
    # do evaluate.py original usado para regras e seq2seq. O BLEU de conjunto aparece só como
    # referência: as duas medidas não são comparáveis entre si (ver Seção 2.6 do TCC).
    print("\nPor categoria (referência contém o fenômeno):")
    marcas = [categorias_presentes(r) for r in refs]
    for nome in CATEGORIAS:
        idx = [i for i, m in enumerate(marcas) if m[nome]]
        b, c = metricas([refs[i] for i in idx], [hips[i] for i in idx])
        media = sum(sacrebleu.sentence_bleu(hips[i], [refs[i]]).score for i in idx) / len(idx)
        print(f"  {ROTULOS[nome]:<28} n={len(idx):>5}  BLEU médio por sentença={media:>6.2f}"
              f"  (BLEU de conjunto={b:>6}, chrF={c:>6})")

    print("\nPor domínio:")
    gov = df["is_government_source"].astype(str).str.lower().eq("true").tolist()
    for rotulo, alvo in (("Geral", False), ("Governamental", True)):
        idx = [i for i, g in enumerate(gov) if g == alvo]
        b, c = metricas([refs[i] for i in idx], [hips[i] for i in idx])
        print(f"  {rotulo:<14} n={len(idx):>5}  BLEU={b:>6}  chrF={c:>6}")

    print("\nPor comprimento da frase em português (palavras):")
    palavras = df["pt"].astype(str).str.split().str.len().tolist()
    for lo, hi in FAIXAS:
        idx = [i for i, n in enumerate(palavras) if lo <= n <= hi]
        b, c = metricas([refs[i] for i in idx], [hips[i] for i in idx])
        rep = sum(tem_repeticao(hips[i]) for i in idx) / len(idx) * 100
        faixa = f"{lo} ou mais" if hi >= 10_000 else f"{lo} a {hi}"
        print(f"  {faixa:<12} n={len(idx):>5}  BLEU={b:>6}  chrF={c:>6}  repetição={rep:.1f}%")

    exatas = sum(r == h for r, h in zip(refs, hips))
    print(f"\nSaídas idênticas à referência: {exatas} ({exatas / len(refs) * 100:.2f}%)")


if __name__ == "__main__":
    main()
