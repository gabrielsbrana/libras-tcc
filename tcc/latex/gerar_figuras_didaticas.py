"""Figuras explicativas do artigo (atenção, LoRA, arquitetura da prova de conceito) e figura da
análise de erros, desenhadas no tamanho final de impressão (largura útil de 6,1 pol.), para que
os textos fiquem entre 8 e 10 pt no PDF.
Uso: python gerar_figuras_didaticas.py <predictions_regras.csv> <predictions_lora_test.csv> <saida/>"""
import sys, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

p_reg, p_lora, OUT = sys.argv[1:4]
G, B, O = '#8a8f98', '#2a6fb0', '#d98324'
INK, NEU, ESC, REF = '#222222', '#5b6472', '#3d4652', '#111111'
LB, LO = '#e8f0f9', '#fbeedf'
plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 240, 'savefig.bbox': 'tight',
                     'savefig.pad_inches': 0.03})
br = lambda v, d=1: f'{v:.{d}f}'.replace('.', ',')


def tela(w, h):
    """Eixos em décimos de polegada: 1 unidade = 0,1 pol. no PDF."""
    fig = plt.figure(figsize=(w, h)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w * 10); ax.set_ylim(0, h * 10); ax.axis('off'); return fig, ax


def caixa(ax, x, y, w, h, txt='', ec=NEU, fc='white', fs=9, bold=False, color=INK, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.4', fc=fc, ec=ec, lw=lw))
    if txt:
        ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs, color=color,
                weight='bold' if bold else 'normal', linespacing=1.2)


def seta(ax, a, b, c=INK, lw=1.1):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=9, color=c, lw=lw, shrinkA=0, shrinkB=0))


# ---- atenção (seq2seq, azul) ---------------------------------------------
fig, ax = tela(6.1, 2.9)
pt = ['Eu', 'não', 'vi', 'ninguém', '.']; gl = ['EU', 'NÃO_VER', 'NINGUÉM', '[PONTO]']
ex = [0.3 + i * 5.85 for i in range(5)]; dx = [32 + i * 7.3 for i in range(4)]
for i, w in enumerate(pt):
    caixa(ax, ex[i], 3.2, 5.4, 3.4, w, fs=8.5)
    caixa(ax, ex[i], 10, 5.4, 3.6, f'h{i+1}', ec=B, fc=LB, fs=9.5)
    seta(ax, (ex[i] + 2.7, 6.7), (ex[i] + 2.7, 9.9), c=NEU)
for i, w in enumerate(gl):
    caixa(ax, dx[i], 10, 6.8, 3.6, f's{i+1}', ec=B, fc=LB, fs=9.5)
    caixa(ax, dx[i], 3.2, 6.8, 3.4, w, fs=8.5)
    seta(ax, (dx[i] + 3.4, 9.9), (dx[i] + 3.4, 6.7), c=NEU)
ax.text(14.6, 0.6, 'Codificador (LSTM bidirecional)', ha='center', fontsize=9, color=B, weight='bold')
ax.text(46.3, 0.6, 'Decodificador (LSTM)', ha='center', fontsize=9, color=B, weight='bold')
caixa(ax, 6.5, 22.5, 16, 5, 'atenção: média\nponderada de h1…h5', ec=B, fs=9, lw=1.6)
pesos = [0.05, 0.42, 0.38, 0.10, 0.05]
for i, p in enumerate(pesos):
    xc = ex[i] + 2.7
    ax.plot([xc, 14.5], [13.7, 22.4], color=B, lw=0.5 + 6 * p, alpha=0.75, solid_capstyle='round')
    ax.text(xc + (-2.6 if i < 2 else 0.5), 15.0, br(p, 2), fontsize=8, color=B)
seta(ax, (22.6, 25), (dx[1] + 3.4, 13.8), c=B, lw=1.4)
ax.text(29.5, 17.2, 'contexto para\ngerar s2', fontsize=8.5, color=B)
ax.text(60.5, 26.8, 'pesos ilustrativos: ao gerar NÃO_VER,\na atenção se concentra em "não" e "vi"',
        ha='right', va='top', fontsize=8.5, color=NEU)
fig.savefig(os.path.join(OUT, 'fig_atencao.png')); plt.close(fig)

# ---- LoRA (laranja) -------------------------------------------------------
fig, ax = tela(5.4, 2.3)
caixa(ax, 0.3, 9, 8, 4.4, 'entrada x', fs=9.5)
caixa(ax, 15, 15.8, 22, 5.6, 'W: pesos pré-treinados\n(congelados)', ec=G, fc='#eceef1', fs=9)
caixa(ax, 14, 3, 8.5, 5.6, 'A\n(r × d)', ec=O, fc=LO, fs=9.5, lw=1.6)
caixa(ax, 29, 3, 8.5, 5.6, 'B\n(d × r)', ec=O, fc=LO, fs=9.5, lw=1.6)
caixa(ax, 41.5, 9, 3.8, 4.4, '+', fs=12, bold=True)
caixa(ax, 47.5, 9, 6.2, 4.4, 'saída', fs=9.5)
seta(ax, (8.4, 12.4), (14.9, 17.5)); seta(ax, (8.4, 10), (13.9, 6.5))
seta(ax, (22.6, 5.8), (28.9, 5.8), c=O); seta(ax, (37.1, 17.5), (42.5, 13.5)); seta(ax, (37.6, 6.5), (42.5, 8.9), c=O)
seta(ax, (45.4, 11.2), (47.4, 11.2))
ax.text(26, 11.2, 'saída = W·x + (α/r)·B·A·x', ha='center', va='center', fontsize=9.5, weight='bold')
ax.text(27, 0.4, 'só A e B são treinadas (r = 16, α = 32): 589.824 parâmetros, 0,97% do modelo',
        ha='center', fontsize=8.5, color=O, weight='bold')
fig.savefig(os.path.join(OUT, 'fig_lora.png')); plt.close(fig)

# ---- arquitetura da prova de conceito ------------------------------------
fig, ax = tela(6.1, 2.5)
caixa(ax, 0.3, 14, 9.8, 6.2, 'Extensão no\nChrome: grava\n10–30 s da aba', ec=ESC, fs=8.5)
caixa(ax, 0.3, 4, 9.8, 6.2, 'Página web:\ntexto ou\narquivo', ec=ESC, fs=8.5)
ax.add_patch(FancyBboxPatch((12.6, 1), 33.4, 23, boxstyle='round,pad=0,rounding_size=0.6',
                            fc='#f6f7f9', ec='#b8bdc5', lw=1, ls='--'))
ax.text(29.3, 21.6, 'Servidor local (FastAPI, CPU)', ha='center', fontsize=8.5, color=NEU, weight='bold')
for t, x in [('ffmpeg:\náudio\n16 kHz', 13.6), ('Whisper:\nfala →\ntexto', 21.1), ('divisão\nem\nfrases', 28.6)]:
    caixa(ax, x, 8.5, 6.4, 7, t, fs=8.5)
for a in (20.1, 27.6): seta(ax, (a, 12), (a + 0.9, 12))
for t, y, cor in [('Regras', 16.3, G), ('Seq2seq*', 9.8, B), ('LoRA', 3.3, O)]:
    caixa(ax, 37.6, y, 7.6, 4.4, t, ec=cor, fs=8.5, lw=1.8)
    seta(ax, (35.1, 12), (37.5, y + 2.2))
    seta(ax, (45.3, y + 2.2), (48.3, 12))
caixa(ax, 48.4, 6.5, 12.4, 11, 'glosas lado a\nlado, no formato\ndo motor de\nsinalização\ndo VLibras', ec=ESC, fs=8.5)
seta(ax, (10.2, 17), (13.5, 13.5)); seta(ax, (10.2, 7.2), (13.5, 10.5))
ax.text(41.4, 0.0, '* arquivo não preservado', ha='center', fontsize=7.5, color=NEU)
fig.savefig(os.path.join(OUT, 'fig_arquitetura_poc.png')); plt.close(fig)

# ---- análise de erros: marcas da glosa ------------------------------------
r = pd.read_csv(p_reg); l = pd.read_csv(p_lora)
Rf = r.glosa.astype(str).str.upper(); Hf = r.glosa_pred_regras.fillna('').astype(str).str.upper()
Lf = l.glosa_pred_lora.fillna('').astype(str).str.upper()
marcas = [('Desambiguação\n(&)', r'&'), ('Negação\nfundida', r'NÃO_'), ('Verbo\ndirecional', r'\dS_\w+_\dS')]
fig, axs = plt.subplots(1, 2, figsize=(6.1, 2.8), gridspec_kw={'width_ratios': [2.6, 1.2]})
ax = axs[0]; x = np.arange(len(marcas)); w = .27
series = [(Rf, 'Referência', REF), (Hf, 'Regras', G), (Lf, 'PTT5 + LoRA', O)]
for k, (serie, rot, cor) in enumerate(series):
    v = [100 * serie.str.contains(p, regex=True).mean() for _, p in marcas]
    bs = ax.bar(x + (k - 1) * w, v, w, color=cor, label=rot)
    for b in bs:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, br(b.get_height()), ha='center', fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels([m[0] for m in marcas], fontsize=8.5)
ax.set_ylabel('Frases com a marca (%)'); ax.set_ylim(0, 82); ax.legend(frameon=False, fontsize=8, loc='upper right')
ax = axs[1]
tam = [s.str.split().str.len().mean() for s, _, _ in series]
bs = ax.bar(['Ref.', 'Regras', 'LoRA'], tam, color=[REF, G, O], width=.62)
for b in bs:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .2, br(b.get_height()), ha='center', fontsize=8)
ax.tick_params(axis='x', labelsize=8.5)
ax.set_ylabel('Glosas por frase (média)'); ax.set_ylim(0, 12.5)
fig.tight_layout(w_pad=1.5)
fig.savefig(os.path.join(OUT, 'fig10_erros.png')); plt.close(fig)
