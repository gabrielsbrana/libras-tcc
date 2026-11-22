"""Figuras explicativas do artigo (atenção, LoRA, arquitetura da prova de conceito) e
figura da análise de erros. Uso: python gerar_figuras_didaticas.py <predictions_regras.csv> <predictions_lora_test.csv> <saida/>"""
import sys, os, re
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
p_reg, p_lora, OUT = sys.argv[1:4]
G, B, O = '#8a8f98', '#2a6fb0', '#d98324'; INK, NEU, ESC = '#222222', '#5b6472', '#3d4652'
plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 220, 'savefig.bbox': 'tight'})
br = lambda v, d=1: f'{v:.{d}f}'.replace('.', ',')

def novo(w, h, xl, yl):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, xl); ax.set_ylim(0, yl); ax.axis('off'); return fig, ax
def caixa(ax, x, y, w, h, txt, ec=NEU, fc='white', fs=11, bold=False, color=INK, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.6', fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs, color=color,
            weight='bold' if bold else 'normal', linespacing=1.25)
def seta(ax, a, b, c=INK, lw=1.4, ms=13):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=ms, color=c, lw=lw, shrinkA=0, shrinkB=0))

# ---- atenção (seq2seq, azul) ---------------------------------------------
fig, ax = novo(11.5, 4.9, 115, 46); ax.set_ylim(-5, 46)
pt = ['Você', 'não', 'viu', 'ninguém', '?']; gl = ['VOCÊ', 'NÃO_VER', 'NINGUÉM', '[INTERROGAÇÃO]']
LB = '#e8f0f9'
ax.text(25, -3.5, 'Codificador (LSTM bidirecional)', ha='center', fontsize=11.5, color=B, weight='bold')
ax.text(85.5, -3.5, 'Decodificador (LSTM)', ha='center', fontsize=11.5, color=B, weight='bold')
for i, w in enumerate(pt):
    x = 2 + i * 9.6
    caixa(ax, x, 2, 8.4, 6, w, fs=9.6)
    caixa(ax, x, 16, 8.4, 8, f'h{i+1}', ec=B, fc=LB, fs=12)
    seta(ax, (x + 4.2, 8.3), (x + 4.2, 15.7), c=NEU)
for i, w in enumerate(gl):
    x = 58 + i * 14.2
    caixa(ax, x, 16, 13.4, 8, f's{i+1}', ec=B, fc=LB, fs=12)
    caixa(ax, x, 2, 13.4, 6, w, fs=8.4)
    seta(ax, (x + 6.7, 15.7), (x + 6.7, 8.3), c=NEU)
caixa(ax, 31, 36, 26, 8.5, 'atenção: média ponderada\nde h1…h5', ec=B, fc='white', fs=10.5, lw=2)
pesos = [0.05, 0.42, 0.38, 0.1, 0.05]
for i, p in enumerate(pesos):
    x = 2 + i * 9.6 + 4.2
    ax.plot([x, 44], [24.2, 35.8], color=B, lw=0.6 + 9 * p, alpha=0.75, solid_capstyle='round')
    ax.text(x - 1.8 if i < 2 else x + 0.6, 26.2 if i else 25.4, br(p, 2), fontsize=9, color=B)
seta(ax, (57.3, 40), (78.9, 24.4), c=B, lw=1.8)
ax.text(68.5, 34.2, 'contexto usado\npara gerar s2', fontsize=10, color=B, ha='left')
ax.text(88, 42.5, 'pesos ilustrativos: ao gerar NÃO_VER,\na atenção concentra-se em "não" e "viu"', ha='center', fontsize=9.8, color=NEU)
fig.savefig(os.path.join(OUT, 'fig_atencao.png')); plt.close(fig)

# ---- LoRA (laranja) -------------------------------------------------------
LO = '#fbeedf'
fig, ax = novo(10, 4.2, 100, 42)
caixa(ax, 2, 17, 15, 8, 'entrada x', fs=12)
caixa(ax, 32, 28, 36, 10, 'W  (pesos pré-treinados)\ncongelados', ec=G, fc='#eceef1', fs=11.5)
caixa(ax, 30, 3, 15, 10, 'A\n(r × d)', ec=O, fc=LO, fs=12, lw=2)
caixa(ax, 55, 3, 15, 10, 'B\n(d × r)', ec=O, fc=LO, fs=12, lw=2)
caixa(ax, 78, 17, 7, 8, '+', fs=17, bold=True)
caixa(ax, 89, 17, 10, 8, 'saída', fs=12)
seta(ax, (17.2, 23), (31.8, 32)); seta(ax, (17.2, 19), (29.8, 8))
seta(ax, (45.2, 8), (54.8, 8), c=O); seta(ax, (68.2, 33), (79, 25.2)); seta(ax, (70.2, 8), (79, 16.8), c=O)
seta(ax, (85.2, 21), (88.8, 21))
ax.text(50, 20.5, 'saída = W·x + (α/r)·B·A·x', ha='center', fontsize=12.5, weight='bold')
ax.text(50, 0, 'só A e B são treinadas: r = 16, α = 32 → 589.824 parâmetros (0,97% do modelo)',
        ha='center', fontsize=10.2, color=O, weight='bold')
fig.savefig(os.path.join(OUT, 'fig_lora.png')); plt.close(fig)

# ---- arquitetura da prova de conceito ------------------------------------
fig, ax = novo(12, 3.9, 120, 39)
caixa(ax, 1, 25, 21, 10, 'Extensão no Chrome\ngrava 10–30 s da aba', ec=ESC, fs=10)
caixa(ax, 1, 9, 21, 10, 'Página web\ntexto ou arquivo', ec=ESC, fs=10)
ax.add_patch(FancyBboxPatch((28, 3), 64, 34, boxstyle='round,pad=0,rounding_size=1', fc='#f6f7f9', ec='#b8bdc5', lw=1.2, ls='--'))
ax.text(60, 34.2, 'Servidor local (FastAPI, CPU)', ha='center', fontsize=10.5, color=NEU, weight='bold')
etapas = [('ffmpeg\náudio 16 kHz', 30), ('Whisper small\nfala → texto', 46), ('divisão\nem frases', 62)]
for t, x in etapas: caixa(ax, x, 16, 13.5, 11, t, fs=9.8)
for a, b in [(43.6, 46), (59.6, 62)]: seta(ax, (a, 21.5), (b - 0.2, 21.5))
caixa(ax, 79, 25.5, 11.5, 7, 'Regras', ec=G, fs=10, lw=2.2)
caixa(ax, 79, 15.5, 11.5, 7, 'Seq2seq*', ec=B, fs=10, lw=2.2)
caixa(ax, 79, 5.5, 11.5, 7, 'LoRA', ec=O, fs=10, lw=2.2)
for y in (29, 19, 9): seta(ax, (75.6, 21.5), (78.8, y))
caixa(ax, 98, 12, 21, 16, 'glosas lado a lado\n\nformato aceito pelo\nmotor de sinalização\ndo VLibras', ec=ESC, fs=9.8)
for y in (29, 19, 9): seta(ax, (90.7, y), (97.8, 20))
seta(ax, (22.2, 30), (29.8, 23)); seta(ax, (22.2, 14), (29.8, 19))
ax.text(84.7, 1.2, '* modelo não preservado', ha='center', fontsize=8.6, color=NEU)
fig.savefig(os.path.join(OUT, 'fig_arquitetura_poc.png')); plt.close(fig)

# ---- análise de erros: marcas da glosa ------------------------------------
r = pd.read_csv(p_reg); l = pd.read_csv(p_lora)
Rf = r.glosa.astype(str).str.upper(); Hf = r.glosa_pred_regras.fillna('').astype(str).str.upper()
Lf = l.glosa_pred_lora.fillna('').astype(str).str.upper()
marcas = [('Desambiguação\n(&)', r'&'), ('Negação\nfundida (NÃO_)', r'NÃO_'), ('Verbo direcional\n(1S_…_2S)', r'\dS_\w+_\dS')]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={'width_ratios': [3, 1.35]})
ax = axs[0]; x = np.arange(len(marcas)); w = .26
for k, (serie, rot, cor) in enumerate([(Rf, 'Referência', '#3d4652'), (Hf, 'Regras', G), (Lf, 'PTT5-small + LoRA', O)]):
    v = [100 * serie.str.contains(p, regex=True).mean() for _, p in marcas]
    bs = ax.bar(x + (k - 1) * w, v, w, color=cor, label=rot)
    for b in bs: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, br(b.get_height()) , ha='center', fontsize=9.6)
ax.set_xticks(x); ax.set_xticklabels([m[0] for m in marcas], fontsize=10.5)
ax.set_ylabel('Frases com a marca (%)'); ax.set_ylim(0, 80); ax.legend(frameon=False, fontsize=10, loc='upper right')
ax = axs[1]
tam = [Rf.str.split().str.len().mean(), Hf.str.split().str.len().mean(), Lf.str.split().str.len().mean()]
bs = ax.bar(['Ref.', 'Regras', 'LoRA'], tam, color=['#3d4652', G, O], width=.6)
for b in bs: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .2, br(b.get_height()), ha='center', fontsize=10)
ax.set_ylabel('Glosas por frase (média)'); ax.set_ylim(0, 12.5)
fig.tight_layout(w_pad=3)
fig.savefig(os.path.join(OUT, 'fig10_erros.png')); plt.close(fig)
