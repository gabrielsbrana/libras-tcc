"""Gera as figuras do artigo (versão SBC). Paleta fixa em todo o texto:
modelos  -> Regras cinza, Seq2seq azul, LoRA laranja
domínios -> Geral verde-azulado, Governamental roxo
Uso: python gerar_figuras_artigo.py <corpus.csv> <predictions_regras.csv> <predictions_lora_test.csv>
       <seq2seq_training_history.csv> <lora_training_history.csv> <saida/>"""
import sys, os, re
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sacrebleu
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from text_utils import categorias_presentes, normalizar_glosa, tem_repeticao, CATEGORIAS  # noqa

corpus, p_reg, p_lora, h_s2s, h_lora, OUT = sys.argv[1:7]
os.makedirs(OUT, exist_ok=True)
G, B, O = '#8a8f98', '#2a6fb0', '#d98324'          # regras, seq2seq, lora
DG, DV = '#2e8b7a', '#7b5ea7'                      # geral, governamental
INK, NEU = '#222222', '#5b6472'
plt.rcParams.update({'font.size': 12, 'font.family': 'DejaVu Sans', 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 220, 'savefig.bbox': 'tight'})
br = lambda v, d=1: f'{v:.{d}f}'.replace('.', ',')
mil = lambda n: f'{n:,}'.replace(',', '.')

# ---------- dados medidos -------------------------------------------------
GLOBAL = {'Regras': (33.98, 69.71), 'Seq2seq': (28.08, 50.67), 'LoRA': (9.19, 24.05)}
DOM = {'Geral': {'Regras': (36.99, 68.61), 'Seq2seq': (26.86, 44.46), 'LoRA': (11.58, 25.11)},
       'Governamental': {'Regras': (30.49, 70.64), 'Seq2seq': (29.26, 55.81), 'LoRA': (6.68, 23.16)}}
# média do BLEU por sentença (mesmo cálculo do evaluate.py original para regras e seq2seq)
S2S_CAT = {'marcador_ponto': 38.37, 'desambiguacao_amp': 30.23, 'composto_underscore': 30.18, 'numeral': 18.48,
           'negacao_fundida': 31.24, 'interrogacao': 57.14, 'verbo_direcional': 18.57,
           'negacao_isolada': 29.74, 'exclamacao': 47.06}
NOME = {'marcador_ponto': 'Ponto final', 'desambiguacao_amp': 'Desambiguação (&)',
        'composto_underscore': 'Sinal composto (_)', 'numeral': 'Numeral', 'negacao_fundida': 'Negação fundida',
        'interrogacao': 'Interrogação', 'verbo_direcional': 'Verbo direcional',
        'negacao_isolada': 'Negação isolada', 'exclamacao': 'Exclamação'}

reg = pd.read_csv(p_reg); lora = pd.read_csv(p_lora)
assert (reg.pt.values == lora.pt.values).all()
R = [normalizar_glosa(x) for x in reg.glosa.astype(str)]
HR = [normalizar_glosa(x) for x in reg.glosa_pred_regras.fillna('').astype(str)]
HL = [normalizar_glosa(x) for x in lora.glosa_pred_lora.fillna('').astype(str)]
assert round(sacrebleu.corpus_bleu(HR, [R]).score, 2) == 33.98
assert round(sacrebleu.corpus_bleu(HL, [R]).score, 2) == 9.19
sbR = [sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(HR, R)]
sbL = [sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(HL, R)]
marc = [categorias_presentes(r) for r in R]
CAT = []
for c in CATEGORIAS:
    idx = [i for i, m in enumerate(marc) if m[c]]
    CAT.append((c, len(idx), np.mean([sbR[i] for i in idx]), S2S_CAT[c], np.mean([sbL[i] for i in idx])))
pd.DataFrame(CAT, columns=['categoria', 'pares', 'regras', 'seq2seq', 'lora']).round(2).to_csv(
    os.path.join(OUT, 'tabela_categorias.csv'), index=False)

palavras = reg.pt.astype(str).str.split().str.len().values
FAIXAS = [(1, 5, '1–5'), (6, 10, '6–10'), (11, 20, '11–20'), (21, 40, '21–40'), (41, 10**5, '41+')]
COMP = []
for lo, hi, rot in FAIXAS:
    idx = [i for i, w in enumerate(palavras) if lo <= w <= hi]
    COMP.append((rot, len(idx),
                 sacrebleu.corpus_bleu([HR[i] for i in idx], [[R[i] for i in idx]]).score,
                 sacrebleu.corpus_bleu([HL[i] for i in idx], [[R[i] for i in idx]]).score,
                 100 * np.mean([tem_repeticao(HL[i]) for i in idx])))
pd.DataFrame(COMP, columns=['faixa', 'pares', 'bleu_regras', 'bleu_lora', 'repeticao_lora_pct']).round(2).to_csv(
    os.path.join(OUT, 'tabela_comprimento.csv'), index=False)

def salvar(fig, nome):
    fig.savefig(os.path.join(OUT, nome)); plt.close(fig)

# ---------- Figura 1: visão geral -------------------------------------------
fig, ax = plt.subplots(figsize=(15.5, 6.4)); ax.set_xlim(0, 130); ax.set_ylim(0, 62); ax.axis('off')
def caixa(x, y, w, h, fc, ec, lw=1.6, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw))
def seta(a, b, c=INK):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=16, color=c, lw=1.6))
def bloco(x, w, titulo, cor):
    caixa(x, 4, w, 52, '#ffffff', cor, lw=2)
    caixa(x, 50, w, 6, cor, cor, r=1.2)
    ax.text(x + w / 2, 53, titulo, ha='center', va='center', color='white', fontsize=13.5, weight='bold')
ESC = '#3d4652'
bloco(1, 20, '1. Problema', ESC)
ax.text(11, 44, 'Vídeo e texto em\nportuguês sem\nversão em Libras', ha='center', va='top', fontsize=11.5, linespacing=1.35)
caixa(3, 13, 16, 13, '#f3f4f6', '#c5c9d0', lw=1)
ax.text(11, 23.6, 'Texto', ha='center', fontsize=10.5, color=NEU)
ax.text(11, 19.5, '↓ glosa ↓', ha='center', fontsize=11.5, weight='bold')
ax.text(11, 15.3, 'avatar', ha='center', fontsize=10.5, color=NEU)
ax.text(11, 8, 'a glosa define o\nque o avatar sinaliza', ha='center', fontsize=9.8, color=NEU, linespacing=1.3)

bloco(24, 21, '2. Dados', ESC)
ax.text(34.5, 46, 'VLibrasBD', ha='center', fontsize=13, weight='bold')
ax.text(34.5, 41.5, '127.349 pares\nportuguês–glosa', ha='center', va='top', fontsize=11, linespacing=1.3)
for k, (rot, n, cor) in enumerate([('Treino', '101.850', '#c9ced6'), ('Validação', '12.731', '#dde1e7'), ('Teste', '12.732', '#eef0f3')]):
    y = 27 - k * 6.2
    caixa(26.5, y, 16, 5, cor, '#aeb4bd', lw=0.8, r=0.6)
    ax.text(27.3, y + 2.5, rot, va='center', fontsize=9.5)
    ax.text(42, y + 2.5, n, va='center', ha='right', fontsize=9.5, weight='bold')
ax.text(34.5, 8.6, 'estratificado por domínio:', ha='center', fontsize=9.6, color=NEU)
ax.text(28.8, 5.6, 'geral', ha='center', fontsize=10.5, color=DG, weight='bold')
ax.text(38.6, 5.6, 'governamental', ha='center', fontsize=9.8, color=DV, weight='bold')

bloco(48, 24, '3. Três tradutores', ESC)
for k, (tit, sub, cor) in enumerate([('Regras', 'vlibras-translate\nsem treino', G),
                                     ('Seq2seq do zero', 'LSTM + atenção\n40.000 pares, CPU', B),
                                     ('PTT5-small + LoRA', '0,97% dos parâmetros\n101.850 pares, GPU T4', O)]):
    y = 35.5 - k * 14.3
    caixa(50, y, 20, 12, '#ffffff', cor, lw=2.2, r=1)
    ax.add_patch(plt.Rectangle((50, y), 1.6, 12, color=cor))
    ax.text(61, y + 8.8, tit, ha='center', fontsize=11.5, weight='bold', color=INK)
    ax.text(61, y + 4.0, sub, ha='center', va='center', fontsize=9.6, color=NEU, linespacing=1.25)

bloco(75, 23, '4. Avaliação', ESC)
ax.text(86.5, 46.5, 'Mesmo teste\n(12.732 pares)', ha='center', va='top', fontsize=11, weight='bold', linespacing=1.3)
ax.text(86.5, 38.5, 'mesma normalização', ha='center', fontsize=10, color=NEU)
for k, t in enumerate(['BLEU e chrF globais', 'por domínio', 'por 9 fenômenos\nlinguísticos da glosa', 'por comprimento\nda frase']):
    ax.text(77.5, 33 - k * 6.4, '•', fontsize=13, va='top')
    ax.text(79.5, 33 - k * 6.4, t, fontsize=10.4, va='top', linespacing=1.2)

bloco(101, 28, '5. Resultados', ESC)
ax.text(115, 46.8, 'BLEU no teste', ha='center', fontsize=10.5, color=NEU)
for k, (nome, v, cor) in enumerate([('Regras', 33.98, G), ('Seq2seq', 28.08, B), ('LoRA', 9.19, O)]):
    y = 40.5 - k * 4.6
    ax.text(110, y + 1.4, nome, ha='right', va='center', fontsize=10.2)
    ax.add_patch(plt.Rectangle((111, y), 14 * v / 34, 2.8, color=cor))
    ax.text(111.6 + 14 * v / 34, y + 1.4, br(v, 2), va='center', fontsize=10.2, weight='bold')
caixa(103, 6, 24, 19, '#f3f4f6', '#c5c9d0', lw=1)
ax.text(115, 22.3, 'Prova de conceito', ha='center', fontsize=11, weight='bold')
ax.text(115, 18.6, 'vídeo ou áudio\n↓ Whisper\nportuguês\n↓ tradutores\nglosas lado a lado', ha='center', va='top',
        fontsize=9.6, linespacing=1.2)
for xa, xb in [(21, 24), (45, 48), (72, 75), (98, 101)]:
    seta((xa + 0.2, 30), (xb - 0.2, 30))
salvar(fig, 'fig01_visao_geral.png')

# ---------- Figura 2: comprimento por domínio (corpus inteiro) --------------
c = pd.read_csv(corpus)
c['n'] = c['pt-br'].astype(str).str.split().str.len().clip(upper=60)
fig, ax = plt.subplots(figsize=(7.5, 3.9)); bins = np.arange(0.5, 61.5, 2)
ger = c[~c.is_government_source].n; gov = c[c.is_government_source].n
ax.hist(ger, bins=bins, color=DG, alpha=.75, label=f'Geral (mediana {int(ger.median())})')
ax.hist(gov, bins=bins, color=DV, alpha=.6, label=f'Governamental (mediana {int(gov.median())})')
ax.set_xlabel('Palavras por sentença em português (60 = 60 ou mais)'); ax.set_ylabel('Sentenças')
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: mil(int(v))))
ax.legend(frameon=False)
salvar(fig, 'fig02_comprimento.png')

# ---------- Figura 3: frequência dos fenômenos no teste ---------------------
fig, ax = plt.subplots(figsize=(7.5, 3.9))
ordem = sorted(CAT, key=lambda r: r[1])
ax.barh([NOME[r[0]] for r in ordem], [r[1] for r in ordem], color=NEU)
for i, r in enumerate(ordem):
    ax.text(r[1] + 100, i, f'{mil(r[1])} ({br(100 * r[1] / 12732)}%)', va='center', fontsize=10.5)
ax.set_xlim(0, 12000); ax.set_xlabel('Pares do conjunto de teste em que o fenômeno aparece na referência')
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: mil(int(v))))
salvar(fig, 'fig03_fenomenos.png')

# ---------- Figura 4: curvas de treino -------------------------------------
s = pd.read_csv(h_s2s); l = pd.read_csv(h_lora)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
ax = axs[0]
ax.plot(s.epoch, s.train_loss, color=B, marker='o', ms=4, lw=1.6, ls='--', label='treino')
ax.plot(s.epoch, s.val_loss, color=B, marker='o', ms=5, lw=2.2, label='validação')
best = s.loc[s.val_loss.idxmin()]
ax.axvline(best.epoch, color='#9aa0a8', lw=1, ls=':')
ax.annotate(f'melhor: época {int(best.epoch)}\n({br(best.val_loss, 2)})', (best.epoch, best.val_loss),
            xytext=(best.epoch - 3.3, best.val_loss + 0.55), fontsize=10, arrowprops=dict(arrowstyle='->', lw=0.9))
ax.set_title('Seq2seq do zero (40.000 pares)', fontsize=12); ax.set_xlabel('Época'); ax.set_ylabel('Perda (entropia cruzada)')
ax.set_xticks(range(1, 11)); ax.legend(frameon=False)
ax = axs[1]
ev = l.dropna(subset=['eval_loss']); tr = l.dropna(subset=['train_loss'])
tr = tr.assign(ep=np.ceil(tr.epoch)).groupby('ep').train_loss.mean().reset_index()
ax.plot(ev.epoch.round(), ev.eval_loss, color=O, marker='o', ms=5, lw=2.2, label='validação')
ax.set_title('PTT5-small + LoRA (101.850 pares)', fontsize=12); ax.set_xlabel('Época')
ax.set_xticks(range(1, 16, 2)); ax.legend(frameon=False)
ax.set_ylim(1.5, 2.4)
fig.tight_layout(w_pad=3)
salvar(fig, 'fig04_curvas_treino.png')

# ---------- Figura 5: global -----------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 3.8)); x = np.arange(2); w = .26
for k, (nome, rot, cor) in enumerate([('Regras', 'Regras (vlibras-translate)', G), ('Seq2seq', 'Seq2seq do zero', B),
                                      ('LoRA', 'PTT5-small + LoRA', O)]):
    v = GLOBAL[nome]; bs = ax.bar(x + (k - 1) * w, v, w, color=cor, label=rot)
    for b in bs: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, br(b.get_height(), 2), ha='center', fontsize=10.5)
ax.set_xticks(x); ax.set_xticklabels(['BLEU', 'chrF']); ax.set_ylabel('Pontuação (0 a 100)'); ax.set_ylim(0, 85)
ax.legend(frameon=False, loc='upper left', fontsize=10.5)
salvar(fig, 'fig05_global.png')

# ---------- Figura 6: por domínio ------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for j, (met, lim) in enumerate([('BLEU', 45), ('chrF', 85)]):
    ax = axs[j]; x = np.arange(2)
    for k, (nome, cor) in enumerate([('Regras', G), ('Seq2seq', B), ('LoRA', O)]):
        v = [DOM[d][nome][j] for d in ('Geral', 'Governamental')]
        bs = ax.bar(x + (k - 1) * w, v, w, color=cor, label={'Regras': 'Regras', 'Seq2seq': 'Seq2seq do zero', 'LoRA': 'PTT5-small + LoRA'}[nome])
        for b in bs: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .8, br(b.get_height()), ha='center', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(['Geral', 'Governamental'])
    for t, cor in zip(ax.get_xticklabels(), (DG, DV)): t.set_color(cor); t.set_fontweight('bold')
    ax.set_title(met, fontsize=12); ax.set_ylim(0, lim)
h, lb = axs[0].get_legend_handles_labels()
fig.legend(h, lb, frameon=False, loc='lower center', ncol=3, fontsize=10.5, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(w_pad=3, rect=(0, 0.06, 1, 1))
salvar(fig, 'fig06_dominio.png')

# ---------- Figura 7: por fenômeno -----------------------------------------
fig, ax = plt.subplots(figsize=(10, 5)); x = np.arange(len(CAT)); w = .27
ax.bar(x - w, [r[2] for r in CAT], w, color=G, label='Regras')
ax.bar(x, [r[3] for r in CAT], w, color=B, label='Seq2seq do zero')
ax.bar(x + w, [r[4] for r in CAT], w, color=O, label='PTT5-small + LoRA')
for i, r in enumerate(CAT):
    d = r[3] - r[2]
    ax.text(i, max(r[2], r[3]) + 1.5, ('+' if d > 0 else '−') + br(abs(d)), ha='center', fontsize=10,
            color='#2f7d4f' if d > 0 else '#b23a3a', weight='bold')
ax.set_xticks(x); ax.set_xticklabels([f'{NOME[r[0]]}\n({mil(r[1])})' for r in CAT], fontsize=10, rotation=30, ha='right', rotation_mode='anchor')
ax.set_ylabel('BLEU médio por sentença'); ax.set_ylim(0, 68)
ax.legend(frameon=False, loc='upper left', ncol=3, fontsize=10.5)
fig.tight_layout()
salvar(fig, 'fig07_fenomenos_bleu.png')

# ---------- Figura 8: por comprimento --------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8)); x = np.arange(len(COMP)); w = .38
ax = axs[0]
b1 = ax.bar(x - w / 2, [r[2] for r in COMP], w, color=G, label='Regras')
b2 = ax.bar(x + w / 2, [r[3] for r in COMP], w, color=O, label='PTT5-small + LoRA')
for b in list(b1) + list(b2): ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .8, br(b.get_height()), ha='center', fontsize=9.6)
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in COMP]); ax.set_xlabel('Palavras na sentença em português')
ax.set_ylabel('BLEU'); ax.set_ylim(0, 58); ax.legend(frameon=False, fontsize=10.5)
ax = axs[1]
bs = ax.bar(x, [r[4] for r in COMP], .55, color=O)
for b in bs: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .8, br(b.get_height()) + '%', ha='center', fontsize=10)
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in COMP]); ax.set_xlabel('Palavras na sentença em português')
ax.set_ylabel('Saídas do LoRA com repetição (%)'); ax.set_ylim(0, 55)
fig.tight_layout(w_pad=3)
salvar(fig, 'fig08_comprimento_resultado.png')

print(pd.DataFrame(CAT, columns=['cat', 'n', 'regras', 's2s', 'lora']).round(2).to_string())
print(pd.DataFrame(COMP, columns=['faixa', 'n', 'regras', 'lora', 'rep']).round(2).to_string())
