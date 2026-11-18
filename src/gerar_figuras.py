"""
Gera as figuras do TCC em results/figuras/.

Uso: python src/gerar_figuras.py
Precisa de data/processed/test.csv e val.csv (ver data/README.md) e de
results/lora_training_history.csv. Os valores de regras e seq2seq estão
escritos no script: as predições individuais dessas duas abordagens não
foram preservadas, só as métricas (ver results/RESULTADOS.md).
"""
import pandas as pd, re, sys, io
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.size':12,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'figure.dpi':200})
B='#2a6fb0'; G='#8a8f98'; O='#d98324'; GR='#3f8f5a'; LG='#d9dce1'
import os
RAIZ=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS=os.path.join(RAIZ,'data','processed'); SAIDA=os.path.join(RAIZ,'results','figuras')
os.makedirs(SAIDA,exist_ok=True); os.chdir(RAIZ)
t=pd.read_csv(os.path.join(DADOS,'test.csv')); v=pd.read_csv(os.path.join(DADOS,'val.csv')); a=pd.concat([t,v])
g=t.glosa.astype(str).str.upper()

# F4 length distribution
a['lp']=a.pt.astype(str).str.split().str.len()
fig,ax=plt.subplots(figsize=(8,4))
bins=range(0,62,2)
ax.hist(a[~a.is_government_source].lp.clip(upper=60),bins=bins,alpha=.75,color=B,label='Geral')
ax.hist(a[a.is_government_source].lp.clip(upper=60),bins=bins,alpha=.65,color=O,label='Governamental')
ax.set_xlabel('Palavras por sentença em português (60 = 60 ou mais)'); ax.set_ylabel('Número de sentenças'); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig('results/figuras/f_comprimento.png'); plt.close()

# F5 category frequency
cats=[('Marcador de ponto final',9310),('Desambiguação de sentido (&)',7197),('Sinal composto (_)',2562),('Numeral',2005),('Negação fundida ao verbo',1316),('Interrogação',848),('Verbo direcional',700),('Negação isolada',379),('Exclamação',100)]
fig,ax=plt.subplots(figsize=(8,4.2))
names=[c[0] for c in cats][::-1]; vals=[c[1] for c in cats][::-1]
ax.barh(names,vals,color=B)
for i,x in enumerate(vals): ax.text(x+120,i,f"{x:,}".replace(',','.')+" ("+f"{100*x/12732:.1f}".replace('.',',')+"%)",va='center',fontsize=11)
ax.set_xlim(0,11800); ax.set_xlabel('Pares do conjunto de teste (total: 12.732)')
fig.tight_layout(); fig.savefig('results/figuras/f_categorias.png'); plt.close()

# F6 lora curve
h=pd.read_csv('results/lora_training_history.csv'); e=h.dropna(subset=['eval_loss'])
fig,ax=plt.subplots(figsize=(8,4))
ax.plot(e.epoch,e.eval_loss,marker='o',color=B)
for ep in (1,5,10,15):
    val=float(e[e.epoch==ep].eval_loss.iloc[0]); ax.annotate(f'{val:.2f}'.replace('.',','),(ep,val),textcoords='offset points',xytext=(0,9),ha='center',fontsize=11)
ax.set_xlabel('Época'); ax.set_ylabel('Perda de validação (entropia cruzada)'); ax.set_xticks(range(1,16)); ax.set_ylim(1.6,2.3)
fig.tight_layout(); fig.savefig('results/figuras/f_lora_curva.png'); plt.close()

# F7 global
fig,ax=plt.subplots(figsize=(7,4))
import numpy as np
x=np.arange(2); w=.27
r=[33.98,69.71]; s=[28.08,50.67]; l=[9.19,24.05]
b1=ax.bar(x-w,r,w,color=G,label='Regras (vlibras-translate)'); b2=ax.bar(x,s,w,color=B,label='Seq2seq do zero'); b3=ax.bar(x+w,l,w,color=O,label='PTT5-small + LoRA')
for bs in (b1,b2,b3):
    for b in bs: ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f'{b.get_height():.2f}'.replace('.',','),ha='center',fontsize=11)
ax.set_xticks(x); ax.set_xticklabels(['BLEU','chrF']); ax.set_ylim(0,90); ax.set_ylabel('Pontuação (0 a 100)'); ax.legend(frameon=False,loc='upper left')
fig.tight_layout(); fig.savefig('results/figuras/f_global.png'); plt.close()

# F8 category
rows=[('Ponto final',39.41,38.37,8.69),('Desambiguação',28.52,30.23,6.50),('Sinal composto',35.47,30.18,7.23),('Numeral',27.18,18.48,6.15),('Negação fundida',38.85,31.24,8.30),('Interrogação',45.60,57.14,23.44),('Verbo direcional',20.27,18.57,6.45),('Negação isolada',30.86,29.74,5.78),('Exclamação',48.55,47.06,14.36)]
w=.27
fig,ax=plt.subplots(figsize=(8.6,4.4))
x=np.arange(len(rows)); 
ax.bar(x-w,[r[1] for r in rows],w,color=G,label='Regras'); ax.bar(x,[r[2] for r in rows],w,color=B,label='Seq2seq do zero'); ax.bar(x+w,[r[3] for r in rows],w,color=O,label='PTT5-small + LoRA')
for i,r in enumerate(rows):
    d=r[2]-r[1]; ax.text(i,max(r[1],r[2])+1.2,('+' if d>0 else '−')+f'{abs(d):.1f}'.replace('.',','),ha='center',fontsize=11,color=GR if d>0 else '#b23a3a')
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows],rotation=30,ha='right'); ax.set_ylabel('BLEU'); ax.set_ylim(0,78); ax.legend(frameon=False,loc='upper left',ncol=3,fontsize=10)
fig.tight_layout(); fig.savefig('results/figuras/f_categoria_bleu.png'); plt.close()

# F9 domain
fig,axs=plt.subplots(1,2,figsize=(8.6,3.9),sharey=False)
for ax,(m,rv,sv,lv) in zip(axs,[('BLEU',[36.99,30.49],[26.86,29.26],[11.58,6.68]),('chrF',[68.61,70.64],[44.46,55.81],[25.11,23.16])]):
    x=np.arange(2)
    b1=ax.bar(x-w,rv,w,color=G,label='Regras'); b2=ax.bar(x,sv,w,color=B,label='Seq2seq do zero'); b3=ax.bar(x+w,lv,w,color=O,label='PTT5 + LoRA')
    for bs in (b1,b2,b3):
        for b in bs: ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f'{b.get_height():.1f}'.replace('.',','),ha='center',fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(['Geral','Governamental']); ax.set_title(m,fontsize=12); ax.set_ylim(0,max(rv+sv)*1.3)
axs[0].legend(frameon=False,loc='upper right',fontsize=9)
fig.tight_layout(); fig.savefig('results/figuras/f_dominio.png'); plt.close()

def box(ax,x,y,wd,ht,txt,fc='#eef3f9',ec=B,fs=11,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),wd,ht,boxstyle='round,pad=0.02,rounding_size=0.08',fc=fc,ec=ec,lw=1.3))
    ax.text(x+wd/2,y+ht/2,txt,ha='center',va='center',fontsize=fs,fontweight='bold' if bold else 'normal')
def arrow(ax,p,q,c='#444',ls='-'):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=13,color=c,lw=1.2,linestyle=ls))

# F1 attention schematic
fig,ax=plt.subplots(figsize=(9.2,4.2)); ax.axis('off'); ax.set_xlim(0,11.4); ax.set_ylim(0,4)
src=['Você','não','viu','ninguém','?']; tgt=['VOCÊ','NÃO_VER','NINGUÉM','[INTERROGAÇÃO]']
for i,wd in enumerate(src):
    box(ax,0.3+i*0.95,0.3,0.8,0.5,wd,fc='white',ec=G,fs=8.5); box(ax,0.3+i*0.95,1.3,0.8,0.6,f'h{i+1}',fs=11); arrow(ax,(0.7+i*0.95,0.82),(0.7+i*0.95,1.28))
    if i<4: arrow(ax,(1.12+i*0.95,1.6),(1.23+i*0.95,1.6))
ax.text(2.6,2.15,'Codificador (LSTM)',ha='center',fontsize=11,color=B)
box(ax,3.4,3.0,2.2,0.7,'Atenção: soma ponderada\ndos estados h1…h5',fc='#fdf1e2',ec=O,fs=10)
wts=[0.05,0.55,0.33,0.05,0.02]
for i,wt in enumerate(wts):
    ax.add_patch(FancyArrowPatch((0.7+i*0.95,1.92),(4.5,2.98),arrowstyle='-',color=O,lw=0.6+5*wt,alpha=.75))
for j,wd in enumerate(tgt):
    wdt=1.05 if j<3 else 1.6; xx=5.9+sum([1.2,1.2,1.2][:j])
    box(ax,xx,1.3,wdt,0.6,f's{j+1}',fc='#e9f4ec',ec=GR,fs=11); box(ax,xx,0.3,wdt,0.5,wd,fc='white',ec=G,fs=8); arrow(ax,(xx+wdt/2,1.28),(xx+wdt/2,0.82))
    if j<3: arrow(ax,(xx+wdt,1.6),(xx+1.2,1.6))
ax.text(8.4,2.15,'Decodificador (LSTM)',ha='center',fontsize=11,color=GR)
arrow(ax,(5.62,3.2),(7.5,1.95),c=O)
ax.text(6.9,2.95,'contexto para gerar s2',fontsize=10,color=O)
ax.text(2.4,2.75,'pesos maiores em\n"não" e "viu"',fontsize=10,color=O,ha='center')
fig.tight_layout(); fig.savefig('results/figuras/f_atencao.png'); plt.close()

# F2 LoRA schematic
fig,ax=plt.subplots(figsize=(8.2,4)); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,5)
box(ax,0.3,2.1,1.3,0.8,'entrada x',fc='white',ec=G)
box(ax,3.0,3.3,3.2,1.1,'W (pesos pré-treinados)\ncongelados',fc=LG,ec=G,fs=11)
box(ax,2.6,0.6,1.7,1.1,'A\n(r × d)',fc='#eef3f9',ec=B); box(ax,4.9,0.6,1.7,1.1,'B\n(d × r)',fc='#eef3f9',ec=B)
ax.text(4.6,0.2,'matrizes de baixo posto (r = 16): únicas treinadas',ha='center',fontsize=10.5,color=B)
box(ax,7.4,2.1,0.8,0.8,'+',fc='white',ec='#444',fs=16); box(ax,8.7,2.1,1.1,0.8,'saída',fc='white',ec=G)
arrow(ax,(1.62,2.7),(2.98,3.7)); arrow(ax,(1.62,2.3),(2.58,1.3)); arrow(ax,(4.32,1.15),(4.88,1.15)); arrow(ax,(6.22,3.7),(7.45,2.92)); arrow(ax,(6.62,1.3),(7.45,2.1)); arrow(ax,(8.22,2.5),(8.68,2.5))
ax.text(5.0,2.45,'saída = W·x + (α/r)·B·A·x',ha='center',fontsize=12,fontweight='bold')
fig.tight_layout(); fig.savefig('results/figuras/f_lora.png'); plt.close()

# F3 methodology overview
fig,ax=plt.subplots(figsize=(9.2,4.6)); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,5)
cols=[('1. Problema','Conteúdo em\nportuguês sem\ntradução para\nLibras; tradutor\naberto usa regras',O),
      ('2. Dados','VLibrasBD\n127.349 pares\nportuguês-glosa\ndivisão 80/10/10\nestratificada',B),
      ('3. Modelos','Regras\n(vlibras-translate)\nSeq2seq LSTM\n+ atenção\nPTT5-small + LoRA',GR),
      ('4. Avaliação','BLEU e chrF\nglobal,\npor 9 categorias\nlinguísticas e\npor domínio','#7a5aa6'),
      ('5. Aplicação','Prova de conceito\nvídeo/áudio →\nWhisper →\nglosas lado\na lado','#b23a3a')]
for i,(tt,body,c) in enumerate(cols):
    x=0.15+i*1.97
    ax.add_patch(FancyBboxPatch((x,0.5),1.72,3.9,boxstyle='round,pad=0.02,rounding_size=0.1',fc='white',ec=c,lw=1.6))
    ax.add_patch(FancyBboxPatch((x,3.7),1.72,0.7,boxstyle='round,pad=0.02,rounding_size=0.1',fc=c,ec=c))
    ax.text(x+0.86,4.05,tt,ha='center',va='center',color='white',fontsize=11.5,fontweight='bold')
    ax.text(x+0.86,2.1,body,ha='center',va='center',fontsize=10.5,linespacing=1.5)
    if i<4: arrow(ax,(x+1.76,2.3),(x+1.95,2.3))
fig.tight_layout(); fig.savefig('results/figuras/f_metodologia.png'); plt.close()
