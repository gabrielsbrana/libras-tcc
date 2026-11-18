# Dados

O corpus não é versionado aqui. Para reproduzir:

1. Baixe o VLibrasBD (`pt_br2libras_gloss.csv`) no Mendeley Data:
   https://data.mendeley.com/datasets/ryj88ckjww/1 (DOI 10.17632/ryj88ckjww.1).
2. Gere a divisão usada no TCC: limpeza de pares vazios e duplicados
   (127.313 pares válidos) e divisão 80/10/10 estratificada por
   `is_government_source`, com colunas `pt`, `glosa` e `is_government_source`.
   Salve em `data/processed/{train,val,test}.csv`.

Tamanhos esperados: 101.850 (treino), 12.731 (validação) e 12.732 (teste;
7.230 geral e 5.502 governamental). O script original de divisão não foi
preservado; `results/predictions_lora_test.csv` contém o conjunto de teste
exato usado na avaliação, com as referências.

Referência: Lima, M. A. et al. VLibrasBD. Data in Brief, v. 62, 111911, 2025.
