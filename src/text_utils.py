"""
Normalização de glosas e categorias linguísticas usadas na avaliação do TCC.

normalizar_glosa(): aplicada à hipótese e à referência antes do BLEU/chrF,
para que as três abordagens sejam comparadas da mesma forma. Preserva os
marcadores [PONTO], [INTERROGAÇÃO], [EXCLAMAÇÃO], "&" e "_", que carregam
informação linguística; uniformiza só caixa e espaçamento.

CATEGORIAS: os 9 fenômenos identificados na auditoria do VLibrasBD, cada um
detectado por expressão regular aplicada à glosa de REFERÊNCIA.

Atenção (registrada no TCC, Seção 3.3): o padrão de "numeral" (\\d) também
casa com os marcadores de pessoa dos verbos direcionais (1S_, 2S_ ...). Os
700 pares com verbo direcional do teste contam também como numeral, e 615
dos 2.005 pares dessa categoria não têm nenhum número além desses
marcadores. O padrão foi mantido assim para reproduzir os números do texto.
"""

import re

CATEGORIAS = {
    "marcador_ponto": r"\[PONTO\]",
    "desambiguacao_amp": r"&",
    "composto_underscore": r"(?<![123][SP])_(?!\d)",
    "numeral": r"\d",
    "negacao_fundida": r"\bNÃO_",
    "interrogacao": r"\[INTERROGAÇÃO\]",
    "verbo_direcional": r"\b[123][SP]_",
    "negacao_isolada": r"\bNÃO\b(?!_)",
    "exclamacao": r"\[EXCLAMAÇÃO\]",
}

ROTULOS = {
    "marcador_ponto": "Marcador de ponto final",
    "desambiguacao_amp": "Desambiguação de sentido",
    "composto_underscore": "Sinal composto",
    "numeral": "Numeral",
    "negacao_fundida": "Negação fundida ao verbo",
    "interrogacao": "Interrogação",
    "verbo_direcional": "Verbo direcional",
    "negacao_isolada": "Negação isolada",
    "exclamacao": "Exclamação",
}


def normalizar_glosa(texto):
    if not isinstance(texto, str):
        return ""
    s = texto.strip().upper()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*\[\s*", " [", s)
    s = re.sub(r"\s*\]\s*", "] ", s)
    return re.sub(r"\s+", " ", s).strip()


def categorias_presentes(glosa_referencia):
    """Retorna {categoria: bool} para uma glosa de referência normalizada."""
    return {nome: bool(re.search(rx, glosa_referencia)) for nome, rx in CATEGORIAS.items()}


def tem_repeticao(glosa, vezes=3):
    """True se a mesma glosa aparece `vezes` ou mais vezes seguidas."""
    return bool(re.search(r"\b(\S+)( \1){%d,}\b" % (vezes - 1), glosa))
