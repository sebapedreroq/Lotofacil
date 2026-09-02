#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROJETO LOTOFÁCIL — MOTOR NUMPY (v1)
Complementa lotofacil_v63_com_score.py: mesma lógica de negócio (filtros F,
F12/F13/F14, grupos V, busca de combinações, score), mas com o cálculo pesado
vetorizado em numpy -- pensado para alimentar uma ferramenta web onde o
usuário escolhe matriz/família e filtros ao vivo, sem esperar minutos.

IDEIA CENTRAL:
  O universo completo (C(25,15) = 3.268.760 jogos) é calculado UMA VEZ
  (construir_universo) e cacheado em disco. A partir daí, qualquer escolha de
  matriz, família ou combinação de filtros é apenas indexação/máscara booleana
  sobre arrays já prontos -- rápido o suficiente para rodar a cada clique.

FILTROS F INDIVIDUALIZADOS (checkboxes na futura interface):
  Cada filtro base agora é uma entrada independente em FILTROS_F_DEFS, com
  nome, faixa e o grupo de dezenas envolvido. Ligar/desligar é só incluir ou
  não a chave no dicionário `filtros_ativos` passado para aplicar_filtros_f().

Uso típico:
    universo = construir_universo()                      # ~10-15s, uma vez só
    mask = selecionar_familia(universo, (4,3,3,3,2))
    mask &= aplicar_filtros_f(universo, filtros_ativos=None)   # None = todos ligados
    mask &= aplicar_f12_f13_f14(universo, ativos=None)          # None = todos ligados
    ranking = calcular_score_v_numpy(universo, mask, jogos_historicos=hist)
    jogos_finais = aplicar_melhor_combinacao_v_numpy(universo, mask)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# ============================================================
# GRUPOS DE DEZENAS (mesmos do script principal)
# ============================================================

LINHAS: Dict[str, Set[int]] = {
    "a": {1, 2, 3, 4, 5}, "b": {6, 7, 8, 9, 10}, "c": {11, 12, 13, 14, 15},
    "d": {16, 17, 18, 19, 20}, "e": {21, 22, 23, 24, 25},
}
COLUNAS: Dict[int, Set[int]] = {
    1: {1, 6, 11, 16, 21}, 2: {2, 7, 12, 17, 22}, 3: {3, 8, 13, 18, 23},
    4: {4, 9, 14, 19, 24}, 5: {5, 10, 15, 20, 25},
}
DEZENAS = set(range(1, 26))
PRIMOS = {2, 3, 5, 7, 11, 13, 17, 19, 23}
IMPARES = {n for n in DEZENAS if n % 2 == 1}
C3 = {3, 8, 13, 18, 23}
F3 = {2, 3, 4, 7, 8, 9, 12, 13, 14}
F4 = {2, 3, 4, 7, 8, 9, 12, 13, 14, 17, 18, 19, 22, 23, 24}
F5 = {17, 18, 19, 22, 23, 24}
F6 = {1, 5, 7, 9, 13, 17, 19, 21, 25}
F8 = {2, 9, 12, 19, 22}
F9 = {4, 7, 14, 17, 24}
F10_A = {1, 10, 11, 20, 21}
F10_B = {5, 6, 15, 16, 25}
EXTREMOS = {1, 2, 4, 5, 21, 22, 24, 25}

V_GRUPOS: Dict[str, Set[int]] = {
    "V1": {1, 2, 3, 6, 7, 8, 11, 12, 13}, "V2": {3, 4, 5, 8, 9, 10, 13, 14, 15},
    "V3": {6, 7, 8, 11, 12, 13, 16, 17, 18}, "V4": {8, 9, 10, 13, 14, 15, 18, 19, 20},
    "V5": {2, 3, 4, 7, 8, 9, 12, 13, 14}, "V6": {12, 13, 14, 17, 18, 19, 22, 23, 24},
    "V6a": {7, 8, 9, 12, 13, 14, 17, 18, 19}, "V7": {1, 2, 6, 7, 11, 12},
    "V8": {4, 5, 9, 10, 14, 15}, "V9": {11, 12, 16, 17, 21, 22}, "V10": {14, 15, 19, 20, 24, 25},
}

# ============================================================
# FILTROS F INDIVIDUALIZADOS -- cada um é (grupo_de_dezenas, min, max)
# "min"/"max" None do lado que não se aplica (ex: F8 é só ">= 1", sem teto)
# ============================================================

FILTROS_F_DEFS: Dict[str, Dict] = {
    "C3":         {"grupo": C3,        "min": 0, "max": 3},
    "IMPARES":    {"grupo": IMPARES,   "min": 6, "max": 10},
    "PRIMOS":     {"grupo": PRIMOS,    "min": 3, "max": 7},
    "F3":         {"grupo": F3,        "min": 3, "max": 8},
    "F4":         {"grupo": F4,        "min": 7, "max": 11},
    "F5":         {"grupo": F5,        "min": 1, "max": 6},
    "F6":         {"grupo": F6,        "min": 3, "max": 7},
    "F8":         {"grupo": F8,        "min": 1, "max": None},
    "F9":         {"grupo": F9,        "min": 1, "max": None},
    "F10_A":      {"grupo": F10_A,     "min": 1, "max": None},
    "F10_B":      {"grupo": F10_B,     "min": 1, "max": None},
    "EXTREMOS":   {"grupo": EXTREMOS,  "min": 3, "max": 7},
    # "SEM_COLUNA_ZERADA" é tratado à parte (não é uma contagem única, ver aplicar_filtros_f)
}
NOME_FILTROS_F = list(FILTROS_F_DEFS.keys()) + ["SEM_COLUNA_ZERADA"]

# sequências máximas r=3 (usadas no F12) e blocos 2x3 (F13/F14), calculados uma vez
SEQS_R3: List[Set[int]] = [set(range(1, 26, 3)), set(range(2, 26, 3)), set(range(3, 26, 3))]


def _numero_por_pos(li: int, ci: int) -> int:
    return li * 5 + ci + 1


def _blocos_f13() -> List[Tuple[Set[int], Set[int]]]:
    """Lista de (bloco, complemento) para o filtro F13 (linhas adjacentes)."""
    blocos = []
    for li in range(0, 4):
        linhas_idx = [li, li + 1]
        todas = {_numero_por_pos(l, c) for l in linhas_idx for c in range(5)}
        for c0 in range(0, 3):
            bloco = {_numero_por_pos(l, c) for l in linhas_idx for c in range(c0, c0 + 3)}
            blocos.append((bloco, todas - bloco))
    return blocos


def _blocos_f14() -> List[Tuple[Set[int], Set[int]]]:
    """Lista de (bloco, complemento) para o filtro F14 (colunas adjacentes)."""
    blocos = []
    for c in range(0, 4):
        cols_idx = [c, c + 1]
        todas = {_numero_por_pos(l, cc) for l in range(5) for cc in cols_idx}
        for l0 in range(0, 3):
            bloco = {_numero_por_pos(l, cc) for l in range(l0, l0 + 3) for cc in cols_idx}
            blocos.append((bloco, todas - bloco))
    return blocos


BLOCOS_F13 = _blocos_f13()
BLOCOS_F14 = _blocos_f14()


# ============================================================
# UNIVERSO COMPLETO (calculado uma vez, cacheável em disco)
# ============================================================

@dataclass
class UniversoLotofacil:
    M: np.ndarray                              # (N, 25) bool -- N = C(25,15) = 3.268.760
    cont_linhas: np.ndarray                     # (N, 5) int -- contagem por linha a-e
    cont_filtros_f: Dict[str, np.ndarray]       # nome -> (N,) int, contagem de cada filtro F
    f12_ok: np.ndarray                          # (N,) bool
    f13_ok: np.ndarray                          # (N,) bool
    f14_ok: np.ndarray                          # (N,) bool
    cont_v: Dict[str, np.ndarray]               # nome do grupo V -> (N,) int, contagem por jogo

    @property
    def N(self) -> int:
        return self.M.shape[0]


def _contagem_grupo(M: np.ndarray, grupo: Set[int]) -> np.ndarray:
    return M[:, [d - 1 for d in grupo]].sum(axis=1)


def construir_universo(cache_path: Optional[str] = "universo_lotofacil.npz") -> UniversoLotofacil:
    """
    Constrói (ou carrega do cache) todos os arrays do universo completo.
    Primeira vez: ~15-20s. Com cache em disco: poucos segundos (I/O).
    """
    cache = Path(cache_path) if cache_path else None
    if cache and cache.exists():
        dados = np.load(cache, allow_pickle=False)
        M = dados["M"]
        cont_linhas = dados["cont_linhas"]
        cont_filtros_f = {nome: dados[f"f_{nome}"] for nome in FILTROS_F_DEFS}
        f12_ok, f13_ok, f14_ok = dados["f12_ok"], dados["f13_ok"], dados["f14_ok"]
        cont_v = {nome: dados[f"v_{nome}"] for nome in V_GRUPOS}
        return UniversoLotofacil(M, cont_linhas, cont_filtros_f, f12_ok, f13_ok, f14_ok, cont_v)

    t0 = time.time()
    import itertools
    combos = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(25), 15)), dtype=np.int8)
    combos = combos.reshape(-1, 15)
    N = combos.shape[0]
    M = np.zeros((N, 25), dtype=bool)
    rows = np.repeat(np.arange(N), 15)
    M[rows, combos.reshape(-1)] = True

    cont_linhas = np.stack([_contagem_grupo(M, LINHAS[k]) for k in ["a", "b", "c", "d", "e"]], axis=1)
    cont_filtros_f = {nome: _contagem_grupo(M, defs["grupo"]) for nome, defs in FILTROS_F_DEFS.items()}

    f12_ok = np.ones(N, dtype=bool)
    for seq in SEQS_R3:
        f12_ok &= _contagem_grupo(M, seq) < 7

    f13_ok = np.ones(N, dtype=bool)
    for bloco, comp in BLOCOS_F13:
        tem_bloco = _contagem_grupo(M, bloco) == len(bloco)
        tem_comp_zero = _contagem_grupo(M, comp) == 0
        f13_ok &= ~(tem_bloco & tem_comp_zero)

    f14_ok = np.ones(N, dtype=bool)
    for bloco, comp in BLOCOS_F14:
        tem_bloco = _contagem_grupo(M, bloco) == len(bloco)
        tem_comp_zero = _contagem_grupo(M, comp) == 0
        f14_ok &= ~(tem_bloco & tem_comp_zero)

    cont_v = {nome: _contagem_grupo(M, grupo) for nome, grupo in V_GRUPOS.items()}

    print(f"[construir_universo] {N} jogos processados em {time.time()-t0:.1f}s")

    if cache:
        salvar = {"M": M, "cont_linhas": cont_linhas, "f12_ok": f12_ok, "f13_ok": f13_ok, "f14_ok": f14_ok}
        salvar.update({f"f_{nome}": arr for nome, arr in cont_filtros_f.items()})
        salvar.update({f"v_{nome}": arr for nome, arr in cont_v.items()})
        np.savez_compressed(cache, **salvar)

    return UniversoLotofacil(M, cont_linhas, cont_filtros_f, f12_ok, f13_ok, f14_ok, cont_v)


# ============================================================
# SELEÇÃO DE MATRIZ / FAMÍLIA E FILTROS F INDIVIDUALIZADOS
# ============================================================

def selecionar_matriz(universo: UniversoLotofacil, matriz: Tuple[int, int, int, int, int]) -> np.ndarray:
    """Máscara booleana: jogos cuja contagem por linha bate EXATAMENTE (ordem a-e) com `matriz`."""
    alvo = np.array(matriz)
    return (universo.cont_linhas == alvo).all(axis=1)


def selecionar_familia(universo: UniversoLotofacil, familia: Tuple[int, int, int, int, int]) -> np.ndarray:
    """Máscara booleana: jogos cuja MULTISET de contagem por linha bate com `familia` (ordem ignorada)."""
    alvo = np.sort(np.array(familia))[::-1]
    linhas_ordenadas = -np.sort(-universo.cont_linhas, axis=1)
    return (linhas_ordenadas == alvo).all(axis=1)


# ============================================================
# "A DICA DO SEU JOSÉ" -- conjunto fixo de 7 famílias, definido pela pessoa
# que idealizou a plataforma. A seleção é a UNIÃO (OR) dos jogos que batem
# em QUALQUER uma dessas 7 famílias (cada jogo só pode bater em uma delas,
# já que as famílias são multisets distintos, então não há dupla-contagem).
# ============================================================
FAMILIAS_DICA_SEU_JOSE: List[Tuple[int, int, int, int, int]] = [
    (5, 4, 3, 2, 1),
    (5, 4, 2, 2, 2),
    (5, 3, 3, 3, 1),
    (5, 3, 3, 2, 2),
    (4, 4, 4, 2, 1),
    (4, 4, 3, 2, 2),
    (4, 3, 3, 3, 2),
]


def selecionar_dica_seu_jose(universo: UniversoLotofacil) -> np.ndarray:
    """Máscara booleana: jogos que batem em QUALQUER uma das 7 famílias de FAMILIAS_DICA_SEU_JOSE."""
    mask = np.zeros(universo.N, dtype=bool)
    for familia in FAMILIAS_DICA_SEU_JOSE:
        mask |= selecionar_familia(universo, familia)
    return mask


def listar_matrizes_familia(familia: Tuple[int, int, int, int, int]) -> List[Tuple[int, int, int, int, int]]:
    """Lista todas as matrizes (permutações de linha) distintas que pertencem a uma família."""
    import itertools
    return sorted(set(itertools.permutations(familia)), reverse=True)


def contar_matrizes_familia(familia: Tuple[int, int, int, int, int]) -> int:
    """
    Número de matrizes distintas de uma família = 5! / (produto dos fatoriais
    das repetições). Ex: 4/3/3/3/2 tem o valor 3 repetido 3 vezes -> 5!/3! = 20.
    """
    import math
    from collections import Counter
    repeticoes = Counter(familia)
    denominador = 1
    for qtd in repeticoes.values():
        denominador *= math.factorial(qtd)
    return math.factorial(5) // denominador


# ============================================================
# Filtros F que aceitam faixa min/max ajustável pelo usuário (em vez de só
# ligar/desligar). O limite absoluto de cada um é o que já existe no código
# -- o usuário só pode ESTREITAR a faixa, nunca alargar além disso.
# ============================================================
FILTROS_AJUSTAVEIS: Dict[str, Tuple[int, int]] = {
    "IMPARES": (6, 10),
    "PRIMOS": (3, 7),
    "F4": (7, 11),
}


def aplicar_filtros_f(
    universo: UniversoLotofacil,
    filtros_ativos: Optional[Dict[str, bool]] = None,
    faixas_customizadas: Optional[Dict[str, Tuple[int, int]]] = None,
) -> np.ndarray:
    """
    Combina (AND) os filtros F que estiverem ativos. `filtros_ativos` é um dict
    {nome: True/False}; qualquer filtro OMITIDO é considerado ATIVO por padrão
    (assim a interface web pode mandar só as exceções desmarcadas).
    Nomes válidos: ver NOME_FILTROS_F.

    `faixas_customizadas`: dict {nome: (min, max)} para os filtros listados em
    FILTROS_AJUSTAVEIS (hoje: IMPARES, PRIMOS, F4). A faixa informada é sempre
    RECORTADA (clamp) dentro do limite absoluto original -- nunca alarga além
    do que já existia no código.
    """
    if filtros_ativos is None:
        filtros_ativos = {}
    if faixas_customizadas is None:
        faixas_customizadas = {}
    mask = np.ones(universo.N, dtype=bool)
    for nome, defs in FILTROS_F_DEFS.items():
        if not filtros_ativos.get(nome, True):
            continue
        cont = universo.cont_filtros_f[nome]
        minimo, maximo = defs["min"], defs["max"]
        if nome in FILTROS_AJUSTAVEIS and nome in faixas_customizadas:
            lim_min, lim_max = FILTROS_AJUSTAVEIS[nome]
            custom_min, custom_max = faixas_customizadas[nome]
            minimo = max(lim_min, min(custom_min, lim_max))
            maximo = min(lim_max, max(custom_max, lim_min))
        if minimo is not None:
            mask &= cont >= minimo
        if maximo is not None:
            mask &= cont <= maximo
    if filtros_ativos.get("SEM_COLUNA_ZERADA", True):
        cont_colunas = np.stack([_contagem_grupo(universo.M, COLUNAS[k]) for k in [1, 2, 3, 4, 5]], axis=1)
        mask &= (cont_colunas > 0).all(axis=1)
    return mask


def aplicar_f12_f13_f14(universo: UniversoLotofacil, ativos: Optional[Dict[str, bool]] = None) -> np.ndarray:
    """Combina (AND) F12/F13/F14 conforme ativos (todos ligados por padrão, cada um pode ser desligado)."""
    if ativos is None:
        ativos = {}
    mask = np.ones(universo.N, dtype=bool)
    if ativos.get("F12", True):
        mask &= universo.f12_ok
    if ativos.get("F13", True):
        mask &= universo.f13_ok
    if ativos.get("F14", True):
        mask &= universo.f14_ok
    return mask


def extrair_jogos(universo: UniversoLotofacil, mask: np.ndarray) -> List[Tuple[int, ...]]:
    """Converte uma máscara booleana final em uma lista de jogos (tuplas de dezenas 1-25)."""
    idx = np.where(mask)[0]
    jogos = []
    for i in idx:
        jogos.append(tuple((np.where(universo.M[i])[0] + 1).tolist()))
    return sorted(jogos)


# ============================================================
# TOP_1 / TOP_2 E BUSCA DE COMBINAÇÕES (vetorizado)
# ============================================================

@dataclass
class CombinacaoVNumpy:
    escolha: Dict[str, int]      # nome -> 1 (top_1) ou 2 (top_2)
    n_top1: int
    qtd_jogos: int


def calcular_top1_top2_numpy(
    universo: UniversoLotofacil, mask: np.ndarray, grupos: Optional[Dict[str, Set[int]]] = None
) -> Dict[str, Dict[int, Set[int]]]:
    if grupos is None:
        grupos = V_GRUPOS
    resultado = {}
    for nome in grupos:
        cont = universo.cont_v[nome][mask]
        vals, counts = np.unique(cont, return_counts=True)
        ordem = np.argsort(-counts)
        top1 = {int(vals[ordem[0]])}
        top2 = set(vals[ordem[:2]].tolist())
        resultado[nome] = {1: top1, 2: top2}
    return resultado


def buscar_melhor_combinacao_v_numpy(
    universo: UniversoLotofacil,
    mask: np.ndarray,
    grupos: Optional[Dict[str, Set[int]]] = None,
    minimo_jogos: int = 1,
    top1_top2_ref: Optional[Dict[str, Dict[int, Set[int]]]] = None,
) -> Tuple[List[CombinacaoVNumpy], Dict[str, Dict[int, Set[int]]]]:
    """
    Versão vetorizada de buscar_melhor_combinacao_v: testa as 2^n combinações
    de top_1/top_2 por grupo, mas cada teste é uma operação numpy sobre todo o
    subconjunto de uma vez (em vez de um loop Python por jogo). Escala bem até
    algumas centenas de milhares de jogos na base.
    """
    if grupos is None:
        grupos = V_GRUPOS
    nomes = list(grupos.keys())
    n = len(nomes)

    top1_top2 = top1_top2_ref if top1_top2_ref is not None else calcular_top1_top2_numpy(universo, mask, grupos)

    # pré-computa, uma única vez, os arrays booleanos "esse jogo bate no top_1"
    # e "esse jogo bate no top_2" por grupo -- evita recalcular np.isin 2^n vezes
    passa_top1 = {nome: np.isin(universo.cont_v[nome][mask], list(top1_top2[nome][1])) for nome in nomes}
    passa_top2 = {nome: np.isin(universo.cont_v[nome][mask], list(top1_top2[nome][2])) for nome in nomes}
    n_sub = int(mask.sum())

    resultados: List[CombinacaoVNumpy] = []
    for mascara in range(2 ** n):
        passa = np.ones(n_sub, dtype=bool)
        n_top1 = 0
        for i, nome in enumerate(nomes):
            if (mascara >> i) & 1:
                n_top1 += 1
                passa &= passa_top1[nome]
            else:
                passa &= passa_top2[nome]
        qtd = int(passa.sum())
        if qtd >= minimo_jogos:
            escolha = {nomes[i]: (1 if (mascara >> i) & 1 else 2) for i in range(n)}
            resultados.append(CombinacaoVNumpy(escolha=escolha, n_top1=n_top1, qtd_jogos=qtd))

    resultados.sort(key=lambda r: (-r.n_top1, -r.qtd_jogos))
    return resultados, top1_top2


def escolher_combinacao_por_alvo(
    resultados: List[CombinacaoVNumpy], alvo_jogos: int
) -> Tuple[CombinacaoVNumpy, int]:
    """
    Entre as combinações já calculadas por buscar_melhor_combinacao_v_numpy,
    escolhe a que tem qtd_jogos mais próxima de `alvo_jogos`.

    Retorna (combinação escolhida, diferença absoluta em relação ao alvo).
    Como o conjunto de combinações é discreto (2^n opções), não há garantia de
    bater exatamente no alvo -- a diferença retornada informa o quão perto se
    chegou, para a interface poder avisar o usuário quando o desvio for grande.
    """
    if not resultados:
        raise ValueError("Nenhuma combinação viável para escolher.")
    melhor = min(resultados, key=lambda r: abs(r.qtd_jogos - alvo_jogos))
    return melhor, abs(melhor.qtd_jogos - alvo_jogos)


def aplicar_melhor_combinacao_v_numpy(
    universo: UniversoLotofacil,
    mask: np.ndarray,
    grupos: Optional[Dict[str, Set[int]]] = None,
    minimo_jogos: int = 1,
    top1_top2_ref: Optional[Dict[str, Dict[int, Set[int]]]] = None,
    alvo_jogos: Optional[int] = None,
) -> Tuple[np.ndarray, CombinacaoVNumpy, List[CombinacaoVNumpy]]:
    """
    Atalho: roda a busca e já devolve a máscara final filtrada pela combinação
    escolhida.

    Se `alvo_jogos` for None (padrão): escolhe a combinação MAIS RESTRITIVA
    que ainda deixa jogos >= minimo_jogos (comportamento original).

    Se `alvo_jogos` for informado: escolhe, entre as combinações viáveis, a
    que tem quantidade de jogos mais PRÓXIMA de `alvo_jogos` (não precisa ser
    exata -- o conjunto de combinações é discreto).
    """
    if grupos is None:
        grupos = V_GRUPOS
    resultados, top1_top2 = buscar_melhor_combinacao_v_numpy(universo, mask, grupos, minimo_jogos, top1_top2_ref)
    if not resultados:
        raise ValueError(
            f"Nenhuma combinação de top_1/top_2 deixou jogos suficientes (mínimo: {minimo_jogos})."
        )
    if alvo_jogos is None:
        melhor = resultados[0]
    else:
        melhor, _diff = escolher_combinacao_por_alvo(resultados, alvo_jogos)
    mask_final = mask.copy()
    for nome in grupos:
        valores_aceitos = top1_top2[nome][melhor.escolha[nome]]
        mask_final &= np.isin(universo.cont_v[nome], list(valores_aceitos))
    return mask_final, melhor, resultados


# ============================================================
# SCORE DOS CRITÉRIOS 5.1-5.6 (vetorizado)
# ============================================================

@dataclass
class ScoreCriteriosVNumpy:
    grupo: str
    concentracao: float
    faixa_curta: float
    valor_fixo: float
    nao_redundancia: float
    reducao: float
    estabilidade: Optional[float]
    score_final: float


def _normalizar_min_max(valores: Dict[str, float]) -> Dict[str, float]:
    vals = [v for v in valores.values() if v is not None]
    if not vals:
        return {k: 0.0 for k in valores}
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return {k: (1.0 if v is not None else None) for k, v in valores.items()}
    return {k: ((v - mn) / (mx - mn) if v is not None else None) for k, v in valores.items()}


def calcular_estabilidade_historica_numpy(
    jogos_historicos: Sequence[Tuple[int, ...]], grupos: Optional[Dict[str, Set[int]]] = None
) -> Dict[str, float]:
    if grupos is None:
        grupos = V_GRUPOS
    metade = len(jogos_historicos) // 2
    primeira, segunda = jogos_historicos[:metade], jogos_historicos[metade:]

    def _cont(jogos_parte, grupo):
        return np.array([len(set(j) & grupo) for j in jogos_parte])

    def _top1_margem(jogos_parte, grupo):
        cont = _cont(jogos_parte, grupo)
        vals, counts = np.unique(cont, return_counts=True)
        ordem = np.argsort(-counts)
        total = counts.sum()
        top1 = int(vals[ordem[0]])
        margem = (counts[ordem[0]] - counts[ordem[1]]) / total if len(ordem) > 1 else 1.0
        return top1, margem

    resultado = {}
    for nome, grupo in grupos.items():
        t1a, ma = _top1_margem(primeira, grupo)
        t1b, mb = _top1_margem(segunda, grupo)
        resultado[nome] = ((ma + mb) / 2) if t1a == t1b else 0.0
    return resultado


def calcular_score_v_numpy(
    universo: UniversoLotofacil,
    mask: np.ndarray,
    grupos: Optional[Dict[str, Set[int]]] = None,
    jogos_historicos: Optional[Sequence[Tuple[int, ...]]] = None,
    top1_top2_ref: Optional[Dict[str, Dict[int, Set[int]]]] = None,
    pesos: Optional[Dict[str, float]] = None,
    minimo_jogos: int = 1,
) -> List[ScoreCriteriosVNumpy]:
    if grupos is None:
        grupos = V_GRUPOS
    nomes = list(grupos.keys())
    total_base = int(mask.sum())

    top1_top2_proprio = calcular_top1_top2_numpy(universo, mask, grupos)
    top2 = {nome: top1_top2_proprio[nome][2] for nome in nomes}

    # 5.1 concentração
    score_51 = {}
    for nome in nomes:
        cont = universo.cont_v[nome][mask]
        score_51[nome] = float(np.isin(cont, list(top2[nome])).sum()) / total_base

    # 5.2 faixa consecutiva mais curta
    score_52 = {}
    for nome in nomes:
        vals = sorted(top2[nome])
        largura = (vals[-1] - vals[0]) if len(vals) == 2 else 1
        score_52[nome] = 1.0 / largura if largura > 0 else 1.0

    # 5.3 valor fixo possível (busca de combinações; referência própria ou externa)
    resultados_busca, _ = buscar_melhor_combinacao_v_numpy(universo, mask, grupos, minimo_jogos, top1_top2_ref)
    total_viaveis = len(resultados_busca)
    freq_top2 = {nome: 0 for nome in nomes}
    for r in resultados_busca:
        for nome in nomes:
            if r.escolha[nome] == 2:
                freq_top2[nome] += 1
    score_53 = {nome: (1 - freq_top2[nome] / total_viaveis) if total_viaveis else 0.0 for nome in nomes}

    # 5.4 não-redundância (correlação de Pearson vetorizada)
    conts = {nome: universo.cont_v[nome][mask].astype(float) for nome in nomes}
    score_54 = {}
    for a in nomes:
        xa = conts[a]
        xa_c = xa - xa.mean()
        corrs = []
        for b in nomes:
            if a == b:
                continue
            xb = conts[b]
            xb_c = xb - xb.mean()
            denom = np.sqrt((xa_c ** 2).sum()) * np.sqrt((xb_c ** 2).sum())
            r = float((xa_c * xb_c).sum() / denom) if denom > 0 else 0.0
            corrs.append(abs(r))
        score_54[a] = 1 - (sum(corrs) / len(corrs))

    # 5.5 redução (fração eliminada só pelo top-2 desse grupo)
    score_55 = {nome: 1 - score_51[nome] for nome in nomes}  # equivalente: sobrevive = cobertura do top-2

    # 5.6 estabilidade (só histórico)
    if jogos_historicos:
        score_56 = calcular_estabilidade_historica_numpy(jogos_historicos, grupos)
    else:
        score_56 = {nome: None for nome in nomes}

    brutos = {"5.1": score_51, "5.2": score_52, "5.3": score_53, "5.4": score_54, "5.5": score_55, "5.6": score_56}
    normalizados = {chave: _normalizar_min_max(valores) for chave, valores in brutos.items()}
    if pesos is None:
        pesos = {chave: 1.0 for chave in brutos}

    resultado: List[ScoreCriteriosVNumpy] = []
    for nome in nomes:
        itens = [(chave, normalizados[chave][nome]) for chave in brutos if normalizados[chave][nome] is not None]
        soma_pesos = sum(pesos[chave] for chave, _ in itens)
        score_final = sum(pesos[chave] * v for chave, v in itens) / soma_pesos if soma_pesos else 0.0
        resultado.append(ScoreCriteriosVNumpy(
            grupo=nome, concentracao=score_51[nome], faixa_curta=score_52[nome],
            valor_fixo=score_53[nome], nao_redundancia=score_54[nome], reducao=score_55[nome],
            estabilidade=score_56[nome], score_final=score_final,
        ))
    resultado.sort(key=lambda r: -r.score_final)
    return resultado
