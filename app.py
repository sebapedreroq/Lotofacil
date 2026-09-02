#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOTOFÁCIL — INTERFACE WEB (Streamlit)
Tela que consome o motor numpy (lotofacil_numpy_engine.py). Fluxo:
  1. Escolher matriz/família (das 5 mais frequentes, ou personalizada).
  2. Marcar/desmarcar filtros F (default: todos ligados).
  3. Marcar/desmarcar grupos V a considerar (default: todos ligados).
  4. Botão "Rodar" -> mostra funil, combinação de V (fixo/flexível), score e jogos finais.

Para rodar localmente:
    streamlit run app.py
"""

import csv
from pathlib import Path

import pandas as pd
import streamlit as st

from lotofacil_numpy_engine import (
    V_GRUPOS, NOME_FILTROS_F, FILTROS_F_DEFS, FILTROS_AJUSTAVEIS, FAMILIAS_DICA_SEU_JOSE,
    construir_universo, selecionar_matriz, selecionar_familia, selecionar_dica_seu_jose,
    selecionar_matrizes_especificas,
    aplicar_filtros_f, aplicar_f12_f13_f14, extrair_jogos,
    calcular_top1_top2_numpy, calcular_score_v_numpy,
    aplicar_melhor_combinacao_v_numpy,
    listar_matrizes_familia, contar_matrizes_familia,
)

st.set_page_config(page_title="Lotofácil — Ferramenta de Filtros", layout="wide")

# ============================================================
# Listas de referência (calculadas sobre o histórico de 3.748 concursos)
# ============================================================
TOP5_FAMILIAS = [
    ((4, 3, 3, 3, 2), 1177, 31.40),
    ((4, 4, 3, 2, 2), 838, 22.36),
    ((4, 4, 3, 3, 1), 456, 12.17),
    ((5, 3, 3, 2, 2), 344, 9.18),
    ((5, 4, 3, 2, 1), 319, 8.51),
]
TOP5_MATRIZES = [
    ((3, 3, 3, 3, 3), 106, 2.83),
    ((3, 4, 3, 2, 3), 72, 1.92),
    ((3, 3, 4, 2, 3), 72, 1.92),
    ((4, 3, 3, 2, 3), 66, 1.76),
    ((3, 2, 3, 3, 4), 63, 1.68),
]


def _fmt(tupla):
    return "/".join(str(x) for x in tupla)


def _parse_tupla(texto: str):
    partes = [p.strip() for p in texto.split("/")]
    if len(partes) != 5:
        raise ValueError("Precisa ter exatamente 5 números separados por '/', ex: 4/3/3/3/2")
    return tuple(int(p) for p in partes)


# ============================================================
# Universo (carregado/computado uma única vez por sessão do servidor)
# ============================================================
@st.cache_resource(show_spinner="Construindo universo combinatório completo (só na 1ª vez)...")
def carregar_universo():
    return construir_universo(cache_path="universo_lotofacil.npz")


@st.cache_data(show_spinner=False)
def carregar_historico_padrao():
    """Tenta carregar um CSV histórico já presente na pasta do app, se existir."""
    caminho = Path("lotofacil-download-resultados.csv")
    if not caminho.exists():
        return None
    jogos = []
    with open(caminho, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            jogos.append(tuple(sorted(int(x) for x in row[2:])))
    return jogos


def carregar_historico_upload(arquivo) -> list:
    conteudo = arquivo.getvalue().decode("utf-8-sig")
    reader = csv.reader(conteudo.splitlines())
    next(reader)
    jogos = []
    for row in reader:
        jogos.append(tuple(sorted(int(x) for x in row[2:])))
    return jogos


universo = carregar_universo()

st.title("🎯 Lotofácil — Filtros F + V")
st.caption(
    "Ferramenta de organização combinatória para o bolão. Reduz o universo de jogos por matriz/família "
    "e filtros, sem prometer previsão de resultado — sorteios são independentes e aleatórios."
)

# ============================================================
# BARRA LATERAL — controles
# ============================================================
with st.sidebar:
    st.header("1. Matriz, Família ou Dica")
    modo = st.radio(
        "Modo de seleção",
        ["Família (ordem ignorada)", "Matriz exata (ordem de linha)", "🎩 A dica do Seu José"],
    )

    if modo.startswith("Família"):
        opcoes = [f"{_fmt(f)}  —  {qtd} concursos ({pct}%)" for f, qtd, pct in TOP5_FAMILIAS] + ["Personalizada"]
        escolha = st.selectbox("Família", opcoes, index=0)
        if escolha == "Personalizada":
            texto = st.text_input("Digite a família (ex: 4/3/3/3/2)", value="4/3/3/3/2")
            alvo = _parse_tupla(texto)
        else:
            idx = opcoes.index(escolha)
            alvo = TOP5_FAMILIAS[idx][0]
    elif modo.startswith("Matriz"):
        opcoes = [f"{_fmt(m)}  —  {qtd} concursos ({pct}%)" for m, qtd, pct in TOP5_MATRIZES] + ["Personalizada"]
        escolha = st.selectbox("Matriz", opcoes, index=0)
        if escolha == "Personalizada":
            texto = st.text_input("Digite a matriz exata (ex: 3/3/4/3/2)", value="3/3/4/3/2")
            alvo = _parse_tupla(texto)
        else:
            idx = opcoes.index(escolha)
            alvo = TOP5_MATRIZES[idx][0]
    else:
        alvo = None
        st.caption(
            "Combina, de uma vez só, os jogos de **7 famílias fixas** escolhidas pelo Seu José:\n\n"
            + "\n".join(f"- {_fmt(f)}" for f in FAMILIAS_DICA_SEU_JOSE)
        )

    st.header("2. Filtros F (faixas cheias)")
    with st.expander("Filtros base", expanded=False):
        st.caption("F1 (ímpares), F2 (primos) e F4 têm faixa ajustável — os demais são liga/desliga.")

        faixas_customizadas = {}
        filtros_ativos = {}

        lim_impares = FILTROS_AJUSTAVEIS["IMPARES"]
        faixas_customizadas["IMPARES"] = st.slider(
            "F1 — Ímpares (faixa)", min_value=lim_impares[0], max_value=lim_impares[1],
            value=lim_impares, key="faixa_impares",
        )
        lim_primos = FILTROS_AJUSTAVEIS["PRIMOS"]
        faixas_customizadas["PRIMOS"] = st.slider(
            "F2 — Primos (faixa)", min_value=lim_primos[0], max_value=lim_primos[1],
            value=lim_primos, key="faixa_primos",
        )
        lim_f4 = FILTROS_AJUSTAVEIS["F4"]
        faixas_customizadas["F4"] = st.slider(
            "F4 — Bloco 3x5 (faixa)", min_value=lim_f4[0], max_value=lim_f4[1],
            value=lim_f4, key="faixa_f4",
        )

        st.divider()
        for nome in FILTROS_F_DEFS:
            if nome in FILTROS_AJUSTAVEIS:
                continue  # já viraram slider acima
            filtros_ativos[nome] = st.checkbox(nome, value=True, key=f"f_{nome}")
        filtros_ativos["SEM_COLUNA_ZERADA"] = st.checkbox("SEM_COLUNA_ZERADA", value=True, key="f_scz")

    with st.expander("F12 / F13 / F14", expanded=False):
        ativos_estruturais = {
            "F12": st.checkbox("F12 (PA razão 3, 7+ elementos)", value=True, key="f12"),
            "F13": st.checkbox("F13 (bloco 2x3 em linhas)", value=True, key="f13"),
            "F14": st.checkbox("F14 (bloco 2x3 em colunas)", value=True, key="f14"),
        }

    st.header("3. Grupos V a considerar")
    with st.expander("Grupos V", expanded=False):
        grupos_ativos = {}
        for nome in V_GRUPOS:
            if st.checkbox(nome, value=True, key=f"v_{nome}"):
                grupos_ativos[nome] = V_GRUPOS[nome]

    st.header("4. Histórico real (opcional)")
    st.caption("Usado para calibrar o critério 5.3 no histórico e calcular a estabilidade (5.6).")
    arquivo_hist = st.file_uploader("CSV oficial (Concurso, Data, Números)", type="csv")

    st.header("5. Como escolher a combinação de V")
    modo_combinacao = st.radio(
        "Critério de escolha",
        ["Mais restritiva possível", "Aproximar de um número de jogos"],
        help="'Mais restritiva' busca o menor número de jogos possível sem zerar. "
             "'Aproximar de um número' busca, entre todas as combinações viáveis, a que chega mais perto do valor que você quer.",
    )
    alvo_jogos = None
    if modo_combinacao == "Aproximar de um número de jogos":
        alvo_jogos = st.number_input(
            "Número de jogos desejado (ex: 100 para um bolão de até 100 jogos)",
            min_value=1, value=100, step=10,
        )

    minimo_jogos = st.number_input("Mínimo de jogos exigido ao buscar a combinação de V", min_value=1, value=1)

    rodar = st.button("▶️ Rodar", type="primary", use_container_width=True)

# ============================================================
# ÁREA PRINCIPAL — seleção de matrizes ativas (Família / Dica) + resultado
# ============================================================
matrizes_selecionadas: list = []  # usado nos modos "Família" e "🎩 Dica do Seu José"

if modo.startswith("Família"):
    st.subheader(f"📐 Matrizes da família {_fmt(alvo)}")
    st.markdown(
        "Uma **família** é um conjunto de contagens por linha, sem se importar com a ordem. Uma **matriz** "
        "é essa mesma família colocada numa ordem específica entre as 5 linhas da grade (a, b, c, d, e). "
        "O número de matrizes de uma família é:"
    )
    st.latex(r"\text{Nº de matrizes} = \dfrac{5!}{r_1! \times r_2! \times \dots \times r_k!}")
    st.caption("onde cada $r_i$ é quantas vezes um mesmo valor se repete dentro da família.")

    matrizes_familia = listar_matrizes_familia(alvo)
    st.caption(
        f"Essa família tem **{len(matrizes_familia)} matrizes**. Por padrão todas ficam ativas — "
        "desmarque as que não quiser incluir na análise."
    )
    opcoes_matrizes = [_fmt(m) for m in matrizes_familia]
    escolhidas_str = st.multiselect(
        "Matrizes ativas", options=opcoes_matrizes, default=opcoes_matrizes, key="matrizes_familia_ativas",
    )
    matrizes_selecionadas = [matrizes_familia[opcoes_matrizes.index(s)] for s in escolhidas_str]
    st.divider()

elif modo.startswith("🎩"):
    st.subheader("🎩 Sobre a Dica do Seu José")
    st.markdown(
        "Combina os jogos de **7 famílias fixas** escolhidas pelo Seu José. Cada família se desdobra em "
        "várias matrizes (ordens de linha) diferentes — por padrão todas ficam ativas, mas dá pra desmarcar "
        "matrizes específicas de cada família logo abaixo."
    )
    st.latex(r"\text{Nº de matrizes} = \dfrac{5!}{r_1! \times r_2! \times \dots \times r_k!}")

    df_resumo = pd.DataFrame([
        {"Família": _fmt(f), "Nº de matrizes": contar_matrizes_familia(f)}
        for f in FAMILIAS_DICA_SEU_JOSE
    ])
    st.dataframe(df_resumo, hide_index=True, use_container_width=True)

    for familia in FAMILIAS_DICA_SEU_JOSE:
        matrizes_familia = listar_matrizes_familia(familia)
        with st.expander(f"{_fmt(familia)} — {len(matrizes_familia)} matrizes"):
            opcoes = [_fmt(m) for m in matrizes_familia]
            escolhidas_str = st.multiselect(
                f"Matrizes ativas em {_fmt(familia)}", options=opcoes, default=opcoes,
                key=f"matrizes_dica_{_fmt(familia)}",
            )
            matrizes_selecionadas.extend(matrizes_familia[opcoes.index(s)] for s in escolhidas_str)

    st.caption(f"Total de matrizes ativas agora: {len(matrizes_selecionadas)} de {df_resumo['Nº de matrizes'].sum()}.")
    st.divider()

if rodar:
    if not grupos_ativos:
        st.error("Selecione pelo menos um grupo V para continuar.")
        st.stop()
    if modo.startswith("Família") and not matrizes_selecionadas:
        st.error("Selecione pelo menos uma matriz dentro da família.")
        st.stop()
    if modo.startswith("🎩") and not matrizes_selecionadas:
        st.error("Selecione pelo menos uma matriz nas famílias da Dica do Seu José.")
        st.stop()

    with st.spinner("Aplicando filtros e calculando..."):
        if modo.startswith("Família"):
            mask = selecionar_matrizes_especificas(universo, matrizes_selecionadas)
            rotulo_selecao = f"Família {_fmt(alvo)} ({len(matrizes_selecionadas)}/{contar_matrizes_familia(alvo)} matrizes ativas)"
        elif modo.startswith("Matriz"):
            mask = selecionar_matriz(universo, alvo)
            rotulo_selecao = f"Matriz {_fmt(alvo)}"
        else:
            mask = selecionar_matrizes_especificas(universo, matrizes_selecionadas)
            rotulo_selecao = f"🎩 Dica do Seu José ({len(matrizes_selecionadas)} matrizes ativas)"

        qtd_apos_selecao = int(mask.sum())
        mask = mask & aplicar_filtros_f(universo, filtros_ativos, faixas_customizadas)
        qtd_apos_f = int(mask.sum())
        mask = mask & aplicar_f12_f13_f14(universo, ativos_estruturais)
        qtd_apos_f12_14 = int(mask.sum())

        if qtd_apos_f12_14 == 0:
            st.error("Nenhum jogo sobrou após os filtros F/F12-F14. Tente desmarcar algum filtro ou alargar as faixas.")
            st.stop()

        # historico: upload > arquivo padrao na pasta > nenhum
        if arquivo_hist is not None:
            jogos_hist = carregar_historico_upload(arquivo_hist)
        else:
            jogos_hist = carregar_historico_padrao()

        top1_top2_ref = calcular_top1_top2_numpy(universo, mask, grupos_ativos)

        ranking = calcular_score_v_numpy(
            universo, mask, grupos=grupos_ativos, jogos_historicos=jogos_hist,
            minimo_jogos=minimo_jogos,
        )

        try:
            mask_final, melhor, resultados = aplicar_melhor_combinacao_v_numpy(
                universo, mask, grupos=grupos_ativos, minimo_jogos=minimo_jogos, alvo_jogos=alvo_jogos,
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()

    if alvo_jogos is not None:
        diferenca = melhor.qtd_jogos - alvo_jogos
        if diferenca == 0:
            st.success(f"🎯 Bateu exatamente no alvo: {melhor.qtd_jogos} jogos.")
        else:
            sinal = "a mais" if diferenca > 0 else "a menos"
            st.warning(
                f"🎯 Não achei uma combinação exata para {alvo_jogos} jogos — a mais próxima possível deu "
                f"**{melhor.qtd_jogos} jogos** ({abs(diferenca)} {sinal}). "
                f"O conjunto de combinações é discreto (2¹¹), então nem sempre dá pra bater o número exato."
            )

        jogos_finais = extrair_jogos(universo, mask_final)

    # ---- Funil ----
    st.subheader("📉 Funil de filtragem")
    df_funil = pd.DataFrame({
        "Etapa": [
            rotulo_selecao,
            "+ Filtros F selecionados",
            "+ F12/F13/F14 selecionados",
            "+ Melhor combinação de V",
        ],
        "Jogos restantes": [qtd_apos_selecao, qtd_apos_f, qtd_apos_f12_14, melhor.qtd_jogos],
    })
    st.dataframe(df_funil, hide_index=True, use_container_width=True)

    # ---- Combinação de V ----
    st.subheader("🎛️ Combinação de V (fixo = top_1, flexível = top_2)")
    linhas_v = []
    for nome in grupos_ativos:
        tipo = "🔒 Fixo (top_1)" if melhor.escolha[nome] == 1 else "🔓 Flexível (top_2)"
        valores = sorted(top1_top2_ref[nome][melhor.escolha[nome]])
        linhas_v.append({"Grupo": nome, "Tipo": tipo, "Valores aceitos": str(valores)})
    st.dataframe(pd.DataFrame(linhas_v), hide_index=True, use_container_width=True)
    st.caption(f"{melhor.n_top1} de {len(grupos_ativos)} grupos ficaram fixos em valor único.")

    # ---- Score ----
    st.subheader("🏆 Score dos critérios (5.1 a 5.6)")
    df_score = pd.DataFrame([{
        "Grupo": r.grupo,
        "5.1 Concentração": round(r.concentracao, 3),
        "5.2 Faixa curta": round(r.faixa_curta, 3),
        "5.3 Valor fixo": round(r.valor_fixo, 3),
        "5.4 Não-redundância": round(r.nao_redundancia, 3),
        "5.5 Redução": round(r.reducao, 3),
        "5.6 Estabilidade": round(r.estabilidade, 4) if r.estabilidade is not None else "—",
        "Score final": round(r.score_final, 3),
    } for r in ranking])
    st.dataframe(
        df_score,
        hide_index=True, use_container_width=True,
        column_config={
            "Score final": st.column_config.ProgressColumn(
                "Score final", min_value=0.0, max_value=1.0, format="%.3f",
            ),
        },
    )
    if not jogos_hist:
        st.caption("⚠️ Nenhum histórico carregado — o critério 5.6 (estabilidade) ficou de fora do score.")

    # ---- Jogos finais ----
    st.subheader(f"🎲 Jogos finais ({len(jogos_finais)})")
    if len(jogos_finais) > 500:
        st.warning(f"{len(jogos_finais)} jogos é bastante para revisar manualmente — considere apertar mais os filtros.")
    df_jogos = pd.DataFrame(jogos_finais, columns=[f"D{i+1}" for i in range(15)])
    st.dataframe(df_jogos, hide_index=True, use_container_width=True, height=400)

    nome_arquivo = _fmt(alvo).replace("/", "-") if alvo is not None else "dica-seu-jose"
    col1, col2 = st.columns(2)
    with col1:
        csv_export = df_jogos.to_csv(index=False, sep=";")
        st.download_button(
            "⬇️ Baixar jogos (.csv)", data=csv_export,
            file_name=f"jogos_{nome_arquivo}.csv",
            mime="text/csv", use_container_width=True,
        )
    with col2:
        txt_export = "\n".join(" ".join(f"{n:02d}" for n in j) for j in jogos_finais)
        st.download_button(
            "⬇️ Baixar jogos (.txt)", data=txt_export,
            file_name=f"jogos_{nome_arquivo}.txt",
            mime="text/plain", use_container_width=True,
        )
else:
    st.info("Configure as opções na barra lateral e clique em **Rodar**.")
