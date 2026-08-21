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
    V_GRUPOS, NOME_FILTROS_F, FILTROS_F_DEFS,
    construir_universo, selecionar_matriz, selecionar_familia,
    aplicar_filtros_f, aplicar_f12_f13_f14, extrair_jogos,
    calcular_top1_top2_numpy, calcular_score_v_numpy,
    aplicar_melhor_combinacao_v_numpy,
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
st.info("🧪 **Versão de teste** elaborado por Seba sob supervisão da equipe Criadora.") 
# ============================================================
# BARRA LATERAL — controles
# ============================================================
with st.sidebar:
    st.header("1. Matriz ou Família")
    modo = st.radio("Modo de seleção", ["Família (ordem ignorada)", "Matriz exata (ordem de linha)"])
 
    if modo.startswith("Família"):
        opcoes = [f"{_fmt(f)}  —  {qtd} concursos ({pct}%)" for f, qtd, pct in TOP5_FAMILIAS] + ["Personalizada"]
        escolha = st.selectbox("Família", opcoes, index=0)
        if escolha == "Personalizada":
            texto = st.text_input("Digite a família (ex: 4/3/3/3/2)", value="4/3/3/3/2")
            alvo = _parse_tupla(texto)
        else:
            idx = opcoes.index(escolha)
            alvo = TOP5_FAMILIAS[idx][0]
    else:
        opcoes = [f"{_fmt(m)}  —  {qtd} concursos ({pct}%)" for m, qtd, pct in TOP5_MATRIZES] + ["Personalizada"]
        escolha = st.selectbox("Matriz", opcoes, index=0)
        if escolha == "Personalizada":
            texto = st.text_input("Digite a matriz exata (ex: 3/3/4/3/2)", value="3/3/4/3/2")
            alvo = _parse_tupla(texto)
        else:
            idx = opcoes.index(escolha)
            alvo = TOP5_MATRIZES[idx][0]
 
    st.header("2. Filtros F (faixas cheias)")
    with st.expander("Filtros base", expanded=False):
        filtros_ativos = {}
        for nome in FILTROS_F_DEFS:
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
 
    minimo_jogos = st.number_input("Mínimo de jogos exigido ao buscar a combinação de V", min_value=1, value=1)
 
    rodar = st.button("▶️ Rodar", type="primary", use_container_width=True)
 
# ============================================================
# ÁREA PRINCIPAL — resultado
# ============================================================
if rodar:
    if not grupos_ativos:
        st.error("Selecione pelo menos um grupo V para continuar.")
        st.stop()
 
    with st.spinner("Aplicando filtros e calculando..."):
        if modo.startswith("Família"):
            mask = selecionar_familia(universo, alvo)
        else:
            mask = selecionar_matriz(universo, alvo)
 
        qtd_apos_selecao = int(mask.sum())
        mask = mask & aplicar_filtros_f(universo, filtros_ativos)
        qtd_apos_f = int(mask.sum())
        mask = mask & aplicar_f12_f13_f14(universo, ativos_estruturais)
        qtd_apos_f12_14 = int(mask.sum())
 
        if qtd_apos_f12_14 == 0:
            st.error("Nenhum jogo sobrou após os filtros F/F12-F14. Tente desmarcar algum filtro.")
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
                universo, mask, grupos=grupos_ativos, minimo_jogos=minimo_jogos,
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()
 
        jogos_finais = extrair_jogos(universo, mask_final)
 
    # ---- Funil ----
    st.subheader("📉 Funil de filtragem")
    df_funil = pd.DataFrame({
        "Etapa": [
            f"{'Família' if modo.startswith('Família') else 'Matriz'} {_fmt(alvo)}",
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

    with st.expander("ℹ️ O que cada critério significa"):
        st.markdown("""
- **5.1 Concentração** — quanto do total de jogos fica concentrado nos 2 valores mais comuns desse grupo.
  Quanto maior, mais "decisivo" é o grupo como filtro.
- **5.2 Faixa curta** — se os 2 valores mais comuns são vizinhos (ex: 5 e 6) ou espalhados. Vizinhos = 1,0
  (faixa mais fácil de justificar); quanto mais distantes, menor o score.
- **5.3 Valor fixo possível** — em quantas das combinações viáveis (testadas por força bruta) esse grupo
  consegue ficar travado num valor único (top_1) sem zerar o resultado junto com os outros grupos.
- **5.4 Não-redundância** — o quanto esse grupo é **diferente** dos outros 10 (baseado na correlação entre
  eles). Grupos muito parecidos entre si desperdiçam poder de filtro; quanto maior, mais "informação nova"
  aquele grupo traz.
- **5.5 Redução** — quantos jogos esse grupo sozinho elimina ao aplicar seu filtro (top-2) sobre a base atual.
- **5.6 Estabilidade** — se o valor mais comum desse grupo se mantém igual comparando a 1ª metade e a 2ª
  metade do histórico real. Se muda de uma metade pra outra, o score é zero (sinal de que pode ser só
  coincidência estatística, não um padrão confiável).
- **Score final** — média dos critérios acima, normalizada entre os grupos avaliados (é uma classificação
  *relativa*: mostra quem se sai melhor dentro desse conjunto, não uma nota absoluta de qualidade).
        """)

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
            "5.1 Concentração": st.column_config.NumberColumn(
                "5.1 Concentração", help="% dos jogos cobertos pelos 2 valores mais comuns desse grupo."),
            "5.2 Faixa curta": st.column_config.NumberColumn(
                "5.2 Faixa curta", help="1,0 = os 2 valores mais comuns são vizinhos. Menor = mais espalhados."),
            "5.3 Valor fixo": st.column_config.NumberColumn(
                "5.3 Valor fixo", help="Fração das combinações viáveis em que esse grupo pôde ficar travado num valor único."),
            "5.4 Não-redundância": st.column_config.NumberColumn(
                "5.4 Não-redundância", help="1 menos a correlação média com os outros grupos. Maior = mais informação nova."),
            "5.5 Redução": st.column_config.NumberColumn(
                "5.5 Redução", help="Fração de jogos eliminada só por esse grupo, sozinho."),
            "5.6 Estabilidade": st.column_config.NumberColumn(
                "5.6 Estabilidade", help="0 se o valor mais comum mudou entre a 1ª e a 2ª metade do histórico real."),
            "Score final": st.column_config.ProgressColumn(
                "Score final", help="Média dos 6 critérios, normalizada entre os grupos avaliados nessa rodada.",
                min_value=0.0, max_value=1.0, format="%.3f",
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
 
    col1, col2 = st.columns(2)
    with col1:
        csv_export = df_jogos.to_csv(index=False, sep=";")
        st.download_button(
            "⬇️ Baixar jogos (.csv)", data=csv_export,
            file_name=f"jogos_{_fmt(alvo).replace('/','-')}.csv",
            mime="text/csv", use_container_width=True,
        )
    with col2:
        txt_export = "\n".join(" ".join(f"{n:02d}" for n in j) for j in jogos_finais)
        st.download_button(
            "⬇️ Baixar jogos (.txt)", data=txt_export,
            file_name=f"jogos_{_fmt(alvo).replace('/','-')}.txt",
            mime="text/plain", use_container_width=True,
        )
else:
    st.info("Configure as opções na barra lateral e clique em **Rodar**.")