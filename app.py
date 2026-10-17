"""Arboreser — apoio à decisão de arborização urbana (Ribeirão Preto/SP)."""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import altair as alt
import folium
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

import logic as L
import ml_extra as MX

st.set_page_config(
    page_title="Arboreser — Arborização baseada em dados",
    page_icon="🌳",
    layout="wide",
    menu_items={"About": "Associação Arboreser — apoio à decisão de plantio em Ribeirão Preto/SP."},
)

VERDE, VERDE_ESCURO = "#50A041", "#2F6B27"
PORTES_ORD = ["Pequeno", "Médio", "Grande"]
COR_PORTE = {"Pequeno": "#7CB9E8", "Médio": "#F2B134", "Grande": "#C0392B"}
CENTRO_RP = [-21.18, -47.81]


def _logo_b64() -> str | None:
    p = Path(__file__).parent / "logo.jpg"
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else None


def cabecalho():
    b64 = _logo_b64()
    img = (f'<img src="data:image/jpeg;base64,{b64}" style="height:64px;border-radius:50%;">' if b64 else "🌳")
    st.markdown(
        f"""
        <style>
        #MainMenu, footer {{visibility: hidden;}}
        .block-container {{padding-top: 1.5rem;}}
        .cab {{background: linear-gradient(90deg,{VERDE_ESCURO},{VERDE}); padding:1rem 1.5rem;
               border-radius:12px; margin-bottom:1.2rem; display:flex; align-items:center; gap:1rem;}}
        .cab h1 {{color:#fff; margin:0; font-size:1.55rem; line-height:1.2;}}
        .cab p {{color:#E6F4E2; margin:.2rem 0 0; font-size:.92rem;}}
        .selo {{display:inline-block; padding:.25rem .8rem; border-radius:999px; color:#fff;
                font-weight:600; font-size:1.05rem;}}
        div[data-testid="stMetric"] {{background:#F1F7EF; padding:.8rem 1rem; border-radius:10px;}}
        </style>
        <div class="cab">{img}<div>
          <h1>Arboreser · Recomendador de arborização</h1>
          <p>Ribeirão Preto / SP — onde e o que plantar, com diversidade e segurança</p>
        </div></div>
        """,
        unsafe_allow_html=True,
    )


def pct(x: float, casas: int = 1) -> str:
    return f"{x * 100:.{casas}f}%".replace(".", ",")


def num(x: float) -> str:
    return f"{int(x):,}".replace(",", ".")


def selo_porte(porte: str) -> str:
    return f'<span class="selo" style="background:{COR_PORTE[porte]}">{porte}</span>'


def nome_esp(r) -> str:
    return f"{r['especie']} ({r['nome_popular']})" if r["nome_popular"] else r["especie"]


@st.cache_resource(show_spinner="Carregando dados e treinando o modelo…")
def base() -> L.Base:
    return L.carregar_base()


def tabela_especies(df: pd.DataFrame):
    out = df[["especie", "nome_popular", "pontuacao", "historico", "justificativa"]].rename(columns={
        "especie": "Espécie", "nome_popular": "Nome popular", "pontuacao": "Pontuação",
        "historico": "Histórico local", "justificativa": "Por que esta espécie",
    })
    st.dataframe(
        out, hide_index=True, width="stretch",
        column_config={"Pontuação": st.column_config.ProgressColumn(
            "Pontuação", min_value=0, max_value=100, format="%.0f"),
            "Por que esta espécie": st.column_config.TextColumn(width="large")},
    )


def baixar(df: pd.DataFrame, nome: str, chave: str):
    c1, c2, _ = st.columns([1, 1, 4])
    c1.download_button("⬇️ CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       f"{nome}.csv", "text/csv", key=f"csv_{chave}")
    buf = BytesIO()
    df.to_excel(buf, index=False, sheet_name="plano")
    c2.download_button("⬇️ Excel", buf.getvalue(), f"{nome}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       key=f"xlsx_{chave}")


def mapa_base(lat, lon, zoom=14) -> folium.Map:
    return folium.Map(location=[lat, lon], zoom_start=zoom, tiles="OpenStreetMap", control_scale=True)



def form_condicoes(chave: str, submit: bool = True) -> dict:
    """Campos das condições do local (usado na página 2 e no laboratório)."""
    f1, f2, f3 = st.columns(3)
    larg = f1.number_input("Largura da calçada (cm)", 100, 800, 250, 10, key=f"larg_{chave}")
    tf = f1.selectbox("Fiação elétrica", L.OPCOES["tipo_fiacao"], key=f"tf_{chave}")
    prot = f2.selectbox("Proteção da fiação", L.OPCOES["protecao"], key=f"prot_{chave}")
    tel = f2.selectbox("Telefonia / TV a cabo", L.OPCOES["telefonia"], key=f"tel_{chave}")
    ed = f3.selectbox("Edificação", L.OPCOES["edificacao"], key=f"ed_{chave}")
    te = f3.selectbox("Tipo de edificação", L.OPCOES["tipo_edificacao"], key=f"te_{chave}")
    rec = f3.selectbox("Recuo", L.OPCOES["recuo"], key=f"rec_{chave}")
    return dict(larg=larg, tipo_fiacao=tf, protecao=("Não se aplica" if tf == "Nenhum" else prot),
                telefonia=tel, edificacao=ed, tipo_edificacao=te, recuo=rec)


def mostrar_what_if(b: L.Base, entrada: dict):
    st.caption("Muda **uma condição por vez** e mostra o que acontece com o porte recomendado. "
               "Serve para entender o que o modelo aprendeu — não é garantia física: o modelo "
               "imita o critério da equipe de campo e pode não ser monotônico (ex.: calçada mais larga "
               "nem sempre aumenta o porte).")
    w = L.analise_what_if(b.modelo, entrada)
    mudam = w[w["muda?"]]
    if mudam.empty:
        st.success("Nenhuma mudança isolada altera o porte recomendado: a recomendação é estável.")
    else:
        st.write(f"**{len(mudam)} de {len(w)}** mudanças isoladas alteram o porte:")
        st.dataframe(mudam.drop(columns="muda?"), hide_index=True, width="stretch",
                     column_config={"P(Grande)": st.column_config.NumberColumn(format="percent")})


def pagina_panorama(b: L.Base):
    st.subheader("Panorama da arborização")
    sat = L.especies_saturadas(b)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Árvores inventariadas", num(len(b.vivas)))
    c2.metric("Pontos de déficit", num(len(b.deficit)))
    c3.metric("Árvores mortas a repor", num(len(b.arvores) - len(b.vivas)))
    c4.metric("Espécies no inventário", num(b.vivas["especie"].nunique()))

    st.markdown("#### Diversidade: regra 10-20-30")
    st.caption("Referência de manejo: nenhuma espécie deve passar de 10% da arborização, "
               "nenhum gênero de 20%, nenhuma família de 30%.")
    top = b.concentracao[b.concentracao["nivel"] == "especie"].nlargest(10, "share").copy()
    nomes = b.vivas.groupby("especie")["nome_popular"].agg(lambda s: s.dropna().mode().iat[0] if s.notna().any() else "")
    top["rotulo"] = top["nome"].map(lambda n: f"{n} ({nomes.get(n, '')})" if nomes.get(n, "") else n)
    top["acima"] = top["share"] >= L.LIM_ESPECIE
    barras = alt.Chart(top).mark_bar().encode(
        x=alt.X("share:Q", title="% das árvores da cidade", axis=alt.Axis(format="%")),
        y=alt.Y("rotulo:N", sort="-x", title=None),
        color=alt.condition("datum.acima", alt.value("#C0392B"), alt.value(VERDE)),
        tooltip=["nome", alt.Tooltip("n", title="árvores"), alt.Tooltip("share", format=".1%")],
    )
    regra = alt.Chart(pd.DataFrame({"x": [L.LIM_ESPECIE]})).mark_rule(strokeDash=[5, 4], color="#444").encode(x="x:Q")
    st.altair_chart((barras + regra).properties(height=300), width="stretch")
    esp_sat = sat[sat["nivel"] == "especie"]
    for _, r in esp_sat.iterrows():
        st.warning(f"**{r['nome']}** representa **{pct(r['share'])}** das árvores — mais que o dobro do teto de 10%. "
                   "O app **não recomenda** novos plantios dessa espécie.")

    st.markdown("#### Setores: onde o déficit pesa mais")
    st.caption("*Déficit relativo* = pontos sem árvore ÷ (pontos sem árvore + árvores existentes). "
               "*Dominância* = % da espécie mais comum no setor.")
    r = L.resumo_setores(b)
    st.dataframe(
        r[["setor", "déficit total", "déficit pequeno", "déficit médio", "déficit grande", "árvores",
           "árvores mortas", "déficit relativo", "espécie dominante", "dominância"]],
        hide_index=True, width="stretch",
        column_config={
            "déficit relativo": st.column_config.ProgressColumn("déficit relativo", min_value=0, max_value=1, format="percent"),
            "dominância": st.column_config.NumberColumn("dominância", format="percent"),
        },
    )
    baixar(r, "resumo_setores", "setores")



def pagina_especie_para_local(b: L.Base):
    st.subheader("Que espécie plantar neste local?")
    c1, c2 = st.columns([1, 2])
    with c1:
        setor = st.selectbox("Setor", sorted(b.deficit["setor"].unique()))
        modo = st.radio("Como descrever o local?",
                        ["Ponto já levantado (clique no mapa)", "Local novo (informar condições)"])
        perfil = st.selectbox("Prioridade da recomendação", list(L.PERFIS))
        with st.expander("Ajustes avançados"):
            raio = st.slider("Raio de vizinhança (m)", 30, 250, L.RAIO_PADRAO_M, 10,
                             help="Árvores dentro deste raio entram na conta de diversidade local.")
            sem_hist = st.checkbox("Incluir espécies sem histórico no inventário", value=True)

    pts = b.deficit[b.deficit["setor"] == setor]
    with c2:
        st.caption("Clique no mapa para indicar o local. Pontos coloridos = déficit levantado "
                   "(azul pequeno · amarelo médio · vermelho grande).")
        m = mapa_base(pts["lat"].mean(), pts["lon"].mean(), 15)
        for _, p in pts.iterrows():
            folium.CircleMarker([p["lat"], p["lon"]], radius=3, color=COR_PORTE[p["porte"]],
                                fill=True, fill_opacity=.8, weight=1).add_to(m)
        ret = st_folium(m, height=380, width=None, returned_objects=["last_clicked"], key=f"mapa_{setor}")

    clique = ret.get("last_clicked") if ret else None
    if not clique:
        st.info("👆 Clique no mapa para escolher o local e ver a recomendação.")
        return
    e, n = L.latlon_para_utm(clique["lat"], clique["lng"])
    viz = L.vizinhanca(b, e, n, raio)

    if modo.startswith("Ponto"):
        ponto, dist = L.ponto_mais_proximo(b, setor, e, n)
        if ponto is None:
            st.warning(f"O ponto levantado mais próximo está a {dist:.0f} m. Clique mais perto de um ponto colorido.")
            return
        porte = ponto["porte"]
        st.markdown(f"**Condições levantadas em campo:** {ponto['condicoes']}  \n"
                    f"**Elementos urbanos:** {ponto['elementos_urbanos']} · **Canteiro livre:** {ponto['canteiro_cm']:.0f} cm")
        st.markdown(f"Porte indicado no levantamento: {selo_porte(porte)}", unsafe_allow_html=True)
        e, n = ponto["E"], ponto["N"]
        viz = L.vizinhanca(b, e, n, raio)
    else:
        st.markdown("**Condições do local**")
        entrada = form_condicoes("pg2")
        porte, probs = L.prever_porte(b.modelo, entrada)
        st.markdown(f"Porte recomendado: {selo_porte(porte)}", unsafe_allow_html=True)
        st.caption("Probabilidades do modelo — " + " · ".join(f"{k}: {pct(v, 0)}" for k, v in probs.items())
                   + f". O app só sobe de porte quando a confiança é ≥ {pct(L.TAU_PADRAO, 0)} "
                     "(árvore grande demais é o erro mais caro).")
        cant = MX.prever_canteiro(b.canteiro, entrada, L.codificar)
        st.markdown(f"Canteiro livre estimado: **{cant:.0f} cm** "
                    f"<span style='color:#666'>(regressão treinada nos {num(b.metricas['extra']['canteiro']['n'])} "
                    "pontos levantados)</span>", unsafe_allow_html=True)
        with st.expander("E se as condições fossem outras? (análise contrafactual)"):
            mostrar_what_if(b, entrada)

    st.caption(f"{len(viz)} árvores num raio de {raio} m deste ponto.")
    recs = L.ranquear_especies(b, porte, viz, perfil, sem_hist, top_n=10)
    if recs.empty:
        st.warning("Nenhuma espécie elegível com estes filtros.")
        return
    st.markdown(f"##### Top {len(recs)} espécies de porte {porte.lower()}")
    tabela_especies(recs)
    with st.expander("🔁 Espécies parecidas com a 1ª colocada (alternativas se faltar muda)"):
        alt_ = MX.especies_parecidas(b.especies, b.vivas, recs.iloc[0]["especie"], n=6)
        st.caption("Similaridade (cosseno) combinando família/gênero/tamanho observado e onde a espécie "
                   "ocorre na cidade. Só entram espécies do mesmo porte e fora dos limites 10-20-30.")
        st.dataframe(alt_, hide_index=True, width="stretch",
                     column_config={"similaridade": st.column_config.ProgressColumn(
                         "similaridade", min_value=0, max_value=1, format="%.2f")})
    with st.expander("Espécies que o app evita recomendar"):
        sat = b.especies[b.especies["saturada"]]
        st.write(", ".join(nome_esp(r) for _, r in sat.iterrows()) or "Nenhuma.")
        st.caption("Passaram dos limites 10-20-30 da arborização atual.")


def pagina_local_para_especie(b: L.Base):
    st.subheader("Onde plantar esta espécie?")
    cat = b.especies.sort_values("especie")
    rotulos = {r["especie"]: nome_esp(r) for _, r in cat.iterrows()}
    c1, c2 = st.columns([2, 1])
    esp = c1.selectbox("Espécie", list(rotulos), format_func=rotulos.get)
    maior = c2.checkbox("Aceitar locais de porte maior", value=False,
                        help="Uma espécie pequena cabe em local grande, mas desperdiça espaço de árvore grande.")
    r = cat[cat["especie"] == esp].iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"Porte<br>{selo_porte(r['porte'])}", unsafe_allow_html=True)
    k2.metric("Árvores no inventário", num(r["n_inventario"]))
    k3.metric("% da arborização", pct(r["share_especie"]))
    k4.metric("Em bom estado", pct(1 - r["pct_problema"], 0) if r["n_inventario"] >= 20 else "sem histórico")
    if r["saturada"]:
        st.error("Esta espécie (ou seu gênero) já ultrapassa o limite de diversidade. **Não recomendamos novos plantios.**")
    elif r["n_inventario"] == 0:
        st.info("Espécie ainda não usada na cidade: ótima para diversidade, mas sem histórico local de desempenho.")

    if r["saturada"]:
        sub = MX.especies_parecidas(b.especies, b.vivas, esp, n=6)
        st.markdown("**Substitutas sugeridas (mesmo porte, fora do limite de diversidade):**")
        st.dataframe(sub, hide_index=True, width="stretch")
    pts, resumo = L.locais_para_especie(b, esp, maior)
    if resumo.empty:
        st.warning("Sem pontos de déficit compatíveis com o porte desta espécie.")
        return
    ordem = st.radio("Ordenar setores por", ["Mais pontos livres", "Setores mais dominados por uma só espécie"],
                     horizontal=True)
    resumo = resumo.sort_values("pontos_livres" if ordem.startswith("Mais") else "dominancia", ascending=False)
    st.caption("*Pontos livres* = nenhuma árvore da mesma espécie ou gênero a menos de 100 m.")
    sel = st.dataframe(
        resumo.rename(columns={"pontos_compativeis": "pontos compatíveis", "pontos_livres": "pontos livres",
                               "arvores_da_especie": "já tem desta espécie", "dominancia": "dominância",
                               "dominante": "espécie dominante"}),
        hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
        column_config={"dominância": st.column_config.NumberColumn(format="percent")},
    )
    linhas = sel.selection.rows if sel and sel.selection else []
    setor = resumo.iloc[linhas[0]]["setor"] if linhas else resumo.iloc[0]["setor"]
    st.markdown(f"##### Pontos sugeridos no setor **{setor}**")
    st.caption("Clique numa linha da tabela acima para trocar de setor. Os pontos mantêm ≥ 40 m de distância "
               "entre si para não concentrar a espécie.")
    qtd = st.slider("Quantos pontos", 5, 50, 15)
    livres = pts[(pts["setor"] == setor) & pts["livre"]]
    escolhidos = L.espalhar(livres, qtd, 40.0)
    if escolhidos.empty:
        st.warning("Este setor não tem pontos livres para a espécie.")
        return
    m = mapa_base(escolhidos["lat"].mean(), escolhidos["lon"].mean(), 15)
    cl = MarkerCluster().add_to(m)
    for _, p in escolhidos.iterrows():
        folium.Marker([p["lat"], p["lon"]], tooltip=p["condicoes"],
                      icon=folium.Icon(color="green", icon="tree", prefix="fa")).add_to(cl)
    st_folium(m, height=420, width=None, returned_objects=[], key=f"mapa_loc_{esp}_{setor}")
    out = escolhidos[["setor", "lat", "lon", "porte", "condicoes"]].copy()
    out["google_maps"] = out.apply(lambda p: f"https://www.google.com/maps?q={p['lat']:.6f},{p['lon']:.6f}", axis=1)
    baixar(out, f"locais_{esp.replace(' ', '_')}_{setor}", "locais")



def pagina_plano(b: L.Base):
    st.subheader("Plano de plantio por setor")
    st.caption("Gera uma lista pronta para campo: cada ponto recebe uma espécie compatível com o porte, "
               "sem repetir espécie/gênero na vizinhança e respeitando um teto por espécie no plano.")
    c1, c2, c3, c4 = st.columns(4)
    setor = c1.selectbox("Setor", sorted(b.deficit["setor"].unique()))
    qtd = c2.slider("Nº de árvores", 10, 300, 60, 10)
    teto = c3.slider("Teto por espécie no plano", 5, 25, 10, 1, format="%d%%") / 100
    perfil = c4.selectbox("Prioridade", list(L.PERFIS))
    sem_hist = st.checkbox("Incluir espécies sem histórico no inventário", value=True)

    if st.button("Gerar plano", type="primary"):
        with st.spinner("Calculando o plano…"):
            st.session_state["plano"] = (setor, L.gerar_plano(b, setor, qtd, perfil, teto,
                                                              incluir_sem_historico=sem_hist))
    if "plano" not in st.session_state:
        return
    setor_p, plano = st.session_state["plano"]
    if plano.empty:
        st.warning("Não foi possível montar o plano com estes parâmetros.")
        return

    cont = plano["especie"].value_counts()
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Árvores no plano ({setor_p})", num(len(plano)))
    k2.metric("Espécies diferentes", num(len(cont)))
    k3.metric("Maior participação", f"{pct(cont.iloc[0] / len(plano))} · {cont.index[0]}")

    g1, g2 = st.columns([1, 1])
    with g1:
        graf = cont.rename_axis("espécie").reset_index(name="árvores")
        st.altair_chart(alt.Chart(graf).mark_bar(color=VERDE).encode(
            x="árvores:Q", y=alt.Y("espécie:N", sort="-x", title=None)).properties(height=max(200, 22 * len(graf))),
            width="stretch")
    with g2:
        m = mapa_base(plano["lat"].mean(), plano["lon"].mean(), 15)
        for _, p in plano.iterrows():
            folium.CircleMarker([p["lat"], p["lon"]], radius=6, color=COR_PORTE[p["porte_do_local"]],
                                fill=True, fill_opacity=.85,
                                tooltip=f"{p['especie']} ({p['porte_do_local']})").add_to(m)
        st_folium(m, height=max(300, 22 * len(graf)), width=None, returned_objects=[], key="mapa_plano")
    st.markdown("#### 🧭 Equipes e rotas (K-Means + roteirização)")
    e1, e2 = st.columns(2)
    n_eq = e1.slider("Nº de equipes", 1, 8, 2)
    por_dia = e2.slider("Mudas por equipe por dia", 5, 60, 20, 5)
    plano_r, res_eq = MX.dividir_equipes(plano, n_eq, por_dia)
    st.dataframe(res_eq, hide_index=True, width="stretch")
    cores = ["#C0392B", "#2980B9", "#27AE60", "#8E44AD", "#E67E22", "#16A085", "#7F8C8D", "#2C3E50"]
    mr = mapa_base(plano_r["lat"].mean(), plano_r["lon"].mean(), 15)
    for eq, g in plano_r.groupby("equipe"):
        cor = cores[(eq - 1) % len(cores)]
        folium.PolyLine(g[["lat", "lon"]].values.tolist(), color=cor, weight=3, opacity=.7).add_to(mr)
        for _, p in g.iterrows():
            folium.CircleMarker([p["lat"], p["lon"]], radius=5, color=cor, fill=True, fill_opacity=.9,
                                tooltip=f"Equipe {eq} · dia {p['dia']} · #{p['ordem']} · {p['especie']}").add_to(mr)
    st_folium(mr, height=420, width=None, returned_objects=[], key=f"mapa_rota_{n_eq}_{por_dia}_{len(plano_r)}")
    saida = plano_r.drop(columns=["E", "N"])
    st.dataframe(saida, hide_index=True, width="stretch",
                 column_config={"google_maps": st.column_config.LinkColumn("mapa", display_text="abrir")})
    baixar(saida, f"plano_plantio_{setor_p}", "plano")


def pagina_mapa(b: L.Base):
    st.subheader("Mapa do déficit de arborização")
    c1, c2, c3 = st.columns(3)
    porte = c1.selectbox("Porte do local", ["Todos"] + L.PORTES)
    setor = c2.selectbox("Setor", ["Todos"] + sorted(b.deficit["setor"].unique()))
    mortas = c3.checkbox("Mostrar árvores mortas (reposição)", value=False)
    d = b.deficit
    if porte != "Todos":
        d = d[d["porte"] == porte]
    if setor != "Todos":
        d = d[d["setor"] == setor]
    st.caption(f"{num(len(d))} pontos de déficit no filtro atual.")
    if d.empty:
        return
    m = mapa_base(d["lat"].mean(), d["lon"].mean(), 12 if setor == "Todos" else 15)
    HeatMap(d[["lat", "lon"]].values.tolist(), radius=8, blur=12, min_opacity=.3).add_to(m)
    if mortas:
        mt = b.arvores[~b.arvores["viva"]]
        if setor != "Todos":
            mt = mt[mt["setor"] == setor]
        for _, p in mt.iterrows():
            folium.CircleMarker([p["lat"], p["lon"]], radius=4, color="#000", fill=True,
                                tooltip="Árvore morta").add_to(m)
    st_folium(m, height=560, width=None, returned_objects=[], key=f"mapa_def_{porte}_{setor}_{mortas}")


@st.cache_data(show_spinner=False)
def _cluster(_b, k):
    return L.agrupar_setores(_b, k)


@st.cache_data(show_spinner=False)
def _sil(_b):
    return L.silhouette_por_k(_b)


@st.cache_data(show_spinner=False)
def _hot(_b, eps, minp):
    return L.hotspots(_b, eps, minp)


def pagina_lab(b: L.Base):
    st.subheader("Laboratório de Machine Learning")
    st.markdown("""
| Parte do app | Técnica | Tipo |
|---|---|---|
| Porte do local | Random Forest (comparado a 4 outros modelos) | **Supervisionado** — classificação |
| Explicação do modelo | Importância por permutação, árvore interpretável, análise contrafactual | **Explicabilidade** |
| Grupos de setores | K-Means + silhouette | **Não supervisionado** — agrupamento |
| Áreas críticas de déficit | DBSCAN sobre coordenadas | **Não supervisionado** — densidade |
| Risco de saúde por espécie | Modelo hierárquico bayesiano (encolhimento) | **Estatístico/ML** — estimativa com incerteza |
| Fila de vistoria | HistGradientBoosting + vizinhança espacial | **Supervisionado** — classificação |
| Rótulos duvidosos, medidas atípicas | Probabilidade fora da amostra, Isolation Forest | **Qualidade de dados** |
| Canteiro livre | HistGradientBoosting (regressão) | **Supervisionado** — regressão |
| Espécies/setores parecidos | Similaridade por cosseno, filtragem colaborativa | **Recomendação** |
| Equipes e rotas | K-Means + vizinho mais próximo + 2-opt | **Otimização** |
| Ranking de espécies e locais | Pontuação por regras (pesos fixos) | *Não é ML* — regra explícita (usa o risco acima) |
""")
    t1, t2, t3, t4, t5, t6 = st.tabs(["1 · Modelos", "2 · O que aprendeu", "3 · E se…?",
                                      "4 · Grupos de setores", "5 · Áreas críticas", "6 · Setores parecidos"])
    lab = b.metricas["laboratorio"]

    with t1:
        st.markdown("##### Comparação de modelos")
        st.caption("Validação cruzada em 5 partes **agrupada por setor**: cada modelo é testado em setores "
                   "que nunca viu, evitando que ele apenas decore o mapa.")
        comp = pd.DataFrame(lab["comparacao"])
        st.dataframe(comp.rename(columns={
            "modelo": "Modelo", "acuracia": "Acurácia", "acuracia_balanceada": "Acurácia balanceada",
            "f1_macro": "F1 macro", "porte_maior_que_o_real": "Porte MAIOR que o real"}),
            hide_index=True, width="stretch",
            column_config={c: st.column_config.NumberColumn(format="percent") for c in
                           ["Acurácia", "Acurácia balanceada", "F1 macro", "Porte MAIOR que o real"]})
        st.altair_chart(alt.Chart(comp).mark_bar(color=VERDE).encode(
            x=alt.X("f1_macro:Q", title="F1 macro", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("modelo:N", sort="-x", title=None),
            tooltip=["modelo", alt.Tooltip("f1_macro", format=".3f")]).properties(height=200), width="stretch")
        melhor = comp.loc[comp["f1_macro"].idxmax()]
        arv = comp[comp["modelo"].str.startswith("Árvore")].iloc[0]
        base_ = comp[comp["modelo"].str.startswith("Baseline")].iloc[0]
        st.info(f"**Leitura:** o melhor modelo ({melhor['modelo']}) tem F1 macro de {pct(melhor['f1_macro'])}, contra "
                f"{pct(base_['f1_macro'])} do baseline. Mas a árvore de decisão simples chega a {pct(arv['f1_macro'])}: "
                "a diferença é pequena, o que indica que o porte indicado em campo segue regras bastante "
                "objetivas. Usamos o Random Forest pelo melhor F1 e por fornecer probabilidades.")

        st.markdown("##### Matriz de confusão (Random Forest)")
        cm = pd.DataFrame(lab["matriz_confusao_rf"], index=lab["classes_matriz"], columns=lab["classes_matriz"])
        long = cm.reset_index(names="real").melt(id_vars="real", var_name="previsto", value_name="n")
        long["% da linha"] = long["n"] / long.groupby("real")["n"].transform("sum")
        h = alt.Chart(long).mark_rect().encode(
            x=alt.X("previsto:N", sort=PORTES_ORD, title="Previsto"), y=alt.Y("real:N", sort=PORTES_ORD, title="Real"),
            color=alt.Color("% da linha:Q", scale=alt.Scale(scheme="greens"), legend=None))
        t = alt.Chart(long).mark_text(fontSize=14).encode(
            x=alt.X("previsto:N", sort=PORTES_ORD), y=alt.Y("real:N", sort=PORTES_ORD),
            text=alt.Text("% da linha:Q", format=".0%"),
            color=alt.condition("datum['% da linha'] > 0.5", alt.value("white"), alt.value("black")))
        st.altair_chart((h + t).properties(height=240, width=320))
        st.caption("Cada linha soma 100%. O erro mais comum é prever *Grande* quando o real é *Médio* — "
                   "por isso o app só sobe de porte com confiança ≥ 70%.")

        st.markdown("##### Ajuste de hiperparâmetros (Random Forest)")
        st.dataframe(pd.DataFrame(lab["ajuste_hiperparametros"]).astype({"max_depth": str}), hide_index=True, width="stretch",
                     column_config={"f1_macro": st.column_config.NumberColumn(format="percent"),
                                    "acuracia": st.column_config.NumberColumn(format="percent")})
        st.caption("Grade de 6 combinações avaliadas com a mesma validação por setor. "
                   "O app usa `min_samples_leaf=5` e profundidade sem limite (melhor F1 da grade).")

    with t2:
        st.markdown("##### Importância das variáveis (por permutação)")
        st.caption("Quanto o F1 cai ao embaralhar cada variável nos setores de teste. Quanto maior, mais o modelo depende dela.")
        imp = pd.Series(lab["importancia_permutacao"]).rename_axis("variável").reset_index(name="queda no F1")
        st.altair_chart(alt.Chart(imp).mark_bar(color=VERDE).encode(
            x="queda no F1:Q", y=alt.Y("variável:N", sort="-x", title=None),
            tooltip=["variável", alt.Tooltip("queda no F1", format=".3f")]).properties(height=260), width="stretch")
        st.markdown("##### Árvore de decisão interpretável (regras em texto)")
        st.caption(f"Versão simples (profundidade 4) com {pct(lab['acuracia_arvore_interpretavel'])} de acerto na validação por setor. "
                   "Lê-se de cima para baixo: cada linha é uma pergunta sobre o local.")
        st.code(lab["regras_arvore"], language="text")

    with t3:
        st.markdown("##### Teste uma condição e veja o que mudaria")
        entrada = form_condicoes("lab")
        porte, probs = L.prever_porte(b.modelo, entrada)
        st.markdown(f"Porte recomendado: {selo_porte(porte)} &nbsp; " +
                    " · ".join(f"{k}: {pct(v, 0)}" for k, v in probs.items()), unsafe_allow_html=True)
        mostrar_what_if(b, entrada)

    with t4:
        st.markdown("##### Agrupamento de setores (K-Means)")
        st.caption("Sem usar nenhum rótulo, o algoritmo agrupa setores parecidos em déficit, dominância de espécies, "
                   "diversidade (índice de Shannon), mortalidade e quantidade de árvores. Setores com menos de 10 árvores inventariadas ficam de fora.")
        sil = _sil(b)
        c1, c2 = st.columns([1, 2])
        k = c1.slider("Número de grupos (k)", 2, 7, 4)
        c2.altair_chart(alt.Chart(sil).mark_line(point=True, color=VERDE).encode(
            x=alt.X("k:O"), y=alt.Y("silhouette:Q", scale=alt.Scale(zero=False)),
            tooltip=["k", alt.Tooltip("silhouette", format=".3f")]).properties(height=140), width="stretch")
        setores, perfil, s = _cluster(b, k)
        st.write(f"Silhouette com k={k}: **{s:.2f}**")
        if s < 0.3:
            st.warning("Silhouette abaixo de 0,3 indica grupos pouco separados: trate o agrupamento como exploração, "
                       "não como classificação definitiva dos setores.")
        st.dataframe(perfil.round(2), hide_index=True, width="stretch")
        sc = alt.Chart(setores).mark_circle(size=120).encode(
            x=alt.X("déficit relativo:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("diversidade (Shannon):Q", scale=alt.Scale(zero=False)),
            color=alt.Color("grupo:N"), tooltip=["setor", "grupo", "espécie dominante",
                                                  alt.Tooltip("déficit relativo", format=".0%")])
        st.altair_chart(sc.properties(height=320), width="stretch")
        baixar(setores[["setor", "grupo"] + L.CAMPOS_CLUSTER + ["espécie dominante"]].round(3),
               "grupos_de_setores", "grupos")

    with t5:
        st.markdown("##### Áreas críticas de déficit (DBSCAN)")
        st.caption("Agrupa pontos de déficit que estão próximos uns dos outros. Pontos isolados viram *ruído*. "
                   "Cada círculo é uma área contígua sem árvores — candidata a mutirão de plantio.")
        c1, c2 = st.columns(2)
        eps = c1.slider("Distância máxima entre pontos vizinhos (m)", 30, 150, 60, 10)
        minp = c2.slider("Mínimo de pontos para formar uma área", 10, 80, 25, 5)
        pts, areas = _hot(b, eps, minp)
        st.write(f"**{len(areas)} áreas** encontradas · {pct((pts['area'] < 0).mean(), 0)} dos pontos ficaram como ruído (isolados).")
        if areas.empty:
            st.warning("Nenhuma área com estes parâmetros.")
            return
        m = mapa_base(areas["lat"].mean(), areas["lon"].mean(), 12)
        for _, r in areas.iterrows():
            folium.CircleMarker([r["lat"], r["lon"]], radius=max(5, r["pontos"] ** 0.5 * 0.9),
                                color="#C0392B", fill=True, fill_opacity=.45,
                                tooltip=f"Área {int(r['area'])}: {int(r['pontos'])} pontos · setor {r['setor_principal']}").add_to(m)
        st_folium(m, height=480, width=None, returned_objects=[], key=f"mapa_hot_{eps}_{minp}")
        tab = areas.rename(columns={"area": "área", "setor_principal": "setor principal",
                                    "pct_pequeno": "% porte pequeno", "pct_grande": "% porte grande",
                                    "extensao_m": "extensão (m)"})
        st.dataframe(tab.head(30).round(5), hide_index=True, width="stretch",
                     column_config={"% porte pequeno": st.column_config.NumberColumn(format="percent"),
                                    "% porte grande": st.column_config.NumberColumn(format="percent")})
        baixar(tab, "areas_criticas", "areas")

    with t6:
        st.markdown("##### Recomendação colaborativa: o que vai bem em setores parecidos")
        st.caption("Dois setores são parecidos se têm composição de espécies parecida (cosseno). A afinidade soma o que "
                   "existe nos vizinhos; a *lacuna* é o que o setor ainda não tem. É um **sinal complementar**: "
                   "frequência em setores vizinhos não prova adequação ambiental.")
        setor = st.selectbox("Setor", sorted(b.vivas["setor"].unique()), key="colab_setor")
        sp = MX.setores_parecidos(b.vivas, setor)
        if sp.empty:
            st.info("Setor com poucas árvores inventariadas.")
        else:
            st.write("Setores mais parecidos: " + ", ".join(f"**{r['setor']}** ({r['similaridade']:.2f})" for _, r in sp.iterrows()))
            st.dataframe(MX.recomendar_por_setores_parecidos(b.especies, b.vivas, setor), hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="percent") for c in ["afinidade", "share_no_setor", "risco_eb"]})




def pagina_saude(b: L.Base):
    st.subheader("Saúde da arborização e manutenção")
    ex = b.metricas["extra"]
    ra, rh = ex["risco_arvore"], ex["risco_hierarquico_validacao"]
    st.caption("Árvores com estado Regular/Ruim/Péssima contam como *problema*. Dados: inventário de Ribeirão Preto.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Árvores vivas avaliadas", num(ra["n"]))
    k2.metric("Com problema fitossanitário", pct(ra["taxa_problema"]))
    k3.metric("AUC do modelo de risco", f"{ra['roc_auc']:.2f}")
    k4.metric("Problemas capturados no top 10%", pct(ra["captura_top10pct"], 0))

    t1, t2, t3 = st.tabs(["Fila de vistoria", "Surtos (DBSCAN)", "Risco por espécie"])
    with t1:
        st.markdown("##### Árvores hoje classificadas como boas, mas com maior risco previsto")
        st.caption("Modelo HistGradientBoosting (altura, copa, colo, inclinação, tipo de copa, gênero e saúde das 10 árvores "
                   "vizinhas), validado em setores nunca vistos. Previsões **fora da amostra**: a árvore nunca foi usada "
                   "para prever o próprio risco. Use como lista de **vistoria preventiva**, não como diagnóstico.")
        st.info(f"PR-AUC = {ra['pr_auc']:.2f} contra {ra['pr_auc_baseline']:.2f} de um sorteio aleatório "
                f"({ra['pr_auc'] / ra['pr_auc_baseline']:.0f}× melhor). Mesmo assim, a maioria da fila ainda é de árvores saudáveis: "
                "o modelo prioriza onde olhar primeiro.")
        f = b.extras["fila_vistoria"]
        setor = st.selectbox("Setor", ["Todos"] + sorted(f["setor"].unique()), key="fila_setor")
        if setor != "Todos":
            f = f[f["setor"] == setor]
        n = st.slider("Quantas árvores", 10, min(300, max(10, len(f))), min(50, len(f)), key="fila_n") if len(f) > 10 else len(f)
        f = f.head(n)
        st.dataframe(f.drop(columns=["lat", "lon"]), hide_index=True, width="stretch",
                     column_config={"risco_ml": st.column_config.ProgressColumn("risco", min_value=0, max_value=1, format="%.2f"),
                                    "google_maps": st.column_config.LinkColumn("mapa", display_text="abrir")})
        baixar(f.drop(columns=["lat", "lon"]), "fila_vistoria", "fila")
    with t2:
        st.markdown("##### Concentrações espaciais de árvores com problema")
        sv = b.extras["surtos"]
        if sv.empty:
            st.info("Nenhuma concentração encontrada.")
        else:
            m = mapa_base(sv["lat"].mean(), sv["lon"].mean(), 12)
            for _, r in sv.iterrows():
                folium.CircleMarker([r["lat"], r["lon"]], radius=max(6, r["arvores_com_problema"] ** 0.5 * 2.5),
                                    color="#C0392B", fill=True, fill_opacity=.5,
                                    tooltip=f"{int(r['arvores_com_problema'])} árvores · {r['especie_principal']}").add_to(m)
            st_folium(m, height=420, width=None, returned_objects=[], key="mapa_surtos")
            st.dataframe(sv, hide_index=True, width="stretch",
                         column_config={"pct_especie_principal": st.column_config.NumberColumn(format="percent")})
            st.caption("DBSCAN (80 m, ≥ 6 árvores). Se uma espécie domina o grupo, vale investigar praga/doença específica.")
    with t3:
        st.markdown("##### Risco estimado por espécie (modelo hierárquico bayesiano)")
        st.caption("Espécie com poucas árvores é *puxada* para a média do gênero/família; sem nenhuma árvore, usa o gênero "
                   "ou a família. O intervalo mostra a incerteza. Esse risco alimenta a nota de *evidência local* do ranking.")
        st.success(f"Validação (metade treina, metade testa): em espécies pouco vistas, o erro (log-loss) caiu de "
                   f"{rh['logloss_bruta_especies_pouco_vistas']:.3f} (taxa bruta) para "
                   f"{rh['logloss_hierarquico_especies_pouco_vistas']:.3f} (hierárquico).")
        e = b.especies[["especie", "nome_popular", "n_inventario", "risco_eb", "risco_lo", "risco_hi", "risco_fonte"]]
        e = e.sort_values("risco_eb", ascending=False)
        st.dataframe(e, hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="percent") for c in ["risco_eb", "risco_lo", "risco_hi"]})
        baixar(e, "risco_por_especie", "risco")



def pagina_qualidade(b: L.Base):
    st.subheader("Auditoria dos dados com machine learning")
    ex = b.metricas["extra"]
    t1, t2, t3 = st.tabs(["Porte duvidoso no déficit", "Medidas atípicas no inventário", "Catálogo × inventário"])
    with t1:
        r = ex["revisao_rotulos"]
        st.markdown(f"**{num(r['n_suspeitos'])} pontos** ({pct(r['pct_suspeitos'])}) têm porte levantado que o modelo, "
                    f"treinado em outros setores, considera quase impossível (P < {pct(r['limite'], 0)}).")
        st.caption("Pode ser erro de digitação, critério diferente entre levantadores ou algo no local que as 7 variáveis não captam. "
                   "É uma **lista de reconferência**, não uma correção automática.")
        rs = b.extras["rotulos_suspeitos"]
        setor = st.selectbox("Setor", ["Todos"] + sorted(rs["setor"].unique()), key="rot_setor")
        if setor != "Todos":
            rs = rs[rs["setor"] == setor]
        st.dataframe(rs.drop(columns=["lat", "lon"]), hide_index=True, width="stretch", column_config={
            "p_porte_levantado": st.column_config.NumberColumn("P(porte levantado)", format="percent"),
            "google_maps": st.column_config.LinkColumn("mapa", display_text="abrir")})
        baixar(rs.drop(columns=["lat", "lon"]), "revisar_deficit", "rot")
    with t2:
        a = ex["anomalias_inventario"]
        st.markdown(f"**{a['n']} árvores** com combinação de altura e copa atípica (Isolation Forest + regras).")
        st.caption("Exemplos: copa de 120 m, copa 4× maior que a altura. Provável erro de digitação (cm × m).")
        an = b.extras["anomalias"]
        st.dataframe(an, hide_index=True, width="stretch", column_config={
            "google_maps": st.column_config.LinkColumn("mapa", display_text="abrir"),
            "anomalia_score": st.column_config.NumberColumn(format="%.2f")})
        baixar(an, "inventario_anomalias", "anom")
    with t3:
        c = ex["auditoria_catalogo"]
        st.markdown(f"Para as **{c['n_especies']} espécies** com ≥ 15 árvores, um classificador (leave-one-out) que olha só a "
                    f"altura e a copa observadas concorda com o porte do Anexo VIII em **{pct(c['concordancia'], 0)}** dos casos.")
        st.warning("Cuidado: o inventário tem árvores **jovens**; altura e copa observadas subestimam o porte adulto. "
                   "Divergência é motivo para o botânico conferir, não prova de erro no Anexo VIII.")
        au = b.extras["auditoria_catalogo"]
        st.dataframe(au[au["porte"] != au["porte_observado"]], hide_index=True, width="stretch",
                     column_config={"p_porte_catalogo": st.column_config.NumberColumn("P(porte do catálogo)", format="percent")})


def pagina_sobre(b: L.Base):
    st.subheader("Como o app decide — e quão confiável é")
    mt = b.metricas
    st.markdown(f"""
**1. Porte (o que cabe no local).** Um modelo (Random Forest) aprendeu, com {num(mt['n'])} pontos levantados
em campo, a relação entre as condições do local (largura da calçada, tipo e proteção da fiação, telefonia,
edificação, recuo) e o porte indicado. O *setor* **não** entra no modelo: ele só aumentava o risco de decorar o mapa.

**2. Espécie (o que plantar).** Entre as espécies do Anexo VIII com o porte do local, a pontuação (0–100) combina:
- **Diversidade da cidade** — espécies raras pontuam mais; as que passam de 10% saem da lista;
- **Diversidade taxonômica** — evita gêneros (>20%) e famílias (>30%) já muito presentes;
- **Diversidade local** — penaliza espécie/gênero/família já existentes no raio escolhido;
- **Evidência local (saúde)** — risco fitossanitário estimado por modelo hierárquico (espécie → gênero → família) e pontuado pelo limite superior do intervalo de 90%: quanto menos árvores observadas, maior a incerteza e menor a nota. Espécie sem histórico não ganha vantagem por nunca ter sido vista com problema.

Os pesos mudam com a *Prioridade* escolhida:
""")
    st.dataframe(pd.DataFrame(L.PERFIS).T.rename(columns={
        "cidade": "diversidade da cidade", "taxon": "taxonômica", "local": "local", "saude": "evidência local"}
    ).map(lambda v: f"{v:.0%}"), width="stretch")

    st.markdown("#### Qualidade do modelo de porte")
    st.caption("Validação cruzada por setor: o modelo é testado em setores que ele nunca viu.")
    tab = pd.DataFrame(mt["por_tau"]).T
    tab.index = [f"confiança mínima {pct(float(i), 0)}" + (" (usada no app)" if abs(float(i) - L.TAU_PADRAO) < 1e-9 else "")
                 for i in tab.index]
    tab = tab.rename(columns={"acuracia": "acerto", "porte_maior_que_o_real": "porte MAIOR que o real",
                              "porte_menor_que_o_real": "porte menor que o real", "f1_macro": "F1 macro"})
    st.dataframe(tab.map(lambda v: pct(v)), width="stretch")
    st.info("Errar para **cima** é o risco (árvore grande sob fiação); por isso o app prefere errar para baixo. "
            "Ainda assim, o modelo é apoio: em pontos já levantados vale o porte indicado pela equipe de campo, "
            "e em locais novos a decisão final é do técnico.")


    st.markdown("#### Escopo e limitações")
    st.markdown("""
- **O que é ML e o que é regra.** Aprendizado supervisionado: porte do local, risco de saúde das árvores e canteiro livre. Estimativa hierárquica bayesiana: risco por espécie. Não supervisionado: grupos de setores, áreas críticas, surtos, anomalias, espécies/setores parecidos. Otimização: equipes e rotas. O ranking de espécies e de locais continua sendo uma pontuação por regras explícitas, com pesos definidos pelos autores — agora alimentada pelo risco estimado por ML.
- **Frequência é diversificação, não adequação.** A frequência da espécie no setor e na cidade é usada como critério de diversificação (evitar monocultura). Ela não mede a adequação ambiental da espécie. A compatibilidade com o local se limita ao porte do Anexo VIII; o histórico de saúde no inventário é apenas um indício complementar.
- **Como o déficit é definido.** O déficit são os pontos identificados em campo como aptos ao plantio; o sistema não infere necessidade ecológica. O *déficit relativo* apenas compara a quantidade desses pontos com as árvores já inventariadas no setor, e os *pontos livres* indicam ausência da espécie ou do gênero num raio de 100 m.
- **Funcionalidades implementadas:** modelo de porte, risco por espécie e por árvore, fila de vistoria, auditoria de dados, canteiro estimado, espécies/setores parecidos, ranking de espécies e locais, plano de plantio com equipes e rotas, agrupamento de setores, áreas críticas, mapas e exportação. **Não implementado:** modelo de adequação da espécie a clima/solo, que exigiria dados que a base atual não tem.
""")

    st.markdown("#### Qualidade dos dados (o que foi tratado na preparação)")
    rel = mt["relatorio_dados"]
    st.markdown(f"""
- {num(rel['deficit_pontos_duplicados_removidos'])} pontos de déficit duplicados removidos; {rel['deficit_linhas_sem_porte']} sem porte descartados.
- {rel['inventario_arvores_mortas']} árvores mortas separadas do cálculo de espécies (são pontos de reposição).
- {rel['inventario_sem_coordenada_ou_fora_da_cidade']} árvore sem coordenada válida descartada.
- {len(rel['grafias_unificadas (inventário -> catálogo)'])} grafias de espécies unificadas entre inventário e Anexo VIII.
- Anexo VIII com porte divergente (adotado o **menor**): {", ".join(rel['catalogo_conflitos_de_porte'])}.
- {rel['catalogo_especies_sem_historico_no_inventario']} das {len(b.especies)} espécies do Anexo VIII nunca aparecem no inventário.
""")
    with st.expander("Grafias unificadas (revisar com o botânico)"):
        st.json(rel["grafias_unificadas (inventário -> catálogo)"])


# ----------------------------------------------------------------------------
cabecalho()
dados = base()
PAGINAS = {
    "📊 Panorama": pagina_panorama,
    "🌱 Que espécie plantar aqui?": pagina_especie_para_local,
    "📍 Onde plantar esta espécie?": pagina_local_para_especie,
    "📋 Plano de plantio": pagina_plano,
    "🗺️ Mapa do déficit": pagina_mapa,
    "🩺 Saúde e manutenção": pagina_saude,
    "🔎 Qualidade dos dados": pagina_qualidade,
    "🧪 Laboratório de ML": pagina_lab,
    "ℹ️ Método e dados": pagina_sobre,
}
escolha = st.sidebar.radio("Menu", list(PAGINAS))
st.sidebar.caption("Fonte: inventário e levantamento de déficit de Ribeirão Preto (jan/2022).")
PAGINAS[escolha](dados)
