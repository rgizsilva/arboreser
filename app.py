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


@st.cache_resource(show_spinner="Carregando dados…")
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


def form_condicoes(chave: str) -> dict:
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


def pagina_panorama(b: L.Base):
    st.subheader("Panorama da arborização")
    sat = L.especies_saturadas(b)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Árvores inventariadas", num(len(b.vivas)))
    c2.metric("Vagas livres para plantio", num(len(b.deficit)))
    c3.metric("Árvores mortas para repor", num(len(b.arvores) - len(b.vivas)))
    c4.metric("Espécies diferentes na cidade", num(b.vivas["especie"].nunique()))

    st.markdown("#### Variedade de espécies")
    st.caption("Para a cidade ficar protegida contra pragas, uma boa regra é que nenhuma espécie passe de 10% das árvores, "
               "nenhum gênero (grupo de espécies parecidas) passe de 20% e nenhuma família passe de 30%.")
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
                   "Por isso o app **não indica** novos plantios dessa espécie.")

    st.markdown("#### Setores: onde faltam mais árvores")
    st.caption("*Falta de árvores* mostra, de cada 100 lugares possíveis para ter uma árvore, quantos ainda estão vazios. "
               "*Espécie mais comum* mostra quanto uma única espécie ocupa do setor.")
    r = L.resumo_setores(b)
    r = r[["setor", "déficit total", "déficit pequeno", "déficit médio", "déficit grande", "árvores",
           "árvores mortas", "déficit relativo", "espécie dominante", "dominância"]].rename(columns={
        "setor": "Setor", "déficit total": "Vagas livres", "déficit pequeno": "Vagas p/ árvore pequena",
        "déficit médio": "Vagas p/ árvore média", "déficit grande": "Vagas p/ árvore grande",
        "árvores": "Árvores existentes", "árvores mortas": "Árvores mortas", "déficit relativo": "Falta de árvores",
        "espécie dominante": "Espécie mais comum", "dominância": "Peso dessa espécie"})
    st.dataframe(
        r, hide_index=True, width="stretch",
        column_config={
            "Falta de árvores": st.column_config.ProgressColumn("Falta de árvores", min_value=0, max_value=1, format="percent"),
            "Peso dessa espécie": st.column_config.NumberColumn(format="percent"),
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
        with st.expander("Mais opções"):
            raio = st.slider("Até quantos metros olhar ao redor", 30, 250, L.RAIO_PADRAO_M, 10,
                             help="As árvores dentro dessa distância entram na conta para evitar repetir espécies na mesma área.")
            sem_hist = st.checkbox("Incluir espécies ainda não plantadas na cidade", value=True)

    pts = b.deficit[b.deficit["setor"] == setor]
    with c2:
        st.caption("Clique no mapa para indicar o local. Os pontos coloridos são lugares livres para plantio "
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
                    f"**Elementos urbanos:** {ponto['elementos_urbanos']} · **Espaço de terra livre:** {ponto['canteiro_cm']:.0f} cm")
        st.markdown(f"Tamanho de árvore indicado no levantamento: {selo_porte(porte)}", unsafe_allow_html=True)
        e, n = ponto["E"], ponto["N"]
        viz = L.vizinhanca(b, e, n, raio)
    else:
        st.markdown("**Condições do local**")
        entrada = form_condicoes("pg2")
        porte, _ = L.prever_porte(b.modelo, entrada)
        st.markdown(f"Tamanho de árvore indicado: {selo_porte(porte)}", unsafe_allow_html=True)
        st.caption("Na dúvida, o app escolhe o tamanho menor: uma árvore grande demais embaixo da fiação traz mais problemas do que uma pequena.")
        cant = MX.prever_canteiro(b.canteiro, entrada, L.codificar)
        st.markdown(f"Espaço de terra livre estimado: **{cant:.0f} cm**")

    st.caption(f"{len(viz)} árvores num raio de {raio} m deste ponto.")
    recs = L.ranquear_especies(b, porte, viz, perfil, sem_hist, top_n=10)
    if recs.empty:
        st.warning("Nenhuma espécie disponível com estas escolhas.")
        return
    st.markdown(f"##### As {len(recs)} melhores espécies de porte {porte.lower()} para este local")
    tabela_especies(recs)
    with st.expander("🔁 Alternativas à primeira da lista (se faltar muda)"):
        alt_ = MX.especies_parecidas(b.especies, b.vivas, recs.iloc[0]["especie"], n=6)
        st.caption("Espécies parecidas em família, tamanho e lugares onde costumam aparecer. "
                   "Todas têm o mesmo porte e ainda cabem na cidade sem exagerar na repetição.")
        alt_ = alt_.rename(columns={"especie": "Espécie", "nome_popular": "Nome popular", "porte": "Porte",
                                    "familia": "Família", "n_inventario": "Árvores na cidade", "similaridade": "Parecida"})
        st.dataframe(alt_, hide_index=True, width="stretch",
                     column_config={"Parecida": st.column_config.ProgressColumn(
                         "Parecida", min_value=0, max_value=1, format="percent")})
    with st.expander("Espécies que o app evita indicar"):
        sat = b.especies[b.especies["saturada"]]
        st.write(", ".join(nome_esp(r) for _, r in sat.iterrows()) or "Nenhuma.")
        st.caption("Essas espécies já ocupam espaço demais na arborização atual.")


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
    k1.markdown(f"Tamanho<br>{selo_porte(r['porte'])}", unsafe_allow_html=True)
    k2.metric("Árvores no inventário", num(r["n_inventario"]))
    k3.metric("% da arborização", pct(r["share_especie"]))
    k4.metric("Em bom estado", pct(1 - r["pct_problema"], 0) if r["n_inventario"] >= 20 else "sem histórico")
    if r["saturada"]:
        st.error("Esta espécie (ou o grupo dela) já está em excesso na cidade. **Não recomendamos novos plantios.**")
    elif r["n_inventario"] == 0:
        st.info("Espécie ainda não plantada na cidade: ótima para variedade, mas ainda sem histórico de como se comporta aqui.")

    if r["saturada"]:
        sub = MX.especies_parecidas(b.especies, b.vivas, esp, n=6)
        st.markdown("**Substitutas sugeridas (mesmo porte, fora do limite de diversidade):**")
        st.dataframe(sub, hide_index=True, width="stretch")
    pts, resumo = L.locais_para_especie(b, esp, maior)
    if resumo.empty:
        st.warning("Não há lugares livres compatíveis com o tamanho desta espécie.")
        return
    ordem = st.radio("Ordenar setores por", ["Mais pontos livres", "Setores mais dominados por uma só espécie"],
                     horizontal=True)
    resumo = resumo.sort_values("pontos_livres" if ordem.startswith("Mais") else "dominancia", ascending=False)
    st.caption("*Lugares sem repetição* = lugares livres sem nenhuma árvore da mesma espécie ou grupo a menos de 100 m.")
    sel = st.dataframe(
        resumo.rename(columns={"pontos_compativeis": "lugares compatíveis", "pontos_livres": "lugares sem repetição",
                               "arvores_da_especie": "já tem desta espécie", "dominancia": "peso da espécie mais comum",
                               "dominante": "espécie mais comum"}),
        hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
        column_config={"peso da espécie mais comum": st.column_config.NumberColumn(format="percent")},
    )
    linhas = sel.selection.rows if sel and sel.selection else []
    setor = resumo.iloc[linhas[0]]["setor"] if linhas else resumo.iloc[0]["setor"]
    st.markdown(f"##### Lugares sugeridos no setor **{setor}**")
    st.caption("Clique numa linha da tabela acima para trocar de setor. Os lugares ficam a pelo menos 40 m um do outro, "
               "para não concentrar a espécie.")
    qtd = st.slider("Quantos pontos", 5, 50, 15)
    livres = pts[(pts["setor"] == setor) & pts["livre"]]
    escolhidos = L.espalhar(livres, qtd, 40.0)
    if escolhidos.empty:
        st.warning("Este setor não tem lugares livres para esta espécie.")
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
    st.caption("Monta uma lista pronta para levar a campo: cada lugar recebe uma espécie do tamanho certo, "
               "sem repetir espécies parecidas por perto e sem exagerar em uma única espécie.")
    c1, c2, c3, c4 = st.columns(4)
    setor = c1.selectbox("Setor", sorted(b.deficit["setor"].unique()))
    qtd = c2.slider("Quantas árvores", 10, 300, 60, 10)
    teto = c3.slider("Máximo de uma mesma espécie no plano", 5, 25, 10, 1, format="%d%%") / 100
    perfil = c4.selectbox("Prioridade", list(L.PERFIS))
    sem_hist = st.checkbox("Incluir espécies ainda não plantadas na cidade", value=True)

    if st.button("Gerar plano", type="primary"):
        with st.spinner("Calculando o plano…"):
            st.session_state["plano"] = (setor, L.gerar_plano(b, setor, qtd, perfil, teto,
                                                              incluir_sem_historico=sem_hist))
    if "plano" not in st.session_state:
        return
    setor_p, plano = st.session_state["plano"]
    if plano.empty:
        st.warning("Não foi possível montar o plano com estas escolhas.")
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
    st.markdown("#### 🧭 Equipes e rotas de plantio")
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
    st.subheader("Mapa das vagas livres para plantio")
    c1, c2, c3 = st.columns(3)
    porte = c1.selectbox("Tamanho indicado do local", ["Todos"] + L.PORTES)
    setor = c2.selectbox("Setor", ["Todos"] + sorted(b.deficit["setor"].unique()))
    mortas = c3.checkbox("Mostrar árvores mortas (reposição)", value=False)
    d = b.deficit
    if porte != "Todos":
        d = d[d["porte"] == porte]
    if setor != "Todos":
        d = d[d["setor"] == setor]
    st.caption(f"{num(len(d))} lugares livres com os filtros escolhidos.")
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


def pagina_saude(b: L.Base):
    st.subheader("Saúde das árvores e manutenção")
    st.caption("Aqui você vê onde vale a pena fazer uma visita de conferência e quais espécies costumam ficar mais saudáveis na cidade.")
    ra = b.metricas["extra"]["risco_arvore"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Árvores avaliadas", num(ra["n"]))
    k2.metric("Com algum problema de saúde", pct(ra["taxa_problema"]))
    k3.metric("Focos com várias árvores doentes", num(len(b.extras["surtos"])))

    t1, t2, t3 = st.tabs(["Árvores para visitar", "Focos de problemas", "Espécies mais resistentes"])
    with t1:
        st.markdown("##### Árvores que hoje parecem boas, mas merecem uma visita")
        st.caption("A lista é montada a partir do tamanho, do formato da copa, do colo e da saúde das árvores vizinhas. "
                   "Ela só indica por onde começar: a decisão final é sempre da equipe, olhando a árvore de perto.")
        f = b.extras["fila_vistoria"]
        setor = st.selectbox("Setor", ["Todos"] + sorted(f["setor"].unique()), key="fila_setor")
        if setor != "Todos":
            f = f[f["setor"] == setor]
        n = st.slider("Quantas árvores mostrar", 10, min(300, max(10, len(f))), min(50, len(f)), key="fila_n") if len(f) > 10 else len(f)
        f = f.head(n).rename(columns={"setor": "Setor", "especie": "Espécie", "nome_popular": "Nome popular",
                                      "estado": "Estado hoje", "altura_cm": "Altura (cm)", "copa_cm": "Copa (cm)",
                                      "colo": "Colo", "risco_ml": "Chance de problema", "google_maps": "Mapa"})
        mostrar = f.drop(columns=["lat", "lon"])
        st.dataframe(mostrar, hide_index=True, width="stretch",
                     column_config={"Chance de problema": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
                                    "Mapa": st.column_config.LinkColumn("Mapa", display_text="abrir")})
        baixar(mostrar, "arvores_para_visitar", "fila")
    with t2:
        st.markdown("##### Lugares onde várias árvores doentes aparecem juntas")
        sv = b.extras["surtos"]
        if sv.empty:
            st.info("Nenhum foco encontrado.")
        else:
            m = mapa_base(sv["lat"].mean(), sv["lon"].mean(), 12)
            for _, r in sv.iterrows():
                folium.CircleMarker([r["lat"], r["lon"]], radius=max(6, r["arvores_com_problema"] ** 0.5 * 2.5),
                                    color="#C0392B", fill=True, fill_opacity=.5,
                                    tooltip=f"{int(r['arvores_com_problema'])} árvores · {r['especie_principal']}").add_to(m)
            st_folium(m, height=420, width=None, returned_objects=[], key="mapa_surtos")
            tab = sv.rename(columns={"arvores_com_problema": "Árvores com problema", "setor_principal": "Setor",
                                     "especie_principal": "Espécie mais afetada", "pct_especie_principal": "% dessa espécie"})
            st.dataframe(tab.drop(columns=["surto", "lat", "lon"]), hide_index=True, width="stretch",
                         column_config={"% dessa espécie": st.column_config.NumberColumn(format="percent")})
            st.caption("Quando uma única espécie domina o foco, vale investigar se há praga ou doença específica.")
    with t3:
        st.markdown("##### Chance de problema por espécie")
        st.caption("Para espécies com poucas árvores, a estimativa também considera as espécies parentes. "
                   "Quanto menor a chance, mais resistente a espécie tem se mostrado na cidade.")
        e = b.especies[["especie", "nome_popular", "n_inventario", "risco_eb", "risco_fonte"]].sort_values("risco_eb")
        e = e.rename(columns={"especie": "Espécie", "nome_popular": "Nome popular", "n_inventario": "Árvores na cidade",
                              "risco_eb": "Chance de problema", "risco_fonte": "Estimativa baseada em"})
        st.dataframe(e, hide_index=True, width="stretch",
                     column_config={"Chance de problema": st.column_config.NumberColumn(format="percent")})
        baixar(e, "chance_de_problema_por_especie", "risco")


cabecalho()
dados = base()
PAGINAS = {
    "📊 Panorama": pagina_panorama,
    "🌱 Que espécie plantar aqui?": pagina_especie_para_local,
    "📍 Onde plantar esta espécie?": pagina_local_para_especie,
    "📋 Plano de plantio": pagina_plano,
    "🗺️ Mapa das vagas livres": pagina_mapa,
    "🩺 Saúde e manutenção": pagina_saude,
}
escolha = st.sidebar.radio("Menu", list(PAGINAS))
st.sidebar.caption("Fonte: inventário e levantamento de déficit de Ribeirão Preto (jan/2022).")
PAGINAS[escolha](dados)
