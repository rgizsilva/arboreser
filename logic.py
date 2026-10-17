from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier

import ml_extra

PASTA = Path(__file__).parent / "dados_processados"
_PARA_UTM = Transformer.from_crs("epsg:4326", "epsg:32723", always_xy=True)

PORTES = ["Pequeno", "Médio", "Grande"]
LIM_ESPECIE, LIM_GENERO, LIM_FAMILIA = 0.10, 0.20, 0.30
RAIO_PADRAO_M = 100
TAU_PADRAO = 0.70

OPCOES = {
    "tipo_fiacao": [
        "Nenhum", "Secundária (127V)", "Primária (13800V)",
        "Primária e Secundária (13800V + 127V)",
    ],
    "protecao": ["Não se aplica", "Protegida", "Desprotegida"],
    "telefonia": ["Ausente", "Presente"],
    "edificacao": ["Presente", "Ausente"],
    "tipo_edificacao": ["Baixa (<12m)", "Média (12<E<50m)", "Alta (>50m)", "Não se aplica"],
    "recuo": ["Ausente", "Presente", "Não se aplica"],
}
_COD = {
    "tipo_fiacao": {v: i for i, v in enumerate(OPCOES["tipo_fiacao"])},
    "protecao": {"Não se aplica": -1, "Desprotegida": 0, "Protegida": 1},
    "telefonia": {"Ausente": 0, "Presente": 1},
    "edificacao": {"Ausente": 0, "Presente": 1},
    "tipo_edificacao": {"Não se aplica": 0, "Baixa (<12m)": 1, "Média (12<E<50m)": 2, "Alta (>50m)": 3},
    "recuo": {"Não se aplica": -1, "Ausente": 0, "Presente": 1},
}
FEATURES = ["larg", "tipo_fiacao", "protecao", "telefonia", "edificacao", "tipo_edificacao", "recuo"]

PERFIS = {
    "Equilibrado": dict(cidade=0.25, taxon=0.10, local=0.30, saude=0.35),
    "Mais diversidade": dict(cidade=0.40, taxon=0.20, local=0.30, saude=0.10),
    "Mais segurança (espécies comprovadas)": dict(cidade=0.10, taxon=0.05, local=0.25, saude=0.60),
}


def codificar(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["larg"] = pd.to_numeric(df["larg"], errors="coerce").fillna(250)
    for col, mapa in _COD.items():
        X[col] = df[col].map(mapa).fillna(-1)
    return X[FEATURES]


def treinar_modelo(deficit: pd.DataFrame) -> RandomForestClassifier:
    d = deficit.dropna(subset=["porte"])
    modelo = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    modelo.fit(codificar(d), d["porte"])
    return modelo


def decidir_porte(probs: dict[str, float], tau: float = TAU_PADRAO) -> str:
    if probs.get("Grande", 0) >= tau:
        return "Grande"
    if probs.get("Médio", 0) + probs.get("Grande", 0) >= tau:
        return "Médio"
    return "Pequeno"


def prever_porte(modelo, entrada: dict, tau: float = TAU_PADRAO) -> tuple[str, dict[str, float]]:
    linha = pd.DataFrame([entrada])
    p = modelo.predict_proba(codificar(linha))[0]
    probs = {c: float(v) for c, v in zip(modelo.classes_, p)}
    probs = {k: probs.get(k, 0.0) for k in PORTES}
    return decidir_porte(probs, tau), probs


@dataclass
class Base:
    deficit: pd.DataFrame
    arvores: pd.DataFrame
    especies: pd.DataFrame
    concentracao: pd.DataFrame
    metricas: dict
    vivas: pd.DataFrame
    arvore_kd: cKDTree
    modelo: RandomForestClassifier
    canteiro: object = None
    extras: dict = None


def carregar_base() -> Base:
    deficit = pd.read_parquet(PASTA / "deficit.parquet")
    arvores = pd.read_parquet(PASTA / "arvores.parquet")
    especies = pd.read_parquet(PASTA / "especies.parquet")
    concentracao = pd.read_parquet(PASTA / "concentracao.parquet")
    metricas = json.loads((PASTA / "metricas.json").read_text(encoding="utf-8"))
    vivas = arvores[arvores["viva"]].reset_index(drop=True)
    extras = {n: pd.read_parquet(PASTA / f"{n}.parquet")
              for n in ("fila_vistoria", "surtos", "rotulos_suspeitos", "anomalias", "auditoria_catalogo")
              if (PASTA / f"{n}.parquet").exists()}
    return Base(
        deficit=deficit, arvores=arvores, especies=especies, concentracao=concentracao,
        metricas=metricas, vivas=vivas,
        arvore_kd=cKDTree(vivas[["E", "N"]].to_numpy()),
        modelo=treinar_modelo(deficit),
        canteiro=ml_extra.treinar_canteiro(deficit, codificar),
        extras=extras,
    )


def latlon_para_utm(lat: float, lon: float) -> tuple[float, float]:
    e, n = _PARA_UTM.transform(lon, lat)
    return float(e), float(n)


def ponto_mais_proximo(base: Base, setor: str, e: float, n: float,
                       max_m: float = 30.0) -> tuple[pd.Series | None, float]:
    pts = base.deficit[base.deficit["setor"] == setor]
    if pts.empty:
        return None, float("inf")
    dist, i = cKDTree(pts[["E", "N"]].to_numpy()).query([e, n])
    return (pts.iloc[int(i)] if dist <= max_m else None), float(dist)


def vizinhanca(base: Base, e: float, n: float, raio: float = RAIO_PADRAO_M) -> pd.DataFrame:
    idx = base.arvore_kd.query_ball_point([e, n], r=raio)
    return base.vivas.iloc[idx]


def _componentes(esp: pd.DataFrame, viz: pd.DataFrame | None,
                 extra: pd.DataFrame | None = None) -> pd.DataFrame:
    c = esp.copy()
    c["cidade"] = 1 - np.minimum(c["share_especie"] / 0.05, 1)
    gen = np.minimum(c["share_genero"] / LIM_GENERO, 1)
    fam = np.minimum(c["share_familia"].fillna(0) / LIM_FAMILIA, 1)
    c["taxon"] = 1 - np.maximum(gen, fam)

    if viz is None or (viz.empty and (extra is None or extra.empty)):
        c["n_local_esp"] = 0
        c["n_local_gen"] = 0
        c["n_local_fam"] = 0
    else:
        v = pd.concat([viz[["especie", "genero", "familia"]],
                       extra[["especie", "genero", "familia"]]]) if extra is not None and not extra.empty \
            else viz[["especie", "genero", "familia"]]
        ce, cg, cf = (v["especie"].value_counts(), v["genero"].value_counts(), v["familia"].value_counts())
        c["n_local_esp"] = c["especie"].map(ce).fillna(0).astype(int)
        c["n_local_gen"] = c["genero"].map(cg).fillna(0).astype(int) - c["n_local_esp"]
        c["n_local_fam"] = c["familia"].map(cf).fillna(0).astype(int) - c["n_local_esp"] - c["n_local_gen"]
        c[["n_local_gen", "n_local_fam"]] = c[["n_local_gen", "n_local_fam"]].clip(lower=0)
    c["local"] = 1 / (1 + c["n_local_esp"] + 0.5 * c["n_local_gen"] + 0.25 * c["n_local_fam"])

    risco = c["risco_hi"] if "risco_hi" in c else c["pct_problema"].fillna(0)
    c["saude"] = 1 - np.minimum(risco / 0.10, 1)
    return c


def _justificativa(r: pd.Series) -> str:
    partes = [f"{r['share_especie']*100:.1f}% da arborização da cidade"
              if r["n_inventario"] > 0 else "ainda não usada na cidade (diversidade máxima)"]
    if r["n_inventario"] >= 20:
        partes.append(f"{(1-r['pct_problema'])*100:.0f}% das {int(r['n_inventario'])} árvores em bom estado")
    elif r["n_inventario"] > 0:
        partes.append(f"só {int(r['n_inventario'])} árvores no inventário (pouco histórico)")
    else:
        partes.append("sem histórico local")
    if "risco_eb" in r.index and r["n_inventario"] < 20:
        partes.append(f"chance de problema estimada em {r['risco_eb']*100:.1f}% (com base no {r['risco_fonte']})")
    if r["n_local_esp"] == 0:
        partes.append("nenhuma igual por perto")
    else:
        partes.append(f"{int(r['n_local_esp'])} da mesma espécie por perto")
    if r["n_local_gen"] > 0:
        partes.append(f"{int(r['n_local_gen'])} do mesmo gênero por perto")
    return " · ".join(partes)


def ranquear_especies(base: Base, porte: str, viz: pd.DataFrame | None = None,
                      perfil: str = "Equilibrado", incluir_sem_historico: bool = True,
                      extra: pd.DataFrame | None = None, top_n: int = 10) -> pd.DataFrame:
    esp = base.especies[base.especies["porte"] == porte]
    esp = esp[~esp["saturada"]]
    if not incluir_sem_historico:
        esp = esp[esp["n_inventario"] > 0]
    if esp.empty:
        return esp
    c = _componentes(esp, viz, extra)
    w = PERFIS[perfil]
    c["pontuacao"] = 100 * sum(w[k] * c[k] for k in w)
    c["justificativa"] = c.apply(_justificativa, axis=1)
    c["historico"] = np.select(
        [c["n_inventario"] >= 20, c["n_inventario"] > 0], ["Comprovada", "Pouco histórico"], "Sem histórico",
    )
    c = c.sort_values(["pontuacao", "n_inventario", "especie"], ascending=[False, False, True])
    return c.head(top_n)


def especies_saturadas(base: Base) -> pd.DataFrame:
    c = base.concentracao
    lim = {"especie": LIM_ESPECIE, "genero": LIM_GENERO, "familia": LIM_FAMILIA}
    return c[c.apply(lambda r: r["share"] >= lim[r["nivel"]], axis=1)].sort_values("share", ascending=False)


def _contagem_raio(pontos: np.ndarray, alvo: pd.DataFrame, raio: float) -> np.ndarray:
    if alvo.empty:
        return np.zeros(len(pontos), dtype=int)
    kd = cKDTree(alvo[["E", "N"]].to_numpy())
    return kd.query_ball_point(pontos, r=raio, return_length=True)


def locais_para_especie(base: Base, especie: str, permitir_porte_maior: bool = False,
                        raio: float = RAIO_PADRAO_M) -> tuple[pd.DataFrame, pd.DataFrame]:
    linha = base.especies[base.especies["especie"] == especie]
    if linha.empty:
        return pd.DataFrame(), pd.DataFrame()
    r = linha.iloc[0]
    ordem = PORTES.index(r["porte"])
    aceitos = PORTES[ordem:] if permitir_porte_maior else [r["porte"]]
    pts = base.deficit[base.deficit["porte"].isin(aceitos)].copy()

    xy = pts[["E", "N"]].to_numpy()
    mesmas = base.vivas[base.vivas["especie"] == especie]
    mesmo_gen = base.vivas[(base.vivas["genero"] == r["genero"]) & (base.vivas["especie"] != especie)]
    pts["n_esp_raio"] = _contagem_raio(xy, mesmas, raio)
    pts["n_gen_raio"] = _contagem_raio(xy, mesmo_gen, raio)
    pts["livre"] = (pts["n_esp_raio"] == 0) & (pts["n_gen_raio"] == 0)

    dom = base.vivas.groupby("setor")["especie"].agg(
        lambda s: s.value_counts(normalize=True).iloc[0]).rename("dominancia")
    dom_nome = base.vivas.groupby("setor")["especie"].agg(lambda s: s.value_counts().index[0]).rename("dominante")
    n_esp_setor = mesmas.groupby("setor").size().rename("arvores_da_especie")

    resumo = pts.groupby("setor").agg(
        pontos_compativeis=("livre", "size"), pontos_livres=("livre", "sum")).reset_index()
    resumo = (resumo.merge(n_esp_setor, on="setor", how="left")
                    .merge(dom, on="setor", how="left").merge(dom_nome, on="setor", how="left"))
    resumo["arvores_da_especie"] = resumo["arvores_da_especie"].fillna(0).astype(int)
    resumo["pontos_livres"] = resumo["pontos_livres"].astype(int)
    return pts, resumo


def espalhar(pontos: pd.DataFrame, n: int, dist_min: float = 40.0) -> pd.DataFrame:
    escolhidos: list[int] = []
    xy = pontos[["E", "N"]].to_numpy()
    for i in range(len(pontos)):
        if len(escolhidos) >= n:
            break
        if not escolhidos or np.min(np.hypot(*(xy[escolhidos] - xy[i]).T)) >= dist_min:
            escolhidos.append(i)
    return pontos.iloc[escolhidos]


def gerar_plano(base: Base, setor: str, n_arvores: int, perfil: str = "Equilibrado",
                max_share_plano: float = 0.10, raio: float = RAIO_PADRAO_M,
                incluir_sem_historico: bool = True, dist_min: float = 15.0) -> pd.DataFrame:
    pts = base.deficit[base.deficit["setor"] == setor].copy()
    if pts.empty:
        return pd.DataFrame()
    pts = espalhar(pts.sample(frac=1, random_state=7), n_arvores, dist_min)
    teto = max(1, int(np.ceil(max_share_plano * len(pts))))
    usados: dict[str, int] = {}
    plantadas: list[dict] = []
    linhas = []
    for _, p in pts.iterrows():
        viz = vizinhanca(base, p["E"], p["N"], raio)
        extra = pd.DataFrame([q for q in plantadas
                              if np.hypot(q["E"] - p["E"], q["N"] - p["N"]) <= raio],
                             columns=["especie", "genero", "familia", "E", "N"])
        cand = ranquear_especies(base, p["porte"], viz, perfil, incluir_sem_historico,
                                 extra=extra, top_n=len(base.especies))
        cand = cand[cand["especie"].map(lambda e: usados.get(e, 0)) < teto]
        if cand.empty:
            continue
        s = cand.iloc[0]
        usados[s["especie"]] = usados.get(s["especie"], 0) + 1
        plantadas.append(dict(especie=s["especie"], genero=s["genero"], familia=s["familia"],
                              E=p["E"], N=p["N"]))
        linhas.append(dict(
            setor=p["setor"], lat=p["lat"], lon=p["lon"], E=p["E"], N=p["N"], porte_do_local=p["porte"],
            especie=s["especie"], nome_popular=s["nome_popular"], pontuacao=round(s["pontuacao"], 1),
            historico=s["historico"], condicoes_do_local=p["condicoes"],
            google_maps=f"https://www.google.com/maps?q={p['lat']:.6f},{p['lon']:.6f}",
        ))
    return pd.DataFrame(linhas)


def resumo_setores(base: Base) -> pd.DataFrame:
    d = base.deficit.pivot_table(index="setor", columns="porte", values="E", aggfunc="size", fill_value=0)
    for p in PORTES:
        if p not in d:
            d[p] = 0
    d = d[PORTES].rename(columns=lambda c: f"déficit {c.lower()}")
    d["déficit total"] = d.sum(axis=1)
    v = base.vivas.groupby("setor").size().rename("árvores")
    m = base.arvores[~base.arvores["viva"]].groupby("setor").size().rename("árvores mortas")
    dom = base.vivas.groupby("setor")["especie"].agg(
        lambda s: s.value_counts(normalize=True).iloc[0]).rename("dominância")
    dom_nome = base.vivas.groupby("setor")["especie"].agg(lambda s: s.value_counts().index[0]).rename("espécie dominante")
    r = d.join(v, how="outer").join(m).join(dom).join(dom_nome).fillna({"árvores": 0, "árvores mortas": 0})
    r["árvores"] = r["árvores"].astype(int)
    r["árvores mortas"] = r["árvores mortas"].astype(int)
    r["déficit total"] = r["déficit total"].fillna(0).astype(int)
    r["déficit relativo"] = r["déficit total"] / (r["déficit total"] + r["árvores"]).replace(0, np.nan)
    return r.reset_index().rename(columns={"index": "setor"}).sort_values("déficit relativo", ascending=False)


from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

_ROTULO_CAMPO = {
    "tipo_fiacao": "Fiação elétrica", "protecao": "Proteção da fiação", "telefonia": "Telefonia/TV",
    "edificacao": "Edificação", "tipo_edificacao": "Tipo de edificação", "recuo": "Recuo",
}


def analise_what_if(modelo, entrada: dict, tau: float = TAU_PADRAO) -> pd.DataFrame:
    base_porte, base_p = prever_porte(modelo, entrada, tau)
    linhas = []
    for campo, opcoes in OPCOES.items():
        for o in opcoes:
            if o == entrada[campo]:
                continue
            alt = {**entrada, campo: o}
            if campo == "tipo_fiacao" and o == "Nenhum":
                alt["protecao"] = "Não se aplica"
            porte, p = prever_porte(modelo, alt, tau)
            linhas.append({"condição alterada": _ROTULO_CAMPO[campo], "de": entrada[campo], "para": o,
                           "porte resultante": porte, "P(Grande)": p["Grande"], "muda?": porte != base_porte})
    for delta in (-100, -50, 50, 100):
        larg = max(100, entrada["larg"] + delta)
        if larg == entrada["larg"]:
            continue
        porte, p = prever_porte(modelo, {**entrada, "larg": larg}, tau)
        linhas.append({"condição alterada": "Largura da calçada", "de": f"{entrada['larg']} cm",
                       "para": f"{larg} cm", "porte resultante": porte, "P(Grande)": p["Grande"],
                       "muda?": porte != base_porte})
    return pd.DataFrame(linhas).sort_values(["muda?", "P(Grande)"], ascending=[False, False])


def _shannon(s: pd.Series) -> float:
    p = s.value_counts(normalize=True)
    return float(-(p * np.log(p)).sum())


CAMPOS_CLUSTER = ["déficit relativo", "% déficit pequeno", "% déficit grande", "dominância",
                  "diversidade (Shannon)", "% árvores mortas", "árvores"]


def perfil_setores(base: Base) -> pd.DataFrame:
    r = resumo_setores(base).set_index("setor")
    r = r[r["árvores"] >= 10]
    sh = base.vivas.groupby("setor")["especie"].agg(_shannon).rename("diversidade (Shannon)")
    r = r.join(sh)
    r["% déficit pequeno"] = r["déficit pequeno"] / r["déficit total"].replace(0, np.nan)
    r["% déficit grande"] = r["déficit grande"] / r["déficit total"].replace(0, np.nan)
    r["% árvores mortas"] = r["árvores mortas"] / (r["árvores"] + r["árvores mortas"])
    return r.dropna(subset=CAMPOS_CLUSTER)


def agrupar_setores(base: Base, k: int = 4) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    p = perfil_setores(base)
    Z = StandardScaler().fit_transform(p[CAMPOS_CLUSTER])
    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Z)
    p = p.copy()
    p["grupo"] = km.labels_ + 1
    sil = float(silhouette_score(Z, km.labels_))
    z = pd.DataFrame(Z, index=p.index, columns=CAMPOS_CLUSTER).groupby(p["grupo"]).mean()
    medias = p.groupby("grupo")[CAMPOS_CLUSTER].mean()
    medias["nº setores"] = p.groupby("grupo").size()
    medias["características marcantes"] = [
        ", ".join(f"{'alto' if z.loc[g, c] > 0 else 'baixo'} {c}" for c in z.loc[g].abs().nlargest(2).index)
        for g in medias.index
    ]
    return p.reset_index(), medias.reset_index(), sil


def silhouette_por_k(base: Base, ks=range(2, 8)) -> pd.DataFrame:
    p = perfil_setores(base)
    Z = StandardScaler().fit_transform(p[CAMPOS_CLUSTER])
    return pd.DataFrame({"k": list(ks), "silhouette": [
        silhouette_score(Z, KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(Z)) for k in ks]})


def hotspots(base: Base, eps_m: float = 60.0, min_pontos: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = base.deficit.copy()
    d["area"] = DBSCAN(eps=eps_m, min_samples=min_pontos).fit_predict(d[["E", "N"]].to_numpy())
    cl = d[d["area"] >= 0]
    if cl.empty:
        return d, pd.DataFrame()
    res = cl.groupby("area").agg(
        pontos=("E", "size"), lat=("lat", "mean"), lon=("lon", "mean"),
        setor_principal=("setor", lambda s: s.value_counts().index[0]),
        pct_pequeno=("porte", lambda s: (s == "Pequeno").mean()),
        pct_grande=("porte", lambda s: (s == "Grande").mean()),
    ).reset_index()
    ext = cl.groupby("area").apply(
        lambda g: float(np.hypot(g["E"].max() - g["E"].min(), g["N"].max() - g["N"].min())), include_groups=False)
    res["extensao_m"] = res["area"].map(ext).round(0)
    return d, res.sort_values("pontos", ascending=False).reset_index(drop=True)
