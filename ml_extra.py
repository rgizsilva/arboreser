
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta as _beta
from sklearn.cluster import DBSCAN, KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest
from sklearn.metrics import average_precision_score, log_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

PROBLEMA = ["Regular", "Ruim", "Péssima"]
PORTES = ["Pequeno", "Médio", "Grande"]


def _encolher(n: pd.Series, y: pd.Series, prior: pd.Series | float, k: float) -> pd.Series:
    """Média posterior (y + k*prior) / (n + k): puxa grupos pequenos para o nível acima."""
    return (y + k * prior) / (n + k)


def risco_hierarquico(v: pd.DataFrame, k_fam: float = 60, k_gen: float = 30, k_esp: float = 20) -> dict:
    """Estima P(problema) por família, gênero e espécie com encolhimento hierárquico.

    v: árvores vivas com colunas especie, genero, familia, problema (0/1).
    Devolve dict com 'global' e Series por nível.
    """
    p0 = float(v["problema"].mean())
    fam = v.groupby("familia")["problema"].agg(["size", "sum"])
    fam["p"] = _encolher(fam["size"], fam["sum"], p0, k_fam)
    v = v.assign(_pf=v["familia"].map(fam["p"]).fillna(p0))
    gen = v.groupby("genero").agg(n=("problema", "size"), y=("problema", "sum"), pf=("_pf", "mean"))
    gen["p"] = _encolher(gen["n"], gen["y"], gen["pf"], k_gen)
    v = v.assign(_pg=v["genero"].map(gen["p"]).fillna(p0))
    esp = v.groupby("especie").agg(n=("problema", "size"), y=("problema", "sum"), pg=("_pg", "mean"))
    esp["p"] = _encolher(esp["n"], esp["y"], esp["pg"], k_esp)
    return {"global": p0, "familia": fam["p"], "genero": gen["p"], "especie": esp["p"],
            "esp_n": esp["n"], "esp_prior": esp["pg"], "k_esp": k_esp}


def estimar_risco(cat: pd.DataFrame, modelo: dict) -> pd.DataFrame:
    """Aplica o modelo hierárquico ao catálogo (inclusive espécies sem nenhuma árvore).

    Acrescenta: risco_eb, risco_lo, risco_hi (intervalo de 90 %) e risco_fonte."""
    c = cat.copy()
    p0 = modelo["global"]
    fam = c["familia"].map(modelo["familia"])
    gen = c["genero"].map(modelo["genero"])
    esp = c["especie"].map(modelo["especie"])
    c["risco_eb"] = esp.fillna(gen).fillna(fam).fillna(p0)
    c["risco_fonte"] = np.select(
        [esp.notna(), gen.notna(), fam.notna()], ["espécie", "gênero", "família"], "cidade")
    n_ef = c["especie"].map(modelo["esp_n"]).fillna(0) + modelo["k_esp"]
    a = c["risco_eb"] * n_ef
    b = (1 - c["risco_eb"]) * n_ef
    c["risco_lo"] = _beta.ppf(0.05, a, b)
    c["risco_hi"] = _beta.ppf(0.95, a, b)
    return c


def validar_risco(v: pd.DataFrame, semente: int = 0) -> dict:
    """Prova que o encolhimento hierárquico prevê melhor que a taxa bruta.
    Treina em metade das árvores e mede log-loss na outra metade (menor = melhor)."""
    rng = np.random.default_rng(semente)
    m = rng.random(len(v)) < 0.5
    A, B = v[m], v[~m]
    mod = risco_hierarquico(A)
    p0 = mod["global"]
    pB_eb = B["especie"].map(mod["especie"]).fillna(B["genero"].map(mod["genero"])).fillna(
        B["familia"].map(mod["familia"])).fillna(p0)
    bruto = A.groupby("especie")["problema"].mean()
    pB_bruto = B["especie"].map(bruto).fillna(p0).clip(0.005, 0.995)  
    eps = 1e-6
    ll = lambda p: float(log_loss(B["problema"], np.clip(p, eps, 1 - eps), labels=[0, 1]))
    pequenas = B["especie"].map(A["especie"].value_counts()).fillna(0) < 20
    res = {"n_treino": int(len(A)), "n_teste": int(len(B)),
           "logloss_taxa_global": ll(np.full(len(B), p0)),
           "logloss_taxa_bruta_por_especie": ll(pB_bruto),
           "logloss_hierarquico": ll(pB_eb)}
    if pequenas.any():
        res["logloss_bruta_especies_pouco_vistas"] = float(log_loss(
            B.loc[pequenas, "problema"], np.clip(pB_bruto[pequenas], eps, 1 - eps), labels=[0, 1]))
        res["logloss_hierarquico_especies_pouco_vistas"] = float(log_loss(
            B.loc[pequenas, "problema"], np.clip(pB_eb[pequenas], eps, 1 - eps), labels=[0, 1]))
    return res


_CAT_COPA = {"Eliptica horizontal": 0, "Eliptica vertical": 1, "Pendente": 2, "Glabosa": 3, "Umbeliforme": 4}
_CAT_COLO = {"Boa": 0, "Média": 1, "Ruim": 2}


def _feat_arvore(v: pd.DataFrame, k_viz: int = 10) -> pd.DataFrame:
    """Variáveis por árvore. Inclui taxa de problema nas k árvores vizinhas (exclui a própria)
    e o 'gênero' como categoria."""
    X = pd.DataFrame(index=v.index)
    X["altura_cm"] = pd.to_numeric(v["altura_cm"], errors="coerce").clip(upper=3000)
    X["copa_cm"] = pd.to_numeric(v["copa_cm"], errors="coerce").clip(upper=2500)
    X["copa_sobre_altura"] = X["copa_cm"] / X["altura_cm"]
    X["inclinacao"] = v["inclinacao"].astype(float)
    X["colo"] = v["colo"].map(_CAT_COLO)
    X["copa_tipo"] = v["copa_tipo"].map(_CAT_COPA)
    X["genero_cod"] = v["genero"].astype("category").cat.codes.astype(float)
    xy = v[["E", "N"]].to_numpy()
    nn = NearestNeighbors(n_neighbors=k_viz + 1).fit(xy)
    idx = nn.kneighbors(xy, return_distance=False)[:, 1:]
    y = v["problema"].to_numpy()
    X["viz_problema"] = y[idx].mean(axis=1)
    return X


def modelo_risco_arvore(vivas: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """HistGradientBoosting com validação cruzada AGRUPADA POR SETOR.
    Devolve (árvores com 'risco_ml' fora-da-amostra, métricas)."""
    v = vivas.dropna(subset=["estado"]).copy()
    v["problema"] = v["estado"].isin(PROBLEMA).astype(int)
    X, y, g = _feat_arvore(v), v["problema"].to_numpy(), v["setor"].to_numpy()
    cat = [list(X.columns).index("genero_cod")]
    oof, oof_base = np.zeros(len(v)), np.zeros(len(v))
    for tr, te in GroupKFold(5).split(X, y, g):
        m = HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.06, max_iter=250, class_weight="balanced",
            categorical_features=cat, random_state=0).fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        oof_base[te] = DummyClassifier(strategy="prior").fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    v["risco_ml"] = oof
    top10 = np.argsort(-oof)[: max(1, int(0.10 * len(v)))]
    metr = {
        "n": int(len(v)), "taxa_problema": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, oof)),
        "pr_auc": float(average_precision_score(y, oof)),
        "pr_auc_baseline": float(y.mean()),
        "captura_top10pct": float(y[top10].sum() / y.sum()),
    }
    return v.drop(columns=["problema"]).assign(problema=y), metr


def treinar_risco_arvore_final(v: pd.DataFrame) -> tuple[HistGradientBoostingClassifier, pd.DataFrame]:
    """Modelo final (todas as árvores) — usado para pontuar árvores novas."""
    v = v.dropna(subset=["estado"]).copy()
    v["problema"] = v["estado"].isin(PROBLEMA).astype(int)
    X = _feat_arvore(v)
    m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250,
                                       class_weight="balanced", random_state=0,
                                       categorical_features=[list(X.columns).index("genero_cod")])
    m.fit(X, v["problema"])
    return m, X


def fila_vistoria(v_risco: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    """Árvores AINDA classificadas como Excelente/Boa com maior risco previsto (vistoria preventiva)."""
    ok = v_risco[~v_risco["estado"].isin(PROBLEMA)]
    f = ok.sort_values("risco_ml", ascending=False).head(n).copy()
    f["google_maps"] = f.apply(lambda r: f"https://www.google.com/maps?q={r['lat']:.6f},{r['lon']:.6f}", axis=1)
    return f[["setor", "especie", "nome_popular", "estado", "altura_cm", "copa_cm", "colo", "risco_ml",
              "lat", "lon", "google_maps"]]


def surtos_fitossanitarios(v: pd.DataFrame, eps_m: float = 80.0, min_pontos: int = 6) -> pd.DataFrame:
    """DBSCAN nas árvores com problema: concentrações espaciais (possível praga/manejo ruim)."""
    p = v[v["estado"].isin(PROBLEMA)].copy()
    if len(p) < min_pontos:
        return pd.DataFrame()
    p["surto"] = DBSCAN(eps=eps_m, min_samples=min_pontos).fit_predict(p[["E", "N"]].to_numpy())
    p = p[p["surto"] >= 0]
    if p.empty:
        return pd.DataFrame()
    r = p.groupby("surto").agg(
        arvores_com_problema=("E", "size"), lat=("lat", "mean"), lon=("lon", "mean"),
        setor_principal=("setor", lambda s: s.value_counts().index[0]),
        especie_principal=("especie", lambda s: s.value_counts().index[0]),
        pct_especie_principal=("especie", lambda s: s.value_counts(normalize=True).iloc[0]),
    ).reset_index()
    return r.sort_values("arvores_com_problema", ascending=False).reset_index(drop=True)


def revisar_rotulos(d: pd.DataFrame, codificar, features_modelo, limite: float = 0.03) -> tuple[pd.DataFrame, dict]:
    """Probabilidades fora-da-amostra (validação por setor). Pontos em que o modelo dá
    P(porte levantado) < `limite` são candidatos a conferência: ou o levantamento tem erro,
    ou há algo no local que as 7 variáveis não captam."""
    from sklearn.ensemble import RandomForestClassifier
    X, y, g = codificar(d), d["porte"].to_numpy(), d["setor"].to_numpy()
    classes = sorted(set(y))
    P = np.zeros((len(d), len(classes)))
    for tr, te in GroupKFold(5).split(X, y, g):
        m = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced",
                                   random_state=42, n_jobs=-1).fit(X.iloc[tr], y[tr])
        P[te] = m.predict_proba(X.iloc[te])
    pos = {c: i for i, c in enumerate(classes)}
    p_lab = P[np.arange(len(d)), [pos[v] for v in y]]
    out = d.copy()
    out["p_porte_levantado"] = p_lab
    out["porte_modelo"] = [classes[i] for i in P.argmax(axis=1)]
    sus = out[out["p_porte_levantado"] < limite].copy()
    sus["google_maps"] = sus.apply(lambda r: f"https://www.google.com/maps?q={r['lat']:.6f},{r['lon']:.6f}", axis=1)
    sus = sus.sort_values("p_porte_levantado")
    resumo = {"limite": limite, "n_suspeitos": int(len(sus)), "pct_suspeitos": float(len(sus) / len(d)),
              "por_setor_top": sus["setor"].value_counts().head(5).to_dict()}
    cols = ["setor", "lat", "lon", "porte", "porte_modelo", "p_porte_levantado", "condicoes", "google_maps"]
    return sus[cols].reset_index(drop=True), resumo


def anomalias_inventario(vivas: pd.DataFrame, contaminacao: float = 0.004) -> pd.DataFrame:
    """IsolationForest em (log altura, log copa, log copa/altura) + regras duras (> 30 m / copa > 25 m)."""
    v = vivas.dropna(subset=["altura_cm", "copa_cm"]).copy()
    v = v[(v["altura_cm"] > 0) & (v["copa_cm"] > 0)]
    X = np.column_stack([np.log(v["altura_cm"]), np.log(v["copa_cm"]), np.log(v["copa_cm"] / v["altura_cm"])])
    iso = IsolationForest(n_estimators=300, contamination=contaminacao, random_state=0).fit(X)
    v["anomalia_score"] = -iso.score_samples(X)
    v["suspeita"] = iso.predict(X) == -1
    v["motivo"] = np.where(v["altura_cm"] > 3000, "altura > 30 m",
                  np.where(v["copa_cm"] > 2500, "copa > 25 m",
                  np.where(v["copa_cm"] / v["altura_cm"] > 4, "copa 4× maior que a altura",
                  np.where(v["copa_cm"] / v["altura_cm"] < 0.1, "copa < 10% da altura", "combinação atípica"))))
    s = v[v["suspeita"]].sort_values("anomalia_score", ascending=False)
    s["google_maps"] = s.apply(lambda r: f"https://www.google.com/maps?q={r['lat']:.6f},{r['lon']:.6f}", axis=1)
    return s[["setor", "especie", "altura_cm", "copa_cm", "motivo", "anomalia_score", "google_maps"]].reset_index(drop=True)



def auditar_catalogo(esp: pd.DataFrame, min_n: int = 15) -> tuple[pd.DataFrame, dict]:
    """Classificador de porte (altura e copa medianas observadas), avaliado por leave-one-out.
    Espécies em que P(porte do catálogo) é baixa merecem revisão — com a ressalva de que
    o inventário contém árvores jovens, então altura/copa observadas SUBESTIMAM o porte adulto."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    e = esp[(esp["n_inventario"] >= min_n)].dropna(subset=["altura_med_cm", "copa_med_cm"]).copy()
    X = np.log(e[["altura_med_cm", "copa_med_cm"]].to_numpy())
    y = e["porte"].to_numpy()
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    P = cross_val_predict(m, X, y, cv=LeaveOneOut(), method="predict_proba")
    classes = sorted(set(y))
    pos = {c: i for i, c in enumerate(classes)}
    e["p_porte_catalogo"] = P[np.arange(len(e)), [pos[v] for v in y]]
    e["porte_observado"] = [classes[i] for i in P.argmax(axis=1)]
    e["divergente"] = e["porte_observado"] != e["porte"]
    acc = float((~e["divergente"]).mean())
    saida = e.sort_values("p_porte_catalogo")[
        ["especie", "nome_popular", "n_inventario", "porte", "porte_observado", "p_porte_catalogo",
         "altura_med_cm", "copa_med_cm"]].reset_index(drop=True)
    return saida, {"n_especies": int(len(e)), "concordancia": acc, "n_divergentes": int(e["divergente"].sum())}



def _feat_canteiro(d: pd.DataFrame, codificar) -> pd.DataFrame:
    return codificar(d)


def avaliar_canteiro(d: pd.DataFrame, codificar) -> dict:
    """MAE (cm) por validação cruzada por setor, contra baseline (mediana) e regra simples."""
    x = d.dropna(subset=["canteiro_cm"])
    X, y, g = _feat_canteiro(x, codificar), x["canteiro_cm"].to_numpy(), x["setor"].to_numpy()
    oof, base = np.zeros(len(x)), np.zeros(len(x))
    for tr, te in GroupKFold(5).split(X, y, g):
        oof[te] = HistGradientBoostingRegressor(max_depth=5, learning_rate=0.06, max_iter=200,
                                                random_state=0).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        base[te] = DummyRegressor(strategy="median").fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    return {"n": int(len(x)), "mae_modelo_cm": float(mean_absolute_error(y, oof)),
            "mae_baseline_cm": float(mean_absolute_error(y, base)),
            "r2": float(1 - ((y - oof) ** 2).sum() / ((y - y.mean()) ** 2).sum())}


def treinar_canteiro(d: pd.DataFrame, codificar) -> HistGradientBoostingRegressor:
    x = d.dropna(subset=["canteiro_cm"])
    return HistGradientBoostingRegressor(max_depth=5, learning_rate=0.06, max_iter=200,
                                         random_state=0).fit(codificar(x), x["canteiro_cm"])


def prever_canteiro(modelo, entrada: dict, codificar) -> float:
    return float(modelo.predict(codificar(pd.DataFrame([entrada])))[0])


def _matriz_esp_setor(vivas: pd.DataFrame) -> pd.DataFrame:
    m = vivas.pivot_table(index="especie", columns="setor", values="E", aggfunc="size", fill_value=0)
    return m.div(m.sum(axis=1), axis=0)  # onde cada espécie está distribuída


def _cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-9)
    return A @ B.T


def especies_parecidas(esp: pd.DataFrame, vivas: pd.DataFrame, especie: str, n: int = 6,
                       mesmo_porte: bool = True, so_nao_saturadas: bool = True) -> pd.DataFrame:
    """Item-item: combina (a) traços (porte, família, gênero, altura/copa) e (b) padrão de ocorrência
    nos setores. Serve para achar substitutas de espécies saturadas ou indisponíveis no viveiro."""
    e = esp.reset_index(drop=True)
    if especie not in set(e["especie"]):
        return pd.DataFrame()
    T = pd.get_dummies(e[["porte", "familia", "genero"]].fillna("?"), dtype=float)
    T["porte"] = 0.0  # evita dupla contagem; porte filtra abaixo
    w = np.array([1.0 if c.startswith("familia_") else 2.0 if c.startswith("genero_") else 0.0 for c in T.columns])
    Tm = T.to_numpy() * w
    for col in ("altura_med_cm", "copa_med_cm"):
        z = e[col].fillna(e.groupby("porte")[col].transform("median")).fillna(e[col].median())
        Tm = np.column_stack([Tm, (z - z.mean()) / (z.std() + 1e-9) * 0.7])
    S = _matriz_esp_setor(vivas).reindex(e["especie"]).fillna(0).to_numpy()
    tem = S.sum(axis=1) > 0
    sim_traco = _cos(Tm, Tm)
    sim_setor = np.where(np.outer(tem, tem), _cos(S, S), 0.0)
    sim = 0.6 * sim_traco + 0.4 * sim_setor
    i = int(e.index[e["especie"] == especie][0])
    r = e.assign(similaridade=sim[i]).drop(index=i)
    if mesmo_porte:
        r = r[r["porte"] == e.loc[i, "porte"]]
    if so_nao_saturadas:
        r = r[~r["saturada"]]
    r = r.sort_values("similaridade", ascending=False).head(n)
    return r[["especie", "nome_popular", "porte", "familia", "n_inventario", "similaridade"]]


def setores_parecidos(vivas: pd.DataFrame, setor: str, n: int = 5) -> pd.DataFrame:
    comp = vivas.pivot_table(index="setor", columns="especie", values="E", aggfunc="size", fill_value=0)
    comp = comp[comp.sum(axis=1) >= 10]
    if setor not in comp.index:
        return pd.DataFrame()
    P = comp.div(comp.sum(axis=1), axis=0)
    s = pd.Series(_cos(P.to_numpy(), P.loc[[setor]].to_numpy())[:, 0], index=P.index).drop(setor)
    return s.sort_values(ascending=False).head(n).rename("similaridade").reset_index()


def recomendar_por_setores_parecidos(esp: pd.DataFrame, vivas: pd.DataFrame, setor: str,
                                     n_viz: int = 5, top: int = 10) -> pd.DataFrame:
    """Filtragem colaborativa (setor-setor): espécies que vão bem nos setores mais parecidos
    (baixo risco fitossanitário estimado) mas que o setor ainda quase não tem."""
    viz = setores_parecidos(vivas, setor, n_viz)
    if viz.empty:
        return pd.DataFrame()
    sub = vivas[vivas["setor"].isin(viz["setor"])]
    peso = sub["setor"].map(viz.set_index("setor")["similaridade"])
    presenca = sub.assign(w=peso).groupby("especie")["w"].sum()
    local = vivas[vivas["setor"] == setor]["especie"].value_counts(normalize=True)
    r = esp[~esp["saturada"] & esp["especie"].isin(presenca.index)].copy()
    r["afinidade"] = r["especie"].map(presenca) / presenca.sum()
    r["share_no_setor"] = r["especie"].map(local).fillna(0.0)
    r["lacuna"] = (r["afinidade"] - r["share_no_setor"]).clip(lower=0)
    r["score"] = r["lacuna"] / (r["lacuna"].max() + 1e-12) * (1 - np.minimum(r["risco_eb"] / 0.10, 1) * 0.5)
    return r.sort_values("score", ascending=False).head(top)[
        ["especie", "nome_popular", "porte", "afinidade", "share_no_setor", "risco_eb", "score"]]


def _dist_rota(xy: np.ndarray, ordem: list[int]) -> float:
    return float(sum(np.hypot(*(xy[ordem[i + 1]] - xy[ordem[i]])) for i in range(len(ordem) - 1)))


def _rota(xy: np.ndarray) -> list[int]:
    """Vizinho mais próximo a partir do ponto mais a oeste + melhoria 2-opt."""
    n = len(xy)
    if n <= 2:
        return list(range(n))
    atual = int(np.argmin(xy[:, 0]))
    ordem, livre = [atual], set(range(n)) - {atual}
    while livre:
        prox = min(livre, key=lambda j: np.hypot(*(xy[j] - xy[atual])))
        ordem.append(prox)
        livre.remove(prox)
        atual = prox
    melhorou = True
    while melhorou:
        melhorou = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                nova = ordem[:i] + ordem[i:j + 1][::-1] + ordem[j + 1:]
                if _dist_rota(xy, nova) + 1e-9 < _dist_rota(xy, ordem):
                    ordem, melhorou = nova, True
    return ordem


def dividir_equipes(plano: pd.DataFrame, n_equipes: int, mudas_por_equipe_dia: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide o plano em equipes geograficamente compactas (K-Means) e ordena cada rota.
    Devolve (plano com equipe/dia/ordem, resumo por equipe com distância a pé estimada)."""
    if plano.empty:
        return plano, pd.DataFrame()
    n_equipes = int(max(1, min(n_equipes, len(plano))))
    p = plano.reset_index(drop=True).copy()
    xy = p[["E", "N"]].to_numpy()
    p["equipe"] = (KMeans(n_clusters=n_equipes, n_init=10, random_state=0).fit_predict(xy) + 1
                   if n_equipes > 1 else 1)
    linhas, resumo = [], []
    for eq, g in p.groupby("equipe"):
        pos = _rota(g[["E", "N"]].to_numpy())
        g = g.iloc[pos].copy()
        g["ordem"] = np.arange(1, len(g) + 1)
        g["dia"] = (g["ordem"] - 1) // mudas_por_equipe_dia + 1
        xy_g = g[["E", "N"]].to_numpy()
        linhas.append(g)
        resumo.append({"equipe": int(eq), "mudas": int(len(g)), "dias": int(g["dia"].max()),
                       "distancia_km": round(_dist_rota(xy_g, list(range(len(g)))) / 1000, 2)})
    return pd.concat(linhas).sort_values(["equipe", "ordem"]).reset_index(drop=True), pd.DataFrame(resumo)
