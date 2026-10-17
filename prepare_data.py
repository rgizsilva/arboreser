
from __future__ import annotations

import difflib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from logic import PORTES, codificar, decidir_porte, FEATURES
import ml_extra as X

PASTA_DADOS = Path("dados")
PASTA_SAIDA = Path("dados_processados")
PASTA_SAIDA.mkdir(exist_ok=True)


_UTM = Transformer.from_crs("epsg:32723", "epsg:4326", always_xy=True)


NAO_JUNTAR = {
    ("Callistemon imperialis", "Callistemon viminalis"),
    ("Inga uruguaiensis", "Eugenia uruguaiensis"),
    ("Tabebuia rosea", "Tabebuia roseoalba"),
}
relatorio: dict[str, object] = {}


def utm(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    lon, lat = _UTM.transform(df[x].to_numpy(float), df[y].to_numpy(float))
    df["lat"], df["lon"] = lat, lon
    return df


def limpar_texto(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)



def catalogo() -> pd.DataFrame:
    e = pd.read_excel(PASTA_DADOS / "ANEXO_VIII_-_Lista_de_espécies.xlsx")
    e.columns = ["especie", "porte"]
    e["especie"] = limpar_texto(e["especie"])
    e["porte"] = limpar_texto(e["porte"]).str.capitalize()
    sem_porte = e["porte"].isna().sum()
    e = e.dropna(subset=["especie", "porte"])

    conflito = e.groupby("especie")["porte"].nunique()
    conflito = conflito[conflito > 1].index.tolist()
    ordem = {p: i for i, p in enumerate(PORTES)}
    # Na divergência, vale o MENOR porte (decisão conservadora).
    e["_o"] = e["porte"].map(ordem)
    e = e.sort_values("_o").drop_duplicates("especie").drop(columns="_o")
    relatorio["catalogo_conflitos_de_porte"] = conflito
    relatorio["catalogo_linhas_sem_porte"] = int(sem_porte)
    return e.reset_index(drop=True)


def inventario(cat: pd.DataFrame) -> pd.DataFrame:
    i = pd.read_excel(PASTA_DADOS / "Anexo_III-Inventário.xlsx", skiprows=1)
    i = i.rename(columns={
        "Setor": "setor", "Latitude": "E", "Longitude": "N", "Família": "familia",
        "Espécie": "especie", "Nome popular": "nome_popular", "Altura (cm)": "altura_cm",
        "Copa (tamanho em cm)": "copa_cm", "Estado Fitossanitário": "estado",
        "Inclinação": "inclinacao", "Copa (Tipo)": "copa_tipo", "Colo": "colo",
    })
    i["copa_tipo"] = limpar_texto(i["copa_tipo"]).str.capitalize()
    i["colo"] = limpar_texto(i["colo"]).str.capitalize()
    i["inclinacao"] = limpar_texto(i["inclinacao"]).str.extract(r"Faixa (\d)")[0].astype(float)
    i["setor"] = limpar_texto(i["setor"]).str.upper().replace({"CENTRAL": "Central"})
    i["especie"] = limpar_texto(i["especie"])
    i["familia"] = limpar_texto(i["familia"])
    i["estado"] = limpar_texto(i["estado"]).str.capitalize()

    
    fora = (i["E"] > 230000) | (i["N"] > 7_680_000) | i[["E", "N"]].isna().any(axis=1)
    relatorio["inventario_sem_coordenada_ou_fora_da_cidade"] = int(fora.sum())
    i = i[~fora].copy()

    # Árvores mortas — são "vagas" de reposição, não espécies
    i["viva"] = ~i["especie"].str.lower().isin(["morta"])
    relatorio["inventario_arvores_mortas"] = int((~i["viva"]).sum())
    i.loc[~i["viva"], "especie"] = "Morta"

    
    nomes_inv = set(i.loc[i["viva"], "especie"].dropna())
    nomes_cat = set(cat["especie"])
    mapa = {}
    for c in sorted(nomes_cat - nomes_inv):
        for cand in difflib.get_close_matches(c, nomes_inv - nomes_cat, n=1, cutoff=0.88):
            if (c, cand) not in NAO_JUNTAR:
                mapa[cand] = c
    mapa = {k: v for k, v in mapa.items() if k in nomes_inv}
    i["especie"] = i["especie"].replace(mapa)
    relatorio["grafias_unificadas (inventário -> catálogo)"] = mapa

    i["genero"] = i["especie"].str.split().str[0]
    # família faltante: usa a mais comum do gênero
    fam_gen = (i.dropna(subset=["familia"]).groupby("genero")["familia"]
                 .agg(lambda s: s.value_counts().index[0]))
    i["familia"] = i["familia"].fillna(i["genero"].map(fam_gen))
    i = utm(i, "E", "N")
    return i[["setor", "E", "N", "lat", "lon", "familia", "genero", "especie", "nome_popular",
              "altura_cm", "copa_cm", "estado", "viva", "inclinacao", "copa_tipo", "colo"]].reset_index(drop=True)


def deficit() -> pd.DataFrame:
    d = pd.read_excel(PASTA_DADOS / "Dados_deficit_jan22.xlsx", sheet_name="Dados_deficit")
    n0 = len(d)
    d = d.rename(columns={
        "Setor": "setor", "Latitude": "E", "Longitude": "N", "Largura do calçamento (cm)": "larg",
        "Porte da espécie para plantio no local": "porte",
        "Dimensão máxima do canteiro livre sugerido (cm)": "canteiro_cm",
    })
    d["setor"] = limpar_texto(d["setor"]).str.upper().replace({"CENTRAL": "Central"})
    d["porte"] = limpar_texto(d["porte"]).str.capitalize()
    relatorio["deficit_linhas_sem_porte"] = int(d["porte"].isna().sum())
    d = d.dropna(subset=["porte"])

    cap = lambda c: limpar_texto(d[c]).str.capitalize()
    d["tipo_fiacao"] = limpar_texto(d["Tipo de fiação elétrica"])
    d["protecao"] = limpar_texto(d["Proteção da fiação"]).fillna("Não se aplica")
    d["telefonia"] = cap("Fiação aérea (Telefonia/ TV a cabo)").fillna("Ausente")
    d["edificacao"] = (d["Presença de Edificação"].astype(str).str.upper()
                       .map({"TRUE": "Presente", "VERDADEIRO": "Presente",
                             "FALSE": "Ausente", "FALSO": "Ausente"}).fillna("Presente"))
    d["tipo_edificacao"] = limpar_texto(d["Tipo de Edificação"])
    d["recuo"] = limpar_texto(d["Recuo"]).fillna("Não se aplica")
    d["larg"] = d["larg"].fillna(d["larg"].median())

    
    dup = d.duplicated(subset=["E", "N"], keep="first")
    relatorio["deficit_pontos_duplicados_removidos"] = int(dup.sum())
    d = d[~dup].copy()

    elem = (d["Elementos Urbanos"].astype("string").str.replace(";#", ", ", regex=False)
            .str.strip().str.capitalize().fillna("Nenhum"))
    d["elementos_urbanos"] = elem
    d["condicoes"] = (
        "Fiação: " + d["tipo_fiacao"].str.replace(r" \(.*\)", "", regex=True)
        + np.where(d["protecao"] == "Não se aplica", "", " (" + d["protecao"].str.lower() + ")")
        + " · calçada " + d["larg"].round().astype(int).astype(str) + " cm"
        + " · recuo " + d["recuo"].str.lower() + " · " + d["tipo_edificacao"].str.lower()
    )
    d = utm(d, "E", "N")
    relatorio["deficit_linhas_originais"] = n0
    relatorio["deficit_linhas_finais"] = int(len(d))
    cols = ["setor", "E", "N", "lat", "lon", "larg", "tipo_fiacao", "protecao", "telefonia",
            "edificacao", "tipo_edificacao", "recuo", "canteiro_cm", "elementos_urbanos",
            "condicoes", "porte"]
    return d[cols].reset_index(drop=True)



def estatisticas_especies(cat: pd.DataFrame, inv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    v = inv[inv["viva"]].copy()
    total = len(v)
    por_esp = v.groupby("especie").agg(
        n_inventario=("especie", "size"), nome_popular=("nome_popular", lambda s: s.dropna().mode().iat[0] if s.notna().any() else ""),
        familia=("familia", lambda s: s.dropna().mode().iat[0] if s.notna().any() else None),
        pct_problema=("estado", lambda s: s.isin(["Regular", "Ruim", "Péssima"]).mean()),
        altura_med_cm=("altura_cm", "median"), copa_med_cm=("copa_cm", "median"),
    ).reset_index()
    por_esp["genero"] = por_esp["especie"].str.split().str[0]
    por_esp["share_especie"] = por_esp["n_inventario"] / total

    gen = v.groupby("genero").size().rename("n").reset_index()
    gen["share"] = gen["n"] / total
    fam = v.dropna(subset=["familia"]).groupby("familia").size().rename("n").reset_index()
    fam["share"] = fam["n"] / total

    conc = pd.concat([
        por_esp.rename(columns={"especie": "nome", "n_inventario": "n", "share_especie": "share"})
               [["nome", "n", "share"]].assign(nivel="especie"),
        gen.rename(columns={"genero": "nome"}).assign(nivel="genero"),
        fam.rename(columns={"familia": "nome"}).assign(nivel="familia"),
    ], ignore_index=True)

    fam_do_genero = por_esp.dropna(subset=["familia"]).groupby("genero")["familia"].agg(lambda s: s.mode().iat[0])
    e = cat.merge(por_esp.drop(columns="genero"), on="especie", how="left")
    e["genero"] = e["especie"].str.split().str[0]
    e["familia"] = e["familia"].fillna(e["genero"].map(fam_do_genero))
    e["n_inventario"] = e["n_inventario"].fillna(0).astype(int)
    e["nome_popular"] = e["nome_popular"].fillna("")
    e["share_especie"] = e["share_especie"].fillna(0.0)
    e["share_genero"] = e["genero"].map(gen.set_index("genero")["share"]).fillna(0.0)
    e["share_familia"] = e["familia"].map(fam.set_index("familia")["share"]).fillna(0.0)
    e["saturada"] = ((e["share_especie"] >= 0.10) | (e["share_genero"] >= 0.20) | (e["share_familia"] >= 0.30))
    relatorio["catalogo_especies_sem_historico_no_inventario"] = int((e["n_inventario"] == 0).sum())
    relatorio["catalogo_especies_saturadas (10-20-30)"] = e.loc[e["saturada"], "especie"].tolist()
    return e, conc


def validar_modelo(d: pd.DataFrame) -> dict:
    X, y, g = codificar(d), d["porte"].to_numpy(), d["setor"].to_numpy()
    classes = sorted(set(y))  
    probs = np.zeros((len(d), len(classes)))
    for tr, te in GroupKFold(5).split(X, y, g):
        m = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced",
                                   random_state=42, n_jobs=-1).fit(X.iloc[tr], y[tr])
        probs[te] = m.predict_proba(X.iloc[te])
    ordem = {p: i for i, p in enumerate(PORTES)}
    real = np.array([ordem[v] for v in y])
    saida = {"n": int(len(d)), "por_tau": {}}
    for tau in (0.5, 0.6, 0.7):
        pred = np.array([ordem[decidir_porte(dict(zip(classes, p)), tau)] for p in probs])
        saida["por_tau"][str(tau)] = {
            "acuracia": float((pred == real).mean()),
            "porte_maior_que_o_real": float((pred > real).mean()),
            "porte_menor_que_o_real": float((pred < real).mean()),
            "f1_macro": float(f1_score(real, pred, average="macro")),
        }
    return saida


NOMES_LEGIVEIS = {
    "larg": "largura da calçada (cm)", "tipo_fiacao": "tipo de fiação elétrica",
    "protecao": "proteção da fiação", "telefonia": "fiação de telefonia/TV",
    "edificacao": "presença de edificação", "tipo_edificacao": "tipo de edificação", "recuo": "recuo",
}


def _binarias(d: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "largura_calcada_cm": d["larg"].astype(float),
        "fiacao_desprotegida": (d["protecao"] == "Desprotegida").astype(int),
        "fiacao_protegida": (d["protecao"] == "Protegida").astype(int),
        "fiacao_primaria": d["tipo_fiacao"].str.contains("Primária").astype(int),
        "recuo_presente": (d["recuo"] == "Presente").astype(int),
        "edificacao_media_ou_alta": d["tipo_edificacao"].isin(["Média (12<E<50m)", "Alta (>50m)"]).astype(int),
        "sem_edificacao": (d["tipo_edificacao"] == "Não se aplica").astype(int),
        "telefonia_presente": (d["telefonia"] == "Presente").astype(int),
    })


def laboratorio_ml(d: pd.DataFrame) -> dict:
    X, y, g = codificar(d), d["porte"].to_numpy(), d["setor"].to_numpy()
    ordem = {p: i for i, p in enumerate(PORTES)}
    real = np.array([ordem[v] for v in y])
    cv = list(GroupKFold(5).split(X, y, g))

    modelos = {
        "Baseline (sempre a classe mais comum)": DummyClassifier(strategy="most_frequent"),
        "Regressão logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Árvore de decisão (prof. 6)": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting (HistGB)": HistGradientBoostingClassifier(class_weight="balanced", random_state=42),
    }
    comp, cm_rf = [], None
    for nome, m in modelos.items():
        pred = np.empty(len(d), dtype=object)
        for tr, te in cv:
            pred[te] = m.fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        p = np.array([ordem[v] for v in pred])
        comp.append({"modelo": nome, "acuracia": float((p == real).mean()),
                     "acuracia_balanceada": float(balanced_accuracy_score(real, p)),
                     "f1_macro": float(f1_score(real, p, average="macro")),
                     "porte_maior_que_o_real": float((p > real).mean())})
        if nome == "Random Forest":
            cm_rf = confusion_matrix(real, p, labels=[0, 1, 2]).tolist()

    grade = []
    for leaf in (1, 5, 20):
        for prof in (8, None):
            m = RandomForestClassifier(n_estimators=100, min_samples_leaf=leaf, max_depth=prof,
                                       class_weight="balanced", random_state=42, n_jobs=-1)
            pred = np.empty(len(d), dtype=object)
            for tr, te in cv:
                pred[te] = m.fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            p = np.array([ordem[v] for v in pred])
            grade.append({"min_samples_leaf": leaf, "max_depth": "sem limite" if prof is None else prof,
                          "f1_macro": float(f1_score(real, p, average="macro")),
                          "acuracia": float((p == real).mean())})

    
    imp = np.zeros(len(FEATURES))
    for tr, te in cv:
        m = RandomForestClassifier(n_estimators=200, min_samples_leaf=5, class_weight="balanced",
                                   random_state=42, n_jobs=-1).fit(X.iloc[tr], y[tr])
        r = permutation_importance(m, X.iloc[te], y[te], scoring="f1_macro", n_repeats=5, random_state=0)
        imp += r.importances_mean / len(cv)
    importancia = {NOMES_LEGIVEIS[f]: float(v) for f, v in zip(FEATURES, imp)}

    
    B = _binarias(d)
    arv = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, class_weight="balanced", random_state=42)
    pred = np.empty(len(d), dtype=object)
    for tr, te in cv:
        pred[te] = arv.fit(B.iloc[tr], y[tr]).predict(B.iloc[te])
    acc_arvore = float((pred == y).mean())
    arv.fit(B, y)
    regras = export_text(arv, feature_names=list(B.columns), class_names=None)
    return {"comparacao": comp, "matriz_confusao_rf": cm_rf, "ajuste_hiperparametros": grade,
            "importancia_permutacao": importancia, "regras_arvore": regras,
            "acuracia_arvore_interpretavel": acc_arvore, "classes_matriz": PORTES}


def ml_extra(inv: pd.DataFrame, d: pd.DataFrame, esp: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Módulos adicionais de ML (ver ml_extra.py). Devolve (especies atualizado, métricas, tabelas)."""
    vivas = inv[inv["viva"]].copy()
    vivas["problema"] = vivas["estado"].isin(X.PROBLEMA).astype(int)
    vivas = vivas.dropna(subset=["estado"])

    print("  · risco hierárquico por espécie...")
    mod = X.risco_hierarquico(vivas)
    esp = X.estimar_risco(esp, mod)
    met = {"risco_hierarquico_validacao": X.validar_risco(vivas)}

    print("  · risco por árvore (fila de vistoria)...")
    v_risco, m_risco = X.modelo_risco_arvore(vivas)
    met["risco_arvore"] = m_risco
    fila = X.fila_vistoria(v_risco, 300)
    surtos = X.surtos_fitossanitarios(vivas)

    print("  · revisão de rótulos do déficit...")
    rotulos, m_rot = X.revisar_rotulos(d, codificar, FEATURES)
    met["revisao_rotulos"] = m_rot

    print("  · anomalias do inventário e auditoria do catálogo...")
    anom = X.anomalias_inventario(vivas)
    met["anomalias_inventario"] = {"n": int(len(anom)), "motivos": anom["motivo"].value_counts().to_dict()}
    aud, m_aud = X.auditar_catalogo(esp)
    met["auditoria_catalogo"] = m_aud

    print("  · canteiro livre (regressão)...")
    met["canteiro"] = X.avaliar_canteiro(d, codificar)

    met["surtos"] = {"n_surtos": int(len(surtos)),
                     "arvores_em_surtos": int(surtos["arvores_com_problema"].sum()) if len(surtos) else 0}
    tabelas = {"fila_vistoria": fila, "surtos": surtos, "rotulos_suspeitos": rotulos,
               "anomalias": anom, "auditoria_catalogo": aud}
    return esp, met, tabelas


def main():
    print("Lendo e limpando...")
    cat = catalogo()
    inv = inventario(cat)
    d = deficit()
    esp, conc = estatisticas_especies(cat, inv)

    print("Validando o modelo (validação cruzada por setor)...")
    met = validar_modelo(d)
    met["relatorio_dados"] = relatorio
    met["distribuicao_porte_deficit"] = d["porte"].value_counts().to_dict()
    met["features"] = FEATURES
    print("Rodando laboratório de ML (comparação de modelos, ajuste, importância)...")
    met["laboratorio"] = laboratorio_ml(d)
    print("Rodando módulos adicionais de ML...")
    esp, met_x, tabelas = ml_extra(inv, d, esp)
    met["extra"] = met_x

    d.to_parquet(PASTA_SAIDA / "deficit.parquet", index=False)
    inv.to_parquet(PASTA_SAIDA / "arvores.parquet", index=False)
    esp.to_parquet(PASTA_SAIDA / "especies.parquet", index=False)
    conc.to_parquet(PASTA_SAIDA / "concentracao.parquet", index=False)
    for nome, t in tabelas.items():
        t.to_parquet(PASTA_SAIDA / f"{nome}.parquet", index=False)
    (PASTA_SAIDA / "metricas.json").write_text(json.dumps(met, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n=== RELATÓRIO DE QUALIDADE DOS DADOS ===")
    for k, v in relatorio.items():
        print(f"- {k}: {v}")
    print("\n=== VALIDAÇÃO DO MODELO ===")
    for tau, m in met["por_tau"].items():
        print(f"tau={tau}: " + ", ".join(f"{k}={v:.3f}" for k, v in m.items()))
    print("\n=== ML ADICIONAL ===")
    print(json.dumps(met_x, ensure_ascii=False, indent=1, default=str))
    print("\nPronto. Rode: streamlit run app.py")


if __name__ == "__main__":
    main()
