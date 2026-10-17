import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import logic as L


@pytest.fixture(scope="module")
def b():
    return L.carregar_base()


def test_decidir_porte_e_conservador():
    assert L.decidir_porte({"Pequeno": .4, "Médio": .3, "Grande": .3}) == "Pequeno"
    assert L.decidir_porte({"Pequeno": .1, "Médio": .15, "Grande": .75}) == "Grande"
    assert L.decidir_porte({"Pequeno": .2, "Médio": .6, "Grande": .2}) == "Médio"


def test_fiacao_desprotegida_gera_porte_pequeno(b):
    porte, _ = L.prever_porte(b.modelo, dict(
        larg=250, tipo_fiacao="Primária e Secundária (13800V + 127V)", protecao="Desprotegida",
        telefonia="Presente", edificacao="Presente", tipo_edificacao="Baixa (<12m)", recuo="Ausente"))
    assert porte == "Pequeno"


def test_especie_saturada_nunca_e_recomendada(b):
    for porte in L.PORTES:
        assert not L.ranquear_especies(b, porte, top_n=500)["saturada"].any()
    assert "Moquilea tomentosa" in b.especies.loc[b.especies["saturada"], "especie"].tolist()


def test_ranking_respeita_porte(b):
    r = L.ranquear_especies(b, "Grande", top_n=50)
    assert set(r["porte"]) == {"Grande"}


def test_vizinhanca_penaliza_repeticao(b):
    ponto = b.deficit.iloc[0]
    sem = L.ranquear_especies(b, ponto["porte"], None, top_n=500).set_index("especie")["pontuacao"]
    viz = b.vivas[b.vivas["especie"].isin(sem.index)].head(1)
    esp = viz.iloc[0]["especie"]
    viz_rep = b.vivas[b.vivas["especie"] == esp].head(20)
    com = L.ranquear_especies(b, ponto["porte"], viz_rep, top_n=500).set_index("especie")["pontuacao"]
    assert com[esp] < sem[esp]


def test_locais_compativeis_com_porte(b):
    esp = b.especies[b.especies["porte"] == "Grande"].iloc[0]["especie"]
    pts, _ = L.locais_para_especie(b, esp)
    assert set(pts["porte"]) == {"Grande"}


def test_plano_respeita_teto_e_porte(b):
    plano = L.gerar_plano(b, "N-06", 50, max_share_plano=0.10)
    assert len(plano) > 0
    assert plano["especie"].value_counts().iloc[0] <= int(np.ceil(0.10 * len(plano)))
    assert "Moquilea tomentosa" not in set(plano["especie"])


def test_coordenadas_dentro_de_ribeirao(b):
    assert b.deficit["lat"].between(-21.35, -21.0).all()
    assert b.deficit["lon"].between(-48.1, -47.6).all()


def test_what_if_muda_porte_ao_desproteger_fiacao(b):
    e = dict(larg=250, tipo_fiacao="Secundária (127V)", protecao="Protegida", telefonia="Presente",
             edificacao="Presente", tipo_edificacao="Baixa (<12m)", recuo="Presente")
    w = L.analise_what_if(b.modelo, e)
    linha = w[(w["condição alterada"] == "Proteção da fiação") & (w["para"] == "Desprotegida")].iloc[0]
    assert linha["porte resultante"] == "Pequeno"


def test_kmeans_setores(b):
    setores, perfil, sil = L.agrupar_setores(b, 4)
    assert setores["grupo"].nunique() == 4 and -1 <= sil <= 1
    assert perfil["nº setores"].sum() == len(setores)


def test_dbscan_hotspots(b):
    pts, areas = L.hotspots(b, 60, 25)
    assert len(areas) > 0 and areas["pontos"].is_monotonic_decreasing
    assert (pts["area"] >= 0).sum() == areas["pontos"].sum()


def test_laboratorio_salvo_em_metricas(b):
    lab = b.metricas["laboratorio"]
    nomes = [m["modelo"] for m in lab["comparacao"]]
    assert "Random Forest" in nomes and len(nomes) == 5
    base_ = next(m for m in lab["comparacao"] if m["modelo"].startswith("Baseline"))
    rf = next(m for m in lab["comparacao"] if m["modelo"] == "Random Forest")
    assert rf["f1_macro"] > base_["f1_macro"]


import ml_extra as MX


def test_risco_hierarquico_cobre_catalogo_inteiro(b):
    e = b.especies
    assert e["risco_eb"].notna().all() and e["risco_eb"].between(0, 1).all()
    assert (e["risco_lo"] <= e["risco_eb"]).all() and (e["risco_eb"] <= e["risco_hi"]).all()
    sem = e[e["n_inventario"] == 0]
    assert set(sem["risco_fonte"]) <= {"gênero", "família", "cidade"}


def test_risco_hierarquico_melhora_especies_pouco_vistas(b):
    v = b.metricas["extra"]["risco_hierarquico_validacao"]
    assert v["logloss_hierarquico_especies_pouco_vistas"] < v["logloss_bruta_especies_pouco_vistas"]


def test_modelo_risco_arvore_supera_acaso(b):
    m = b.metricas["extra"]["risco_arvore"]
    assert m["roc_auc"] > 0.7 and m["pr_auc"] > 3 * m["pr_auc_baseline"]
    f = b.extras["fila_vistoria"]
    assert not f["estado"].isin(MX.PROBLEMA).any()


def test_canteiro_regressao_bate_baseline(b):
    c = b.metricas["extra"]["canteiro"]
    assert c["mae_modelo_cm"] < c["mae_baseline_cm"] / 3
    entrada = dict(larg=300, tipo_fiacao="Nenhum", protecao="Não se aplica", telefonia="Ausente",
                   edificacao="Presente", tipo_edificacao="Baixa (<12m)", recuo="Presente")
    assert 100 < MX.prever_canteiro(b.canteiro, entrada, L.codificar) < 260


def test_especies_parecidas_respeita_porte_e_saturacao(b):
    esp = b.especies[b.especies["n_inventario"] > 100].iloc[0]
    r = MX.especies_parecidas(b.especies, b.vivas, esp["especie"], n=5)
    assert len(r) == 5 and (r["porte"] == esp["porte"]).all() and esp["especie"] not in set(r["especie"])
    assert "Moquilea tomentosa" not in set(r["especie"])


def test_colaborativo_nao_recomenda_saturada(b):
    r = MX.recomendar_por_setores_parecidos(b.especies, b.vivas, "N-06")
    assert not r.empty and "Moquilea tomentosa" not in set(r["especie"])


def test_equipes_cobrem_o_plano_e_rota_nao_piora(b):
    plano = L.gerar_plano(b, "N-06", 40)
    p2, res = MX.dividir_equipes(plano, 3, 10)
    assert len(p2) == len(plano) and res["mudas"].sum() == len(plano) and p2["equipe"].nunique() == 3
    assert p2.groupby("equipe")["ordem"].apply(lambda s: list(s) == list(range(1, len(s) + 1))).all()
    import numpy as np
    xy = plano[["E", "N"]].to_numpy()
    assert MX._dist_rota(xy, MX._rota(xy)) <= MX._dist_rota(xy, list(range(len(xy)))) + 1e-6


def test_rotulos_suspeitos_tem_prob_baixa(b):
    assert (b.extras["rotulos_suspeitos"]["p_porte_levantado"] < 0.03).all()
