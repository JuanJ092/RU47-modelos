"""
PASO 4 - Rigor estadistico.

  python paso4_rigor.py            (todo; el NFG de RU47 tarda varios minutos)
  python paso4_rigor.py --rapido   (omite el NFG de RU47 en los modelos nulos)

Requiere: pip install powerlaw
  A) Modelos nulos (Newman cap. 13): grafos aleatorios dirigidos con los mismos grados de entrada y salida
     (modelo de configuracion) y los pesos barajados. Se compara clustering, camino medio y modularidad.
  B) Distribuciones de grado y de fuerza (grado ponderado): ajuste de ley de potencia (Clauset, Shalizi y
     Newman 2009) y prueba de razon de verosimilitud contra lognormal y exponencial.
  C) Robustez del MFG frente a la solucion de flujos: FBA, pFBA y promedio de MFG de flujos muestreados
     (como el promedio de 442 MFG del articulo).
Genera en 'resultados': paso4_nulos.csv, paso4_distribuciones.csv, paso4_distribuciones.png,
  paso4_robustez_fba.csv, paso4_robustez_fba.png
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from scipy.stats import spearmanr
import cobra
import flujo as F

warnings.filterwarnings("ignore")
OUT = "resultados"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(F.SEMILLA)


# ------------------------------------------------------------------ A) modelos nulos
def metricas(G, grande):
    U = nx.Graph(G)
    if grande:
        cl = nx.approximation.average_clustering(U, trials=20000, seed=F.SEMILLA)
    else:
        cl = nx.average_clustering(U)
    scc = max(nx.strongly_connected_components(G), key=len)
    H = G.subgraph(scc)
    fuentes = list(H.nodes)
    if len(fuentes) > 300:
        fuentes = list(rng.choice(fuentes, 300, replace=False))
    d = []
    for s in fuentes:
        d += [v for k, v in nx.single_source_shortest_path_length(H, s).items() if k != s]
    com = nx.community.louvain_communities(G, weight="weight", seed=F.SEMILLA)
    return {"clustering": cl, "camino_medio": np.mean(d), "frac_comp_fuerte": len(scc) / G.number_of_nodes(),
            "modularidad": nx.community.modularity(G, com, weight="weight"), "n_comunidades": len(com)}


def nulo(G):
    nodos = list(G.nodes)
    kin = [G.in_degree(v) for v in nodos]
    kout = [G.out_degree(v) for v in nodos]
    M = nx.directed_configuration_model(kin, kout, seed=int(rng.integers(1e9)))
    R = nx.DiGraph(M)
    R.remove_edges_from(nx.selfloop_edges(R))
    pesos = rng.permutation([w for _, _, w in G.edges(data="weight")])
    for (u, v), w in zip(R.edges, pesos):
        R[u][v]["weight"] = float(w)
    return R


def modelos_nulos(grafos, n_nulos):
    filas = []
    for nombre, W, nodos in grafos:
        G = F.a_networkx(W, list(range(W.shape[0])))
        grande = W.shape[0] > 1500
        obs = metricas(G, grande)
        nul = [metricas(nulo(G), grande) for _ in range(n_nulos[grande])]
        for k, v in obs.items():
            vals = np.array([x[k] for x in nul])
            sd = vals.std(ddof=1)
            filas.append({"grafo": nombre, "metrica": k, "observado": v, "nulo_media": vals.mean(),
                          "nulo_desv": sd, "z": (v - vals.mean()) / sd if sd > 0 else np.nan,
                          "n_nulos": len(vals)})
        print(f"  nulos de {nombre}: " + ", ".join(
            f"{f['metrica']} obs={f['observado']:.3f} nulo={f['nulo_media']:.3f} z={f['z']:.1f}"
            for f in filas if f["grafo"] == nombre))
    return filas


# ------------------------------------------------------------------ B) distribuciones
def distribuciones(grafos):
    import powerlaw
    plt = F.estilo()
    filas = []
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for fila_ax, (nombre, W, _) in zip(axes, grafos):
        medidas = {
            "grado_entrada": np.asarray((W > 0).sum(0)).ravel(),
            "grado_salida": np.asarray((W > 0).sum(1)).ravel(),
            "fuerza_entrada": np.asarray(W.sum(0)).ravel(),
            "fuerza_salida": np.asarray(W.sum(1)).ravel(),
        }
        for ax, tipo in zip(fila_ax, ["grado", "fuerza"]):
            for direc, color in [("entrada", F.AZUL), ("salida", F.GRIS)]:
                x = medidas[f"{tipo}_{direc}"]
                x = x[x > 0]
                fit = powerlaw.Fit(x, discrete=(tipo == "grado"), verbose=False)
                R_ln, p_ln = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)
                R_ex, p_ex = fit.distribution_compare("power_law", "exponential", normalized_ratio=True)
                filas.append({"grafo": nombre, "medida": f"{tipo}_{direc}", "alfa": fit.power_law.alpha,
                              "xmin": fit.power_law.xmin, "n_cola": int((x >= fit.power_law.xmin).sum()),
                              "n_total": len(x), "KS": fit.power_law.D,
                              "R_vs_lognormal": R_ln, "p_vs_lognormal": p_ln,
                              "R_vs_exponencial": R_ex, "p_vs_exponencial": p_ex})
                fit.plot_ccdf(ax=ax, color=color, marker=".", ls="none", ms=3, label=f"{tipo} de {direc}")
                fit.lognormal.plot_ccdf(ax=ax, color=color, ls="--", lw=1)
            ax.set_title(f"{nombre}: {tipo}")
            ax.set_xlabel(tipo + (" (k)" if tipo == "grado" else " (suma de pesos)"))
            ax.set_ylabel("P(X >= x)")
            ax.legend(fontsize=7)
    fig.text(0.01, 0.005, "Puntos: datos. Lineas discontinuas: ajuste lognormal en la cola.", fontsize=7, color=F.TINTA2)
    fig.tight_layout()
    F.guardar_fig(fig, os.path.join(OUT, "paso4_distribuciones.png"))
    return filas


# ------------------------------------------------------------------ C) robustez frente a la solucion de flujos
def robustez_fba(model, n_muestras=200):
    D = F.Desdoblado(model)
    glc = {"EX_glc__D_e": F.GLUCOSA}
    W_p, n_p, mu, fl_p, _ = F.mfg_condicion(model, D, glc, metodo="pfba")
    W_f, n_f, _, fl_f, _ = F.mfg_condicion(model, D, glc, metodo="fba")
    with model:
        F.fijar_medio(model, extra=glc)
        obj = F.objetivo(model)
        model.reactions.get_by_id(obj).lower_bound = 0.99 * mu
        muestras = cobra.sampling.sample(model, n_muestras, method="optgp", processes=1, seed=F.SEMILLA)
    Wsum = None
    for _, fila in muestras.iterrows():
        M, _ = D.mfg(fila[[r.id for r in model.reactions]].values)
        Wsum = M if Wsum is None else Wsum + M
    W_m, n_m = F.submatriz_activa(Wsum / len(muestras), D.nodos, umbral=1e-9)

    prs = {k: dict(zip(n, F.pagerank(W))) for k, W, n in
           [("pFBA", W_p, n_p), ("FBA", W_f, n_f), ("muestreo", W_m, n_m)]}
    filas = []
    for a, b in [("pFBA", "FBA"), ("pFBA", "muestreo"), ("FBA", "muestreo")]:
        comunes = sorted(set(prs[a]) & set(prs[b]))
        top_a = set(sorted(prs[a], key=prs[a].get, reverse=True)[:20])
        top_b = set(sorted(prs[b], key=prs[b].get, reverse=True)[:20])
        filas.append({"comparacion": f"{a} vs {b}", "nodos_" + a: len(prs[a]), "nodos_" + b: len(prs[b]),
                      "jaccard_nodos": len(comunes) / len(set(prs[a]) | set(prs[b])),
                      "spearman_pagerank": spearmanr([prs[a][k] for k in comunes], [prs[b][k] for k in comunes])[0],
                      "coinciden_top20": len(top_a & top_b)})
    print("  robustez:", "; ".join(f"{f['comparacion']}: Spearman={f['spearman_pagerank']:.2f}, top20 en comun={f['coinciden_top20']}" for f in filas))

    plt = F.estilo()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, b in zip(axes, ["FBA", "muestreo"]):
        comunes = sorted(set(prs["pFBA"]) & set(prs[b]))
        x = np.array([prs["pFBA"][k] for k in comunes]); y = np.array([prs[b][k] for k in comunes])
        ax.loglog(x, y, ".", color=F.AZUL, ms=4, alpha=0.7)
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lim, lim, color=F.GRIS, lw=1)
        for j, k in enumerate(sorted(comunes, key=lambda k: -prs["pFBA"][k])[:4]):
            ax.annotate(k, (prs["pFBA"][k], prs[b][k]), fontsize=7, color=F.TINTA2,
                        xytext=(0.04, 0.9 - 0.08 * j), textcoords="axes fraction", ha="left", va="center",
                        arrowprops=dict(arrowstyle="-", color=F.GRIS, lw=0.6))
        ax.set_xlabel("PageRank en el MFG con pFBA")
        ax.set_ylabel(f"PageRank en el MFG con {b}")
        ax.set_title(f"pFBA frente a {b} (rho = {spearmanr(x, y)[0]:.2f})")
    fig.tight_layout()
    F.guardar_fig(fig, os.path.join(OUT, "paso4_robustez_fba.png"))
    return filas


def main():
    rapido = "--rapido" in sys.argv
    model = F.cargar_modelo("RU47_model.xml")
    D = F.Desdoblado(model)
    glc = {"EX_glc__D_e": F.GLUCOSA}
    W_mfg, n_mfg, *_ = F.mfg_condicion(model, D, glc)
    W_nfg, n_nfg = F.submatriz_activa(D.nfg(), D.nodos)
    eco, _ = F.ecoli_core()
    De = F.Desdoblado(eco)
    We_mfg, ne_mfg, *_ = F.mfg_condicion(eco, De, glc)
    We_nfg, ne_nfg = F.submatriz_activa(De.nfg(), De.nodos)

    grafos = [("Ecoli_NFG", We_nfg, ne_nfg), ("Ecoli_MFG_glucosa", We_mfg, ne_mfg), ("RU47_MFG_glucosa", W_mfg, n_mfg)]
    if not rapido:
        grafos.append(("RU47_NFG", W_nfg, n_nfg))
    print("A) modelos nulos ...")
    pd.DataFrame(modelos_nulos(grafos, n_nulos={False: 50, True: 10})).to_csv(
        os.path.join(OUT, "paso4_nulos.csv"), index=False)
    print("B) distribuciones ...")
    d = distribuciones([("RU47_NFG", W_nfg, n_nfg), ("RU47_MFG_glucosa", W_mfg, n_mfg)])
    pd.DataFrame(d).to_csv(os.path.join(OUT, "paso4_distribuciones.csv"), index=False)
    print("C) robustez frente a la solucion de flujos (muestreo, varios minutos) ...")
    pd.DataFrame(robustez_fba(model)).to_csv(os.path.join(OUT, "paso4_robustez_fba.csv"), index=False)
    print("Listo: resultados/paso4_*")


if __name__ == "__main__":
    main()
