"""
PASO 5 - Robustez de la red (Newman cap. 15, percolacion) y relacion con la esencialidad (FBA).

  python paso5_robustez.py

  A) Se eliminan reacciones (nodos) al azar o en orden de importancia (PageRank o grado) y se mide la
     fraccion de reacciones que sigue en la componente gigante fuerte y debil. Indice de robustez R =
     area bajo la curva (Schneider et al. 2011): mas alto = mas robusta.
  B) Eliminacion de cada reaccion en el modelo (FBA): una reaccion es esencial si sin ella el crecimiento
     cae por debajo del 1 % del original. Se pregunta si las reacciones mas centrales en la red son las
     esenciales (area bajo la curva ROC del PageRank y del grado como predictores).
Genera en 'resultados': paso5_percolacion.csv, paso5_percolacion.png, paso5_esencialidad.csv,
  paso5_esencialidad_resumen.csv, paso5_esencialidad.png
"""
import os
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
import cobra
import flujo as F

OUT = "resultados"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(F.SEMILLA)
FRACS = np.round(np.arange(0, 0.51, 0.02), 2)


def gigantes(W, vivos, N):
    if len(vivos) == 0:
        return 0.0, 0.0
    S = W[vivos][:, vivos]
    _, ls = connected_components(S, directed=True, connection="strong")
    _, lw = connected_components(S, directed=True, connection="weak")
    return np.bincount(ls).max() / N, np.bincount(lw).max() / N


def percolar(nombre, W, n_azar=20):
    N = W.shape[0]
    pr = F.pagerank(W)
    grado = np.asarray((W > 0).sum(0)).ravel() + np.asarray((W > 0).sum(1)).ravel()
    ordenes = {"PageRank": np.argsort(-pr), "grado": np.argsort(-grado)}
    filas = []
    for estr, orden in ordenes.items():
        for f in FRACS:
            fuera = set(orden[: int(round(f * N))])
            vivos = np.array([i for i in range(N) if i not in fuera])
            sf, sw = gigantes(W, vivos, N)
            filas.append({"grafo": nombre, "estrategia": estr, "f": f, "rep": 0, "S_fuerte": sf, "S_debil": sw})
    for rep in range(n_azar):
        perm = rng.permutation(N)
        for f in FRACS:
            vivos = np.sort(perm[int(round(f * N)):])
            sf, sw = gigantes(W, vivos, N)
            filas.append({"grafo": nombre, "estrategia": "azar", "f": f, "rep": rep, "S_fuerte": sf, "S_debil": sw})
    return filas


def esencialidad(nombre, model, D, umbral=0.01):
    glc = {"EX_glc__D_e": F.GLUCOSA}
    with model:
        F.fijar_medio(model, extra=glc)
        mu0 = model.optimize().objective_value
        dele = cobra.flux_analysis.single_reaction_deletion(model, processes=1)
    dele["reaccion"] = [list(i)[0] for i in dele["ids"]]
    dele["crec_rel"] = dele["growth"].fillna(0) / mu0
    dele["esencial"] = dele["crec_rel"] < umbral
    ess = dict(zip(dele.reaccion, dele.esencial))

    W_m, n_m, *_ = F.mfg_condicion(model, D, glc)
    W_n, n_n = F.submatriz_activa(D.nfg(), D.nodos)
    tabla = pd.DataFrame({"reaccion": [r.id for r in model.reactions],
                          "esencial": [ess.get(r.id, False) for r in model.reactions],
                          "crec_rel": [dict(zip(dele.reaccion, dele.crec_rel)).get(r.id, np.nan) for r in model.reactions]})
    resumen = []
    for gname, W, nodos in [("MFG_glucosa", W_m, n_m), ("NFG", W_n, n_n)]:
        pr = F.pagerank(W)
        grado = np.asarray((W > 0).sum(0)).ravel() + np.asarray((W > 0).sum(1)).ravel()
        dpr, dgr = {}, {}
        for v, p, g in zip(nodos, pr, grado):   # una reaccion = maximo de sus dos sentidos
            r = F.rid(v)
            dpr[r] = max(dpr.get(r, 0), p)
            dgr[r] = max(dgr.get(r, 0), g)
        tabla[f"pagerank_{gname}"] = tabla.reaccion.map(dpr)
        tabla[f"grado_{gname}"] = tabla.reaccion.map(dgr)
        sub = tabla.dropna(subset=[f"pagerank_{gname}"])
        y = sub.esencial.values
        for med in ["pagerank", "grado"]:
            x = sub[f"{med}_{gname}"].values
            if 0 < y.sum() < len(y):
                auc = roc_auc_score(y, x)
                p = mannwhitneyu(x[y], x[~y], alternative="greater").pvalue
            else:
                auc, p = np.nan, np.nan
            resumen.append({"organismo": nombre, "grafo": gname, "medida": med, "reacciones": len(sub),
                            "esenciales": int(y.sum()), "AUC": auc, "p_MannWhitney": p})
    tabla.insert(0, "organismo", nombre)
    print(f"  {nombre}: {int(tabla.esencial.sum())} reacciones esenciales de {len(tabla)}; " +
          "; ".join(f"{r['grafo']} {r['medida']} AUC={r['AUC']:.2f}" for r in resumen))
    return tabla, resumen


def main():
    model = F.cargar_modelo("RU47_model.xml")
    D = F.Desdoblado(model)
    eco, _ = F.ecoli_core()
    De = F.Desdoblado(eco)
    glc = {"EX_glc__D_e": F.GLUCOSA}
    grafos = [("Ecoli_NFG", *F.submatriz_activa(De.nfg(), De.nodos)),
              ("Ecoli_MFG_glucosa", *F.mfg_condicion(eco, De, glc)[:2]),
              ("RU47_NFG", *F.submatriz_activa(D.nfg(), D.nodos)),
              ("RU47_MFG_glucosa", *F.mfg_condicion(model, D, glc)[:2])]

    print("A) percolacion ...")
    filas = []
    for nombre, W, _ in grafos:
        filas += percolar(nombre, W)
    perc = pd.DataFrame(filas)
    perc.to_csv(os.path.join(OUT, "paso5_percolacion.csv"), index=False)
    R = perc.groupby(["grafo", "estrategia", "rep"]).S_fuerte.mean().groupby(["grafo", "estrategia"]).mean()
    print("  indice de robustez R (componente fuerte):\n" + R.round(3).to_string())

    plt = F.estilo()
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    for ax, (nombre, _, _) in zip(axes, grafos):
        d = perc[perc.grafo == nombre]
        az = d[d.estrategia == "azar"].groupby("f").S_fuerte
        ax.fill_between(FRACS, az.quantile(0.05), az.quantile(0.95), color=F.GRIS, alpha=0.35, lw=0)
        ax.plot(FRACS, az.mean(), color=F.GRIS, lw=2, label="al azar")
        for estr, color in [("PageRank", F.AZUL), ("grado", F.NARANJA)]:
            e = d[d.estrategia == estr]
            ax.plot(e.f, e.S_fuerte, color=color, lw=2, label=f"por {estr}")
        ax.set_title(nombre.replace("_", " "))
        ax.set_xlabel("fraccion de reacciones eliminadas")
    axes[0].set_ylabel("fraccion en la componente fuerte gigante")
    axes[-1].legend(fontsize=8)
    F.guardar_fig(fig, os.path.join(OUT, "paso5_percolacion.png"))

    print("B) esencialidad por FBA (eliminacion de cada reaccion) ...")
    t1, r1 = esencialidad("RU47", model, D)
    t2, r2 = esencialidad("E. coli core", eco, De)
    pd.concat([t1, t2]).to_csv(os.path.join(OUT, "paso5_esencialidad.csv"), index=False)
    res = pd.DataFrame(r1 + r2)
    res.to_csv(os.path.join(OUT, "paso5_esencialidad_resumen.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, g in zip(axes, ["MFG_glucosa", "NFG"]):
        sub = t1.dropna(subset=[f"pagerank_{g}"])
        datos = [sub[~sub.esencial][f"pagerank_{g}"], sub[sub.esencial][f"pagerank_{g}"]]
        for k, (dat, color) in enumerate(zip(datos, [F.GRIS, F.AZUL])):
            x = k + rng.uniform(-0.18, 0.18, len(dat))
            ax.semilogy(x, dat, ".", color=color, ms=3, alpha=0.6)
            ax.plot([k - 0.25, k + 0.25], [np.median(dat)] * 2, color=F.TINTA, lw=2)
        auc = res[(res.organismo == "RU47") & (res.grafo == g) & (res.medida == "pagerank")].AUC.iloc[0]
        ax.set_xticks([0, 1]); ax.set_xticklabels([f"no esencial (n={len(datos[0])})", f"esencial (n={len(datos[1])})"])
        ax.set_ylabel("PageRank de la reaccion")
        ax.set_title(f"RU47 {g.replace('_', ' ')}: AUC = {auc:.2f}")
    fig.text(0.01, -0.02, "Raya negra: mediana. AUC = probabilidad de que una reaccion esencial tenga mas PageRank "
             "que una no esencial (0,5 = azar).", fontsize=7, color=F.TINTA2)
    F.guardar_fig(fig, os.path.join(OUT, "paso5_esencialidad.png"))
    print("Listo: resultados/paso5_*")


if __name__ == "__main__":
    main()
