"""
PASO 3 - Mass Flow Graphs en distintas condiciones de crecimiento (analogo a las Figs. 4 y 5).

  python paso3_condiciones.py

Condiciones de RU47 (captacion de la fuente de carbono = 18,5 mmol/gDW/h, como en el articulo):
  glucosa aerobia; glucosa con O2 limitado (25 % del O2 que usa en glucosa aerobia);
  glucosa con N y P limitados (50 % del NH4 y del Pi que usa en glucosa aerobia, como en el articulo);
  succinato, malato y citrato (acidos organicos de exudados de raiz); L-glutamato; benzoato (aromatico).
  (El modelo no crece en anaerobiosis, ni con nitrato: se informa y se usa O2 limitado.)
Condiciones de E. coli core: las cuatro del articulo.

Genera en 'resultados':
  paso3_condiciones.csv          crecimiento, tamano del MFG, comunidades y parecido con glucosa
  paso3_pagerank_rutas.csv       PageRank acumulado por ruta en cada condicion
  paso3_reacciones_top.csv       10 reacciones mas centrales por condicion
  paso3_cambio_rutas.png         cambio del PageRank por ruta respecto a glucosa (RU47 y E. coli)
  RU47_MFG_<condicion>.gexf      cada MFG, para abrir en Gephi
"""
import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr
import flujo as F

OUT = "resultados"
os.makedirs(OUT, exist_ok=True)
C = F.GLUCOSA


def condiciones_ru47(model):
    with model:
        F.fijar_medio(model, extra={"EX_glc__D_e": C})
        fl = F.resolver(model, "pfba")[1]
    o2, nh4, pi = -fl["EX_o2_e"], -fl["EX_nh4_e"], -fl["EX_pi_e"]
    print(f"  RU47 en glucosa aerobia usa O2 = {o2:.2f}, NH4 = {nh4:.2f}, Pi = {pi:.2f} mmol/gDW/h")
    return {
        "glucosa": dict(extra={"EX_glc__D_e": C}),
        "glucosa_O2_limitado": dict(extra={"EX_glc__D_e": C}, limites={"EX_o2_e": 0.25 * o2}),
        "glucosa_N_P_limitados": dict(extra={"EX_glc__D_e": C}, limites={"EX_nh4_e": 0.5 * nh4, "EX_pi_e": 0.5 * pi}),
        "succinato": dict(extra={"EX_succ_e": C}),
        "malato": dict(extra={"EX_mal__L_e": C}),
        "citrato": dict(extra={"EX_cit_e": C}),
        "glutamato": dict(extra={"EX_glu__L_e": C}),
        "benzoato": dict(extra={"EX_bz_e": C}),
        "glucosa_anaerobio": dict(extra={"EX_glc__D_e": C, "EX_no3_e": 1000}, quitar=("EX_o2_e",)),
    }


CONDICIONES_ECOLI = {   # las del articulo
    "glucosa": dict(extra={"EX_glc__D_e": C}),
    "etanol": dict(extra={"EX_etoh_e": C}),
    "glucosa_anaerobio": dict(extra={"EX_glc__D_e": C}, quitar=("EX_o2_e",)),
    "glucosa_N_P_limitados": dict(extra={"EX_glc__D_e": C}, limites={"EX_nh4_e": 4.5, "EX_pi_e": 3.04}),
}


def correr(org, model, rutas, conds, gexf=False):
    D = F.Desdoblado(model)
    filas, prs, tops, ref = [], [], [], None
    for nombre, kw in conds.items():
        W, nodos, mu, fl, err = F.mfg_condicion(model, D, **kw)
        if W is None:
            print(f"  {org} {nombre}: sin crecimiento")
            filas.append({"organismo": org, "condicion": nombre, "crecimiento": 0.0})
            continue
        pr = F.pagerank(W)
        cats = [rutas.get(F.rid(v), "Sin asignar") for v in nodos]
        Wg, ng = F.componente_gigante(W, nodos)
        res = F.markov_stability(Wg, n_tries=50, n_workers=1)
        ncom = np.asarray(res["number_of_communities"])
        sel0 = [int(i) for i in res["selected_partitions"] if 1 < ncom[int(i)] < len(ng)]
        sel = [i for i in sel0 if np.bincount(np.asarray(res["community_id"][i])).min() >= 3] or sel0
        gruesa = int(ncom[min(sel, key=lambda i: ncom[i])]) if sel else np.nan
        d = dict(zip(nodos, pr))
        fila = {"organismo": org, "condicion": nombre, "crecimiento": mu, "nodos": W.shape[0],
                "aristas": W.nnz, "error_balance": err, "comunidades_particion_gruesa": gruesa}
        if ref is None:
            ref = d
        else:
            comunes = sorted(set(d) & set(ref))
            fila["jaccard_reacciones_vs_glucosa"] = len(comunes) / len(set(d) | set(ref))
            fila["spearman_pagerank_vs_glucosa"] = spearmanr([d[k] for k in comunes], [ref[k] for k in comunes])[0]
        filas.append(fila)
        df = pd.DataFrame({"nodo": nodos, "categoria": cats, "pagerank": pr})
        g = df.groupby("categoria").pagerank.sum()
        for c in F.CATEGORIAS:
            prs.append({"organismo": org, "condicion": nombre, "categoria": c, "pagerank_acumulado": g.get(c, 0.0)})
        df["condicion"] = nombre
        df["nombre"] = [model.reactions.get_by_id(F.rid(v)).name for v in nodos]
        tops.append(df.sort_values("pagerank", ascending=False).head(10))
        if gexf:
            G = F.a_networkx(W, nodos)
            for v in G.nodes:
                G.nodes[v]["ruta"] = rutas.get(F.rid(v), "Sin asignar")
                G.nodes[v]["pagerank"] = float(d[v])
            nx.write_gexf(G, os.path.join(OUT, f"{org}_MFG_{nombre}.gexf"))
        print(f"  {org} {nombre}: crecimiento {mu:.3f} 1/h, {W.shape[0]} nodos, {W.nnz} aristas, "
              f"particion gruesa: {gruesa} comunidades")
    return filas, prs, tops


def figura(tabla):
    plt = F.estilo()
    from matplotlib.colors import TwoSlopeNorm
    orgs = ["RU47", "E. coli core"]
    piv = {o: tabla[tabla.organismo == o].pivot_table(index="categoria", columns="condicion",
                                                      values="pagerank_acumulado", aggfunc="sum") for o in orgs}
    cats = [c for c in F.CATEGORIAS if any(c in piv[o].index and piv[o].loc[c].max() > 0.005 for o in orgs)]
    mats = []
    for o in orgs:
        orden = [c for c in tabla[tabla.organismo == o].condicion.unique() if c in piv[o].columns and c != "glucosa"]
        t = piv[o].reindex(cats).fillna(0)
        mats.append(t[orden].sub(t["glucosa"], axis=0))
    vmax = max(np.abs(m.values).max() for m in mats)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True,
                             gridspec_kw={"width_ratios": [mats[0].shape[1], mats[1].shape[1]], "wspace": 0.05})
    for ax, dif, org in zip(axes, mats, orgs):
        im = ax.imshow(dif.values, cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
        for (r, c), v in np.ndenumerate(dif.values):
            if abs(v) >= 0.02:
                ax.text(c, r, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 * vmax else F.TINTA)
        ax.set_xticks(range(dif.shape[1]))
        ax.set_xticklabels([c.replace("_", " ") for c in dif.columns], rotation=35, ha="right")
        ax.set_title(org)
        ax.spines[:].set_visible(False)
    axes[0].set_yticks(range(len(cats))); axes[0].set_yticklabels(cats)
    fig.colorbar(im, ax=axes, shrink=0.6, label="cambio del PageRank de la ruta respecto a glucosa aerobia")
    fig.suptitle("Cambio en la importancia de cada ruta segun la condicion de crecimiento", x=0.02, ha="left",
                 fontweight="bold", fontsize=11)
    F.guardar_fig(fig, os.path.join(OUT, "paso3_cambio_rutas.png"))


def main():
    model = F.cargar_modelo("RU47_model.xml")
    rutas = F.leer_rutas(os.path.join(OUT, "RU47_rutas.csv"))
    f1, p1, t1 = correr("RU47", model, rutas, condiciones_ru47(model), gexf=True)
    eco, rutas_eco = F.ecoli_core()
    f2, p2, t2 = correr("E. coli core", eco, rutas_eco, CONDICIONES_ECOLI)
    pd.DataFrame(f1 + f2).to_csv(os.path.join(OUT, "paso3_condiciones.csv"), index=False)
    tabla = pd.DataFrame(p1 + p2)
    tabla.to_csv(os.path.join(OUT, "paso3_pagerank_rutas.csv"), index=False, encoding="utf-8-sig")
    pd.concat(t1 + t2).to_csv(os.path.join(OUT, "paso3_reacciones_top.csv"), index=False, encoding="utf-8-sig")
    figura(tabla)
    print("Listo: resultados/paso3_*")


if __name__ == "__main__":
    main()
