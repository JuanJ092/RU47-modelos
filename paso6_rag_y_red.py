"""
PASO 6 - Grafo de adyacencia de reacciones (RAG, Ec. 3 del articulo) frente al NFG,
y figura de la red (MFG en glucosa coloreado por comunidades).

  python paso6_rag_y_red.py            (E. coli completo; RU47 usa resultados/ms_RU47_RAG.pkl si existe)
  python paso6_rag_y_red.py --rag      (calcula tambien las comunidades del RAG de RU47: ~1 hora)

Genera en 'resultados':
  paso6_pagerank_RAG_vs_NFG.csv     percentil de PageRank de cada reaccion en el RAG y en el NFG
  paso6_pagerank_RAG_vs_NFG.png     comparacion de percentiles (como la Fig. 2d del articulo)
  paso6_comunidades_RAG_vs_NFG.csv  comunidades de cada grafo: tamano, rutas y metabolitos dominantes
  paso6_resumen.csv                 numero de comunidades, NMI con rutas y peso de los metabolitos de reserva
  paso6_red_MFG_glucosa.png         dibujo del MFG en glucosa de E. coli y de RU47 por comunidades
  <org>_MFG_glucosa_comunidades.gexf  MFG con atributos 'comunidad', 'ruta' y 'pagerank' para Gephi
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from sklearn.metrics import normalized_mutual_info_score as nmi
import cobra
import flujo as F

OUT = "resultados"
os.makedirs(OUT, exist_ok=True)
RESERVA = {"h2o_c", "h_c", "h_p", "h2o_p", "atp_c", "adp_c", "pi_c", "nad_c", "nadh_c", "nadp_c", "nadph_c",
           "ppi_c", "co2_c", "coa_c", "h_e", "h2o_e"}


def rag(model):
    """Ec. 3: A = S_bool^T S_bool (no dirigido; autolazos = metabolitos de cada reaccion)."""
    S = sparse.csc_matrix(cobra.util.array.create_stoichiometric_matrix(model, array_type="lil"))
    Sb = (S != 0).astype(float)
    # los nodos llevan sufijo "_f" para usar la misma convencion que el NFG (flujo.rid quita el sufijo)
    return (Sb.T @ Sb).tocsr(), [r.id + "_f" for r in model.reactions]


def percentiles(x):
    orden = np.argsort(np.argsort(x))
    return 100.0 * orden / (len(x) - 1)


def comparar_pagerank(nombre, model):
    A, nod_a = rag(model)
    Wa, nod_a = F.submatriz_activa(A, nod_a)
    D = F.Desdoblado(model)
    Wd, nod_d = F.submatriz_activa(D.nfg(), D.nodos)
    pa = dict(zip([F.rid(v) for v in nod_a], percentiles(F.pagerank(Wa))))
    pd_ = dict(zip(nod_d, percentiles(F.pagerank(Wd))))
    filas = []
    for v in nod_d:
        r = F.rid(v)
        if r in pa:
            filas.append({"organismo": nombre, "nodo_NFG": v, "reaccion": r, "sentido": v.rsplit("_", 1)[1],
                          "percentil_RAG": pa[r], "percentil_NFG": pd_[v], "cambio": pd_[v] - pa[r]})
    df = pd.DataFrame(filas)
    rho = df[["percentil_RAG", "percentil_NFG"]].corr(method="spearman").iloc[0, 1]
    print(f"  {nombre}: RAG {Wa.shape[0]} nodos y {Wa.nnz // 2} aristas; NFG {Wd.shape[0]} nodos; "
          f"Spearman de percentiles = {rho:.2f}")
    return df, rho, (Wa, nod_a), (Wd, nod_d)


def metabolitos_comunidad(model, reacciones):
    cuenta = {}
    for r in reacciones:
        for m in model.reactions.get_by_id(r).metabolites:
            cuenta[m.id] = cuenta.get(m.id, 0) + 1
    top = sorted(cuenta.items(), key=lambda kv: -kv[1])[:4]
    return top


def resumir_particion(nombre, grafo, model, rutas, nodos, com):
    filas = []
    cats = np.array([rutas.get(F.rid(v), "Sin asignar") for v in nodos])
    for c in np.unique(com):
        idx = np.where(com == c)[0]
        reacs = sorted({F.rid(nodos[i]) for i in idx})
        top = metabolitos_comunidad(model, reacs)
        dom = pd.Series(cats[idx]).value_counts().head(3)
        filas.append({"organismo": nombre, "grafo": grafo, "comunidad": f"C{c + 1}", "n_nodos": len(idx),
                      "rutas_dominantes": "; ".join(f"{k} ({v})" for k, v in dom.items()),
                      "metabolitos_mas_frecuentes": "; ".join(f"{m} ({n}/{len(reacs)})" for m, n in top),
                      "fraccion_con_metabolito_mas_frecuente": top[0][1] / len(reacs),
                      # "definida por un metabolito de reserva": el mas frecuente es de reserva y esta en >= 50 %
                      "definida_por_metabolito_de_reserva": top[0][0] in RESERVA and top[0][1] >= 0.5 * len(reacs)})
    return filas, nmi(cats, com)


def particion_gruesa(res, n):
    ncom = np.asarray(res["number_of_communities"])
    sel0 = [int(i) for i in res["selected_partitions"] if 1 < ncom[int(i)] < n]
    sel = [i for i in sel0 if np.bincount(np.asarray(res["community_id"][i])).min() >= 3] or sel0
    i = min(sel, key=lambda j: ncom[j])
    return i, np.asarray(res["community_id"][i])


def markov_no_dirigido(W, **kw):
    import warnings
    warnings.filterwarnings("ignore")
    import pygenstability as pgs
    return pgs.run(sparse.csr_matrix(W), constructor="continuous_normalized", exp_comp_mode="expm",
                   tqdm_disable=True, result_file=os.devnull, **kw)


def figura_percentiles(tablas):
    plt = F.estilo()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, (nombre, df, rho) in zip(axes, tablas):
        rev = df.groupby("reaccion").filter(lambda g: len(g) == 2)
        for _, g in rev.groupby("reaccion"):
            ax.plot(g.percentil_RAG, g.percentil_NFG, color=F.REJILLA, lw=0.6, zorder=1)
        for s, color, et in [("f", F.AZUL, "directa"), ("b", F.NARANJA, "inversa")]:
            d = df[df.sentido == s]
            ax.scatter(d.percentil_RAG, d.percentil_NFG, s=8 if len(df) < 500 else 2, color=color,
                       alpha=0.8 if len(df) < 500 else 0.35, label=f"reaccion {et}", zorder=2, lw=0)
        ax.plot([0, 100], [0, 100], color=F.GRIS, lw=1, zorder=0)
        if len(df) < 500:
            extremos = pd.concat([df.nlargest(2, "cambio"), df.nsmallest(2, "cambio")])
            for _, r in extremos.iterrows():
                arriba = r.cambio > 0
                ax.annotate(r.nodo_NFG.rsplit("_", 1)[0], (r.percentil_RAG, r.percentil_NFG), fontsize=7,
                            color=F.TINTA2, xytext=(5, -9) if arriba else (-5, 6), ha="left" if arriba else "right",
                            textcoords="offset points")
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.set_xlabel("percentil de PageRank en el RAG (sin direccion)")
        ax.set_ylabel("percentil de PageRank en el NFG (con direccion)")
        ax.set_title(f"{nombre} (rho de Spearman = {rho:.2f})")
    axes[0].legend(loc="lower right", fontsize=7)
    fig.text(0.01, -0.03, "Encima de la diagonal: la reaccion gana importancia al incluir la direccion; debajo: la pierde. "
             "Las lineas unen los dos sentidos de una reaccion reversible.", fontsize=7, color=F.TINTA2)
    fig.tight_layout(w_pad=3)
    F.guardar_fig(fig, os.path.join(OUT, "paso6_pagerank_RAG_vs_NFG.png"))


def figura_red(paneles):
    plt = F.estilo()
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), gridspec_kw={"width_ratios": [1, 1.5]})
    for ax, (titulo, W, nodos, com, nombres) in zip(axes, paneles):
        G = F.a_networkx(W, nodos)
        pr = dict(zip(nodos, F.pagerank(W)))
        U = nx.Graph()
        for u, v, d in G.edges(data=True):
            w = U[u][v]["weight"] + d["weight"] if U.has_edge(u, v) else d["weight"]
            U.add_edge(u, v, weight=w)
        for u, v, d in U.edges(data=True):
            d["w"] = np.log1p(d["weight"])
        # disposicion por comunidades: cada comunidad en su zona; dentro de ella, Kamada-Kawai sin pesos
        # (con pesos, los nodos de mayor flujo se amontonan en un punto)
        comunidades = sorted(set(com), key=lambda c: -np.sum(com == c))
        tam = {c: np.sum(com == c) for c in comunidades}
        pos = {}
        for k, c in enumerate(comunidades):
            ang = np.pi / 2 + 2 * np.pi * k / len(comunidades)
            centro = np.array([np.cos(ang), np.sin(ang)]) * 2.4
            miembros = [v for v, cc in zip(nodos, com) if cc == c]
            sub = U.subgraph(miembros)
            if len(sub) > 2 and nx.is_connected(sub):
                p = nx.kamada_kawai_layout(sub, weight=None)
            else:
                p = nx.spring_layout(sub, weight=None, seed=F.SEMILLA, k=1.5 / np.sqrt(max(len(sub), 1)))
            xy = np.array(list(p.values())) if p else np.zeros((1, 2))
            xy = xy - xy.mean(0)
            escala = np.abs(xy).max() or 1.0
            radio = 0.45 + 1.1 * np.sqrt(tam[c] / len(com))
            for v, q in zip(p.keys(), xy):
                pos[v] = centro + radio * q / escala
        comd = dict(zip(nodos, com))
        pesos = np.array([np.log1p(d["weight"]) for _, _, d in G.edges(data=True)])
        anchos = 0.15 + 2.2 * pesos / pesos.max()
        nx.draw_networkx_edges(G, pos, ax=ax, width=anchos, edge_color=F.GRIS, alpha=0.25, arrows=False)
        orden = sorted(G.nodes, key=lambda v: pr[v])
        nx.draw_networkx_nodes(G, pos, nodelist=orden, ax=ax,
                               node_size=[12 + 1800 * pr[v] for v in orden],
                               node_color=[F.SERIES[comd[v] % len(F.SERIES)] for v in orden],
                               edgecolors="white", linewidths=0.6)
        # rotulos de las cuatro reacciones mas centrales en columna junto a su comunidad (suelen solaparse)
        top = sorted(G.nodes, key=lambda v: -pr[v])[:4]
        xs = np.array([p[0] for p in pos.values()]); ys = np.array([p[1] for p in pos.values()])
        x0 = np.mean([pos[v][0] for v in top]); y0 = np.mean([pos[v][1] for v in top])
        lado = 1 if x0 < xs.mean() else -1
        for j, v in enumerate(top):
            destino = (x0 - lado * 1.6, y0 + 0.9 - 0.32 * j)
            ax.annotate(v.rsplit("_", 1)[0], pos[v], xytext=destino, textcoords="data", fontsize=7.5,
                        color=F.TINTA, ha="right" if lado == 1 else "left", va="center",
                        arrowprops=dict(arrowstyle="-", color=F.TINTA2, lw=0.5, shrinkA=2, shrinkB=2))
        ax.set_xlim(min(xs.min(), x0 - 3.2) - 0.2, max(xs.max(), x0 + 3.2) + 0.2)
        ax.set_title(titulo)
        ax.axis("off")
        h = [Line2D([], [], marker="o", ls="", color=F.SERIES[c % len(F.SERIES)], label=f"C{c + 1}: {nombres.get(c, '')}")
             for c in sorted(set(com))]
        ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=7.5)
    fig.text(0.01, 0.0, "Tamano del nodo: PageRank. Grosor de la arista: flujo de masa (escala logaritmica). "
             "Se rotulan las cuatro reacciones mas centrales. Cada comunidad ocupa su propia zona.", fontsize=7, color=F.TINTA2)
    F.guardar_fig(fig, os.path.join(OUT, "paso6_red_MFG_glucosa.png"))


def nombre_comunidad(model, rutas, nodos, com, c):
    """Nombre corto: las rutas mas frecuentes (sin transporte ni intercambio) y unos marcadores clave."""
    idx = np.where(com == c)[0]
    reacs = sorted({F.rid(nodos[i]) for i in idx})
    if F.objetivo(model) in reacs:
        return "energia y biomasa"
    if "EDD" in reacs:
        return "entrada de glucosa (Entner-Doudoroff)"
    if "CS" in reacs:
        return "ciclo TCA"
    if "PDH" in reacs and "ACKr" in reacs:
        return "piruvato deshidrogenasa y acetato"
    cats = pd.Series([rutas.get(r, "Sin asignar") for r in reacs])
    cats = cats[~cats.isin(["Transporte", "Intercambio", "Sin asignar"])]
    if not len(cats):
        return "transporte e intercambio"
    cortos = {"Glucolisis/Gluconeogenesis": "glucolisis", "Cofactores y vitaminas": "cofactores"}
    cats = cats.replace(cortos)
    vc = cats.value_counts().sort_index().sort_values(ascending=False, kind="stable")
    nombre = vc.index[0] if len(vc) < 2 or vc.iloc[1] < 0.5 * vc.iloc[0] else f"{vc.index[0]} y {vc.index[1].lower()}"
    return nombre[0].lower() + nombre[1:]


def main():
    rutas_ru = F.leer_rutas(os.path.join(OUT, "RU47_rutas.csv"))
    model = F.cargar_modelo("RU47_model.xml")
    eco, rutas_eco = F.ecoli_core()
    glc = {"EX_glc__D_e": F.GLUCOSA}

    # ---------------- A) PageRank RAG vs NFG
    print("A) PageRank en el RAG frente al NFG ...")
    t_eco, rho_e, A_eco, D_eco = comparar_pagerank("E. coli core", eco)
    t_ru, rho_r, A_ru, D_ru = comparar_pagerank("RU47", model)
    pd.concat([t_eco, t_ru]).to_csv(os.path.join(OUT, "paso6_pagerank_RAG_vs_NFG.csv"), index=False)
    figura_percentiles([("E. coli core", t_eco, rho_e), ("RU47", t_ru, rho_r)])

    # ---------------- B) comunidades RAG vs NFG
    print("B) comunidades del RAG (Markov Stability no dirigida) frente al NFG (dirigida) ...")
    filas, resumen = [], []
    casos = []
    Wa, na = F.componente_gigante(*A_eco)
    res_a = markov_no_dirigido(Wa, min_scale=-1, max_scale=1.5, n_scale=30, n_tries=100, n_workers=1)
    Wd, nd = F.componente_gigante(*D_eco)
    res_d = F.markov_stability(Wd, n_tries=100, n_workers=1)
    casos += [("E. coli core", "RAG", eco, rutas_eco, na, res_a), ("E. coli core", "NFG", eco, rutas_eco, nd, res_d)]
    pkl_rag = os.path.join(OUT, "ms_RU47_RAG.pkl")
    if "--rag" in sys.argv and not os.path.exists(pkl_rag):
        Wa, na = F.componente_gigante(*A_ru)
        res = markov_no_dirigido(Wa, min_scale=-1.5, max_scale=1.5, n_scale=12, n_tries=10, n_workers=1)
        res["nodos"] = na
        pickle.dump(res, open(pkl_rag, "wb"))
    if os.path.exists(pkl_rag):
        res = pickle.load(open(pkl_rag, "rb"))
        casos.append(("RU47", "RAG", model, rutas_ru, res["nodos"], res))
    pkl_nfg = os.path.join(OUT, "ms_RU47_NFG.pkl")
    if os.path.exists(pkl_nfg):
        res = pickle.load(open(pkl_nfg, "rb"))
        casos.append(("RU47", "NFG", model, rutas_ru, res["nodos"], res))
    for org, g, mod, rut, nodos, res in casos:
        i, com = particion_gruesa(res, len(nodos))
        f, n_ = resumir_particion(org, g, mod, rut, nodos, com)
        filas += f
        resumen.append({"organismo": org, "grafo": g, "t": float(res["scales"][i]),
                        "n_comunidades": int(len(np.unique(com))), "VI_t": float(res["NVI"][i]),
                        "NMI_con_rutas": n_,
                        "fraccion_de_nodos_en_comunidades_definidas_por_reserva":
                            sum(x["n_nodos"] for x in f if x["definida_por_metabolito_de_reserva"]) / len(nodos)})
        print(f"  {org} {g}: {len(np.unique(com))} comunidades (t={res['scales'][i]:.3g}), NMI con rutas={n_:.2f}, "
              f"nodos en comunidades definidas por un metabolito de reserva: "
              f"{resumen[-1]['fraccion_de_nodos_en_comunidades_definidas_por_reserva']:.0%}")
    pd.DataFrame(filas).to_csv(os.path.join(OUT, "paso6_comunidades_RAG_vs_NFG.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(resumen).to_csv(os.path.join(OUT, "paso6_resumen.csv"), index=False)

    # ---------------- C) figura de la red y GEXF con comunidades
    print("C) figura de la red ...")
    paneles = []
    for org, mod, rut, D, arch in [("E. coli core", eco, rutas_eco, F.Desdoblado(eco), "Ecoli_MFG_glucosa"),
                                   ("RU47", model, rutas_ru, F.Desdoblado(model), "RU47_MFG_glucosa")]:
        W, nodos, *_ = F.mfg_condicion(mod, D, glc)
        part = pd.read_csv(os.path.join(OUT, f"paso2_{arch}_particiones.csv"))
        cols = [c for c in part.columns if c.startswith("t=")]
        n_c = {c: int(c.split("(")[1].split()[0]) for c in cols}
        validas = [c for c in cols if part[c].value_counts().min() >= 3] or cols
        col = min(validas, key=lambda c: n_c[c])
        mapa = dict(zip(part.nodo, part[col] - 1))
        keep = [i for i, v in enumerate(nodos) if v in mapa]
        W = W[keep][:, keep]
        nodos = [nodos[i] for i in keep]
        com = np.array([mapa[v] for v in nodos])
        nombres = {c: nombre_comunidad(mod, rut, nodos, com, c) for c in np.unique(com)}
        paneles.append((f"{org}: MFG en glucosa, {len(np.unique(com))} comunidades", W, nodos, com, nombres))
        G = F.a_networkx(W, nodos)
        pr = F.pagerank(W)
        for v, p, c in zip(nodos, pr, com):
            G.nodes[v].update({"comunidad": f"C{c + 1}", "ruta": rut.get(F.rid(v), "Sin asignar"), "pagerank": float(p)})
        nx.write_gexf(G, os.path.join(OUT, f"{arch}_comunidades.gexf"))
    figura_red(paneles)
    print("Listo: resultados/paso6_*")


if __name__ == "__main__":
    main()
