"""
PASO 2 - Comunidades multiescala con Markov Stability dirigida (analogo a las Figs. 3, 4 y 6).

  python paso2_comunidades.py            (MFG de RU47 y grafos de E. coli: minutos)
  python paso2_comunidades.py --nfg      (agrega el NFG de RU47: puede tardar horas en un portatil)

Requiere: pip install pygenstability scikit-learn
Genera en 'resultados':
  paso2_<grafo>_escalas.csv        tiempo de Markov t, numero de comunidades, VI(t)
  paso2_<grafo>_particiones.csv    comunidad de cada reaccion en cada particion robusta seleccionada
  paso2_<grafo>_escalas.png        VI(t,t'), numero de comunidades y VI(t) vs t (como la Fig. 6)
  paso2_<grafo>_composicion.png    composicion por ruta de cada comunidad (particion mas gruesa robusta)
  paso2_<grafo>_comunidades.csv    tamano, rutas dominantes y reacciones mas centrales de esa particion
  paso2_resumen.csv                particiones robustas de cada grafo y su parecido con las rutas
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score as nmi
import flujo as F

OUT = "resultados"
os.makedirs(OUT, exist_ok=True)


def analizar(nombre, W, nodos, rutas, res=None, nodos_completos=None, **kw):
    if res is None:
        W, nodos = F.componente_gigante(W, nodos)
        print(f"  {nombre}: {W.shape[0]} nodos, Markov Stability ...")
        res = F.markov_stability(W, **kw)
    elif W is not None:
        W, _ = F.componente_gigante(W, nodos_completos) if nodos_completos else (W, None)
        if W.shape[0] != len(nodos):
            W = None
    t = np.asarray(res["scales"])
    ncom = np.asarray(res["number_of_communities"])
    nvi = np.asarray(res["NVI"])
    pd.DataFrame({"t": t, "n_comunidades": ncom, "VI_t": nvi}).to_csv(
        os.path.join(OUT, f"paso2_{nombre}_escalas.csv"), index=False)

    # particiones robustas: las que selecciona PyGenStability (minimos de VI(t,t') en bloques)
    sel = [int(i) for i in res["selected_partitions"] if 1 < ncom[int(i)] < len(nodos)]
    cats = np.array([rutas.get(F.rid(v), "Sin asignar") for v in nodos])
    part = pd.DataFrame({"nodo": nodos, "reaccion": [F.rid(v) for v in nodos], "categoria": cats})
    filas = []
    for i in sel:
        com = np.asarray(res["community_id"][i])
        col = f"t={t[i]:.3g} ({ncom[i]} com.)"
        part[col] = com + 1
        filas.append({"grafo": nombre, "t": t[i], "n_comunidades": int(ncom[i]), "VI_t": nvi[i],
                      "NMI_con_rutas": nmi(cats, com), "VI_con_rutas": F.vi_normalizada(cats, com)})
    part.to_csv(os.path.join(OUT, f"paso2_{nombre}_particiones.csv"), index=False, encoding="utf-8-sig")

    plt = F.estilo()
    # ---- figura de escalas (como el panel superior de la Fig. 6)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    if res.get("ttprime") is not None:
        lt = np.log10(t)
        ax.imshow(res["ttprime"], origin="lower", aspect="auto", cmap="Blues_r",
                  extent=[lt[0], lt[-1], lt[0], lt[-1]], alpha=0.55)
    ax2 = ax.twinx()
    ax.plot(np.log10(t), np.log10(ncom), color=F.TINTA, lw=2, label="numero de comunidades")
    ax2.plot(np.log10(t), nvi, color=F.NARANJA, lw=1.5, label="VI(t) (robustez)")
    for i in sel:
        ax.axvline(np.log10(t[i]), color=F.TINTA2, ls=":", lw=1)
        ax.text(np.log10(t[i]), np.log10(ncom[i]), f" {ncom[i]}", color=F.TINTA, fontsize=8, va="bottom")
    ax.set_xlabel("log10 del tiempo de Markov t")
    ax.set_ylabel("log10 (numero de comunidades)")
    ax2.set_ylabel("VI(t)", color=F.NARANJA)
    ax2.spines["right"].set_visible(True)
    ax.set_title(f"{nombre}: comunidades a distintas escalas")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    F.guardar_fig(fig, os.path.join(OUT, f"paso2_{nombre}_escalas.png"))

    # ---- composicion por ruta de la particion robusta mas gruesa
    if sel:
        # particion robusta mas gruesa sin comunidades de menos de 3 reacciones
        def ok(j):
            return np.bincount(np.asarray(res["community_id"][j])).min() >= 3
        validas = [j for j in sel if ok(j)] or sel
        i = min(validas, key=lambda j: ncom[j])
        com = np.asarray(res["community_id"][i]) + 1
        # resumen de cada comunidad: tamano, rutas dominantes y reacciones mas centrales
        if W is not None:
            pr = F.pagerank(W)
            resumen = []
            for c in np.unique(com):
                idx = np.where(com == c)[0]
                orden = idx[np.argsort(-pr[idx])][:6]
                dom = pd.Series(cats[idx]).value_counts().head(3)
                resumen.append({"comunidad": f"C{c}", "n_reacciones": len(idx),
                                "pagerank_acumulado": pr[idx].sum(),
                                "rutas_dominantes": "; ".join(f"{k} ({v})" for k, v in dom.items()),
                                "reacciones_mas_centrales": "; ".join(nodos[k] for k in orden)})
            pd.DataFrame(resumen).to_csv(os.path.join(OUT, f"paso2_{nombre}_comunidades.csv"),
                                         index=False, encoding="utf-8-sig")
        tab = pd.crosstab(pd.Series(cats, name="ruta"), pd.Series(com, name="comunidad"))
        tab = tab.loc[[c for c in F.CATEGORIAS if c in tab.index]]
        frac = tab / tab.sum(0)
        fig, ax = plt.subplots(figsize=(0.55 * tab.shape[1] + 3.2, 0.32 * tab.shape[0] + 1.3))
        ax.imshow(frac.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for (r, c), v in np.ndenumerate(tab.values):
            if v:
                ax.text(c, r, v, ha="center", va="center", fontsize=7,
                        color="white" if frac.values[r, c] > 0.55 else F.TINTA)
        ax.set_xticks(range(tab.shape[1])); ax.set_xticklabels([f"C{c}" for c in tab.columns])
        ax.set_yticks(range(tab.shape[0])); ax.set_yticklabels(tab.index)
        ax.set_title(f"{nombre}: {ncom[i]} comunidades (t = {t[i]:.3g}, VI = {nvi[i]:.2f})")
        ax.set_xlabel("comunidad (numero = reacciones; color = fraccion de la comunidad)")
        ax.spines[:].set_visible(False)
        F.guardar_fig(fig, os.path.join(OUT, f"paso2_{nombre}_composicion.png"))
    for f in filas:
        print(f"    t={f['t']:.3g}: {f['n_comunidades']} comunidades, VI(t)={f['VI_t']:.3f}, NMI con rutas={f['NMI_con_rutas']:.2f}")
    return filas


def main():
    con_nfg = "--nfg" in sys.argv
    filas = []
    model = F.cargar_modelo("RU47_model.xml")
    rutas = F.leer_rutas(os.path.join(OUT, "RU47_rutas.csv"))
    D = F.Desdoblado(model)
    eco, rutas_eco = F.ecoli_core()
    De = F.Desdoblado(eco)
    glc = {"EX_glc__D_e": F.GLUCOSA}

    W, nodos = F.submatriz_activa(De.nfg(), De.nodos)
    filas += analizar("Ecoli_NFG", W, nodos, rutas_eco, n_workers=1)
    W, nodos, *_ = F.mfg_condicion(eco, De, glc)
    filas += analizar("Ecoli_MFG_glucosa", W, nodos, rutas_eco, n_workers=1)
    W, nodos, *_ = F.mfg_condicion(model, D, glc)
    filas += analizar("RU47_MFG_glucosa", W, nodos, rutas, n_workers=1)

    pkl = os.path.join(OUT, "ms_RU47_NFG.pkl")
    if os.path.exists(pkl):
        res = pickle.load(open(pkl, "rb"))
        Wn, nn = F.submatriz_activa(D.nfg(), D.nodos)
        filas += analizar("RU47_NFG", Wn, res["nodos"], rutas, res=res, nodos_completos=nn)
    elif con_nfg:
        W, nodos = F.submatriz_activa(D.nfg(), D.nodos)
        filas += analizar("RU47_NFG", W, nodos, rutas, min_t=-0.5, max_t=2.0, n_t=12, n_tries=10, n_workers=1)
    else:
        print("  (NFG de RU47 omitido: corre con --nfg o deja resultados/ms_RU47_NFG.pkl)")
    pd.DataFrame(filas).to_csv(os.path.join(OUT, "paso2_resumen.csv"), index=False)
    print("Listo: resultados/paso2_*")


if __name__ == "__main__":
    main()
