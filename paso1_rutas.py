"""
PASO 1 - Rutas metabolicas y PageRank por ruta (analogo a la Fig. 5 del articulo).

  python paso1_rutas.py

Lee RU47_model.xml (y, si estan, iJN1462.json para heredar subsistemas).
Genera en la carpeta 'resultados':
  RU47_rutas.csv                  ruta (subsistema y categoria) de cada reaccion
  paso1_pagerank_rutas.csv        PageRank acumulado por categoria, RU47 y E. coli, NFG y MFG
  paso1_reacciones_top.csv        las 20 reacciones de mayor PageRank en cada grafo de RU47
  paso1_pagerank_rutas.png        figura comparativa
"""
import os
import numpy as np
import pandas as pd
import cobra
import flujo as F

OUT = "resultados"
os.makedirs(OUT, exist_ok=True)


def grafos(model, rutas, nombre):
    D = F.Desdoblado(model)
    filas, tops = [], []
    F.fijar_medio(model, extra={"EX_glc__D_e": F.GLUCOSA})
    mu, flujos = F.resolver(model, "pfba")
    M, err = D.mfg(flujos)
    print(f"  {nombre}: crecimiento en glucosa (M9) = {mu:.3f} 1/h, error de balance = {err:.1e}")
    for etiqueta, W in [("NFG", D.nfg()), ("MFG_glucosa", M)]:
        Ws, nodos = F.submatriz_activa(W, D.nodos, umbral=1e-9 if etiqueta != "NFG" else 0)
        pr = F.pagerank(Ws)
        cat = [rutas.get(F.rid(v), "Sin asignar") for v in nodos]
        df = pd.DataFrame({"nodo": nodos, "categoria": cat, "pagerank": pr})
        g = df.groupby("categoria").agg(pagerank_acumulado=("pagerank", "sum"), n_nodos=("nodo", "size"))
        for c in F.CATEGORIAS:
            filas.append({"organismo": nombre, "grafo": etiqueta, "categoria": c,
                          "pagerank_acumulado": g.pagerank_acumulado.get(c, 0.0),
                          "n_nodos": int(g.n_nodos.get(c, 0))})
        df["grafo"] = etiqueta
        df["nombre"] = [model.reactions.get_by_id(F.rid(v)).name for v in nodos]
        tops.append(df.sort_values("pagerank", ascending=False).head(20))
        print(f"  {nombre} {etiqueta}: {Ws.shape[0]} nodos, {Ws.nnz} aristas")
    return filas, pd.concat(tops)


def main():
    model = F.cargar_modelo("RU47_model.xml")

    # ---- anotacion de rutas de RU47
    ruta_csv = os.path.join(OUT, "RU47_rutas.csv")
    if os.path.exists("iJN1462.json"):
        refs = [("iJN1462 (P. putida)", cobra.io.load_json_model("iJN1462.json")),
                ("iJO1366 (E. coli)", cobra.io.load_model("iJO1366"))]
        an = F.anotar_rutas(model, refs)
        an.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
    else:
        an = pd.read_csv(ruta_csv)
    print("Origen de la anotacion:", an.fuente.replace("", "sin anotar").fillna("sin anotar").value_counts().to_dict())
    print("Reacciones por categoria:", an.categoria.value_counts().to_dict())
    rutas = dict(zip(an.reaccion, an.categoria))

    # ---- E. coli core (trae sus propios subsistemas)
    # ---- E. coli core: sus subsistemas se heredan de iJO1366 (mismos ids BiGG)
    eco = cobra.io.load_model("textbook")
    an_eco = F.anotar_rutas(eco, [("iJO1366 (E. coli)", cobra.io.load_model("iJO1366"))])
    an_eco.to_csv(os.path.join(OUT, "Ecoli_core_rutas.csv"), index=False, encoding="utf-8-sig")
    rutas_eco = dict(zip(an_eco.reaccion, an_eco.categoria))
    print("E. coli core, reacciones por categoria:", an_eco.categoria.value_counts().to_dict())

    f1, top_ru = grafos(model, rutas, "RU47")
    f2, _ = grafos(eco, rutas_eco, "E. coli core")
    tabla = pd.DataFrame(f1 + f2)
    tabla.to_csv(os.path.join(OUT, "paso1_pagerank_rutas.csv"), index=False, encoding="utf-8-sig")
    top_ru.to_csv(os.path.join(OUT, "paso1_reacciones_top.csv"), index=False, encoding="utf-8-sig")

    # ---- figura
    plt = F.estilo()
    cats = [c for c in F.CATEGORIAS if tabla[tabla.categoria == c].pagerank_acumulado.sum() > 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.32 * len(cats) + 1.4), sharey=True)
    y = np.arange(len(cats))
    for ax, g, titulo in zip(axes, ["NFG", "MFG_glucosa"], ["NFG (todo el metabolismo)", "MFG (aerobio en glucosa)"]):
        for k, (org, color) in enumerate([("RU47", F.AZUL), ("E. coli core", F.GRIS)]):
            v = tabla[(tabla.organismo == org) & (tabla.grafo == g)].set_index("categoria").pagerank_acumulado.reindex(cats)
            ax.barh(y + (k - 0.5) * 0.38, v.values, height=0.36, color=color, label=org)
        ax.set_title(titulo)
        ax.set_xlabel("PageRank acumulado de la ruta")
        ax.invert_yaxis()
        ax.xaxis.grid(True); ax.set_axisbelow(True)
    axes[0].set_yticks(y); axes[0].set_yticklabels(cats)
    axes[1].legend(loc="lower right")
    F.guardar_fig(fig, os.path.join(OUT, "paso1_pagerank_rutas.png"))
    print("Listo: resultados/paso1_*.csv y paso1_pagerank_rutas.png")


if __name__ == "__main__":
    main()
