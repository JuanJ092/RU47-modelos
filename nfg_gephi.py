"""Archivos GEXF del NFG para Gephi:
  RU47_NFG_comunidades_agregado.gexf : grafo de las 15 comunidades del NFG de RU47 (nodo = comunidad)
  Ecoli_NFG_comunidades.gexf         : NFG completo de E. coli core con sus 5 comunidades
"""
import os, numpy as np, pandas as pd, networkx as nx
from scipy import sparse
import flujo as F
OUT = "resultados"

def excl(c):
    return c in ("Transporte", "Intercambio", "Sin asignar")

# ---------------- RU47: grafo de comunidades
model = F.cargar_modelo("RU47_model.xml")
D = F.Desdoblado(model)
W, nodos = F.submatriz_activa(D.nfg(), D.nodos)
part = pd.read_csv(os.path.join(OUT, "paso2_RU47_NFG_particiones.csv"))
col = [c for c in part.columns if c.startswith("t=")][0]
com = dict(zip(part.nodo, part[col]))
idx = [i for i, v in enumerate(nodos) if v in com]
W = W.tocsr()[idx][:, idx]; nodos = [nodos[i] for i in idx]
c = np.array([com[v] for v in nodos]); ks = sorted(set(c))
H = sparse.csr_matrix((np.ones(len(c)), (np.arange(len(c)), [ks.index(x) for x in c])), shape=(len(c), len(ks)))
A = (H.T @ W @ H).toarray()
pr = F.pagerank(W)
rutas = F.leer_rutas(os.path.join(OUT, "RU47_rutas.csv"))
# nombres revisados a mano a partir de las rutas y de las reacciones mas centrales de cada comunidad
NOMBRES = {1: "Transporte, aminoácidos y nucleótidos", 2: "Glucólisis, pentosas fosfato y envoltura",
           3: "β-oxidación (I)", 4: "Fosfolípidos y PHA", 5: "β-oxidación (II)", 6: "β-oxidación (III)",
           7: "Acil-CoA y butanoato", 8: "Transporte de manganeso", 9: "β-oxidación (IV)",
           10: "Propanodiol", 11: "Cobalamina", 12: "Transporte de crotonobetaína",
           13: "Tiamina y purinas", 14: "Lactaldehído", 15: "Transporte de melibiosa"}
G = nx.DiGraph()
for a, k in enumerate(ks):
    m = c == k
    cats = pd.Series([rutas.get(F.rid(v), "Sin asignar") for v, mm in zip(nodos, m) if mm])
    vc = cats.value_counts()
    bio = vc[[not excl(x) for x in vc.index]]
    principal = bio.index[0] if len(bio) else vc.index[0]
    G.add_node(f"C{k}", comunidad=f"C{k}", label=f"C{k} {NOMBRES.get(k, principal)} ({m.sum()})", n_reacciones=int(m.sum()),
               pagerank=float(pr[m].sum()), ruta_principal=principal,
               rutas="; ".join(f"{x} ({n})" for x, n in vc.head(4).items()),
               fraccion_interna=float(A[a, a] / A[a].sum()) if A[a].sum() else 0.0)
    print(f"C{k}: {m.sum()} reacciones, PR={pr[m].sum():.3f}, interna={G.nodes[f'C{k}']['fraccion_interna']:.2f}, "
          + "; ".join(f"{x} ({n})" for x, n in bio.head(3).items()))
for a, ka in enumerate(ks):
    tot = A[a].sum()
    for b, kb in enumerate(ks):
        if a != b and A[a, b] > 0:
            # weight = fraccion del peso que sale de la comunidad de origen (Gephi usa "weight" para el grosor)
            G.add_edge(f"C{ka}", f"C{kb}", weight=float(A[a, b] / tot), peso_nfg=float(A[a, b]))
nx.write_gexf(G, os.path.join(OUT, "RU47_NFG_comunidades_agregado.gexf"))
print("RU47 agregado:", G.number_of_nodes(), "nodos,", G.number_of_edges(), "aristas")
w = np.array([d["weight"] for *_, d in G.edges(data=True)])
print("aristas con >=1 % de la salida:", (w >= 0.01).sum(), "| >=5 %:", (w >= 0.05).sum())

# ---------------- E. coli: NFG completo con comunidades
eco, rutas_eco = F.ecoli_core()
De = F.Desdoblado(eco)
We, ne = F.submatriz_activa(De.nfg(), De.nodos)
pe = pd.read_csv(os.path.join(OUT, "paso2_Ecoli_NFG_particiones.csv"))
cole = [x for x in pe.columns if "(5 com.)" in x][0]
come = dict(zip(pe.nodo, pe[cole]))
pre = F.pagerank(We)
nombres = {1: "C1 fermentacion e intercambio", 2: "C2 biomasa y glucolisis", 3: "C3 ciclo TCA y anapleroticas",
           4: "C4 cadena respiratoria y succinato", 5: "C5 pentosas fosfato y glucolisis"}
Ge = nx.DiGraph()
for i, v in enumerate(ne):
    k = int(come.get(v, 0))
    Ge.add_node(v, label=v, comunidad=f"C{k}", nombre_comunidad=nombres.get(k, ""),
                ruta=rutas_eco.get(F.rid(v), "Sin asignar"), pagerank=float(pre[i]))
We = sparse.coo_matrix(We)
for i, j, x in zip(We.row, We.col, We.data):
    if x > 0:
        Ge.add_edge(ne[i], ne[j], weight=float(x))
nx.write_gexf(Ge, os.path.join(OUT, "Ecoli_NFG_comunidades.gexf"))
print("E. coli NFG:", Ge.number_of_nodes(), "nodos,", Ge.number_of_edges(), "aristas; sin comunidad:",
      sum(1 for v in ne if v not in come))
