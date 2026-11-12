"""
flujo.py - funciones comunes para los analisis de grafos de flujo
(Beguerisse-Diaz et al. 2018; Newman 2010) aplicados a Pseudomonas sp. RU47.

Lo usan todos los scripts paso1_... a paso5_...; debe estar en la misma carpeta.
"""
import logging
import os
import re
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
import cobra

logging.getLogger("cobra").setLevel(logging.ERROR)   # oculta "Ignoring reaction ..." y similares

GLUCOSA = 18.5     # mmol/gDW/h, captacion de la fuente de carbono (valor del articulo)
LAMBDA = 0.85      # PageRank y teletransportacion de Markov Stability (valor del articulo)
SEMILLA = 42

# Medio minimo M9 sin fuente de carbono (ids BiGG). La fuente de carbono se agrega por condicion.
M9_BASE = {
    "EX_o2_e": 1000, "EX_nh4_e": 1000, "EX_pi_e": 1000, "EX_so4_e": 1000, "EX_mg2_e": 1000,
    "EX_ca2_e": 1000, "EX_k_e": 1000, "EX_na1_e": 1000, "EX_cl_e": 1000, "EX_fe2_e": 1000,
    "EX_fe3_e": 1000, "EX_mn2_e": 1000, "EX_zn2_e": 1000, "EX_cu2_e": 1000, "EX_cobalt2_e": 1000,
    "EX_mobd_e": 1000, "EX_ni2_e": 1000, "EX_h2o_e": 1000, "EX_h_e": 1000, "EX_co2_e": 1000,
}

# ------------------------------------------------------------------ modelo y medio
def cargar_modelo(ruta):
    return cobra.io.read_sbml_model(ruta)


def fijar_medio(model, extra=None, quitar=(), limites=None):
    """Cierra toda captacion salvo M9_BASE + 'extra' {EX_id: captacion}.
    'quitar' cierra intercambios del medio base (p. ej. oxigeno). 'limites' fija captaciones maximas."""
    medio = dict(M9_BASE)
    medio.update(extra or {})
    for q in quitar:
        medio.pop(q, None)
    medio.update(limites or {})
    for ex in model.exchanges:
        ex.lower_bound = -medio[ex.id] if ex.id in medio else 0.0
    return model


def resolver(model, metodo="pfba"):
    """Devuelve (crecimiento, flujos). pFBA = FBA + minima suma de flujos (solucion mas reproducible)."""
    sol = model.optimize()
    if sol.status != "optimal" or sol.objective_value < 1e-6:
        return 0.0, None
    if metodo == "fba":
        return sol.objective_value, sol.fluxes
    p = cobra.flux_analysis.pfba(model)
    return sol.objective_value, p.fluxes


# ------------------------------------------------------------------ grafos de flujo
def es_reversible(r):
    # Toda reaccion que pueda ir en sentido inverso (lb < 0) se desdobla, incluidas las que SOLO van
    # en inverso (lb < 0 y ub <= 0); los intercambios (EX_) tambien, como en el articulo.
    # Con E. coli core reproduce el NFG del articulo: 154 nodos y 1604 aristas.
    return r.lower_bound < 0 or r.id.startswith("EX_")


class Desdoblado:
    """Matriz estequiometrica desdoblada (Ec. 4) y matrices de produccion/consumo (Ec. 5)."""

    def __init__(self, model):
        S = sparse.csc_matrix(cobra.util.array.create_stoichiometric_matrix(model, array_type="lil"))
        rev = np.array([es_reversible(r) for r in model.reactions])
        idx_rev = np.where(rev)[0]
        self.S2 = sparse.hstack([S, -S[:, idx_rev]]).tocsc()
        self.col_rxn = np.concatenate([np.arange(len(model.reactions)), idx_rev])
        self.sentido = np.array(["f"] * len(model.reactions) + ["b"] * len(idx_rev))
        self.rxn_ids = [r.id for r in model.reactions]
        self.nodos = [f"{self.rxn_ids[j]}_{s}" for j, s in zip(self.col_rxn, self.sentido)]
        self.Sp = self.S2.maximum(0).tocsr()
        self.Sm = (-self.S2).maximum(0).tocsr()
        self.n_met = S.shape[0]

    def nfg(self):
        """Ec. (8)."""
        wp = np.asarray(self.Sp.sum(axis=1)).ravel()
        wm = np.asarray(self.Sm.sum(axis=1)).ravel()
        ip = np.divide(1.0, wp, out=np.zeros_like(wp), where=wp > 0)
        im = np.divide(1.0, wm, out=np.zeros_like(wm), where=wm > 0)
        return ((sparse.diags(ip) @ self.Sp).T @ (sparse.diags(im) @ self.Sm) / self.n_met).tocsr()

    def v2m(self, flujos):
        v = np.asarray(flujos)[self.col_rxn]
        return np.where(self.sentido == "f", np.maximum(v, 0), np.maximum(-v, 0))

    def mfg(self, flujos):
        """Ecs. (9) y (12). Devuelve (M, error de balance)."""
        v2 = self.v2m(flujos)
        V = sparse.diags(v2)
        jp = self.Sp @ v2
        jm = self.Sm @ v2
        ij = np.divide(1.0, jp, out=np.zeros_like(jp), where=jp > 1e-9)
        M = ((self.Sp @ V).T @ sparse.diags(ij) @ (self.Sm @ V)).tocsr()
        return M, float(np.max(np.abs(jp - jm)))


def submatriz_activa(W, nodos, umbral=0.0):
    """Quita nodos sin aristas y autolazos. Devuelve (W_sub csr, lista de nodos)."""
    W = sparse.csr_matrix(W, copy=True)
    W.setdiag(0)
    W.data[W.data <= umbral] = 0
    W.eliminate_zeros()
    act = np.where((np.asarray(W.sum(0)).ravel() + np.asarray(W.sum(1)).ravel()) > 0)[0]
    return W[act][:, act].tocsr(), [nodos[i] for i in act]


def a_networkx(W, nodos):
    G = nx.from_scipy_sparse_array(W, create_using=nx.DiGraph, edge_attribute="weight")
    return nx.relabel_nodes(G, dict(enumerate(nodos)))


def pagerank(W, alpha=LAMBDA, tol=1e-12, maxiter=1000):
    """PageRank ponderado con teletransportacion (identico a networkx, en matrices dispersas)."""
    n = W.shape[0]
    out = np.asarray(W.sum(1)).ravel()
    dang = out == 0
    P = sparse.diags(np.divide(1.0, out, out=np.zeros_like(out), where=~dang)) @ W
    x = np.full(n, 1.0 / n)
    for _ in range(maxiter):
        xn = alpha * (P.T @ x) + (alpha * x[dang].sum() + 1 - alpha) / n
        if np.abs(xn - x).sum() < n * tol:
            return xn
        x = xn
    return x


# ------------------------------------------------------------------ rutas metabolicas
CATEGORIAS = [
    "Glucolisis/Gluconeogenesis", "Pentosas fosfato", "Ciclo TCA y glioxilato",
    "Piruvato y anapleroticas", "Fosforilacion oxidativa", "Aminoacidos", "Nucleotidos",
    "Cofactores y vitaminas", "Lipidos y PHA", "Envoltura celular", "Carbono alternativo y aromaticos",
    "Iones, hierro y azufre", "Transporte", "Intercambio", "Biomasa", "Sin asignar",
]

_REGLAS = [  # (patron, categoria); se evalua en orden sobre el subsistema en minusculas
    (r"biomass|growth", "Biomasa"),
    (r"exhange|exchange|demand|sink", "Intercambio"),
    (r"transport|porin|solvent_extrusion", "Transporte"),
    (r"oxidative.phosph", "Fosforilacion oxidativa"),
    (r"glycoly|gluconeo", "Glucolisis/Gluconeogenesis"),
    (r"pentose", "Pentosas fosfato"),
    (r"tca|citric|glyoxyl", "Ciclo TCA y glioxilato"),
    (r"pyruvate|anaplerot|methylglyoxal", "Piruvato y anapleroticas"),
    (r"cofactor|folate|vitamin|thiamin|riboflavin|nicotin|biotin|ubiquinone|porphyrin|quinone|molybdo|pantothen|terpenoid|queuosine", "Cofactores y vitaminas"),
    (r"purine|pyrimidine|nucleotide", "Nucleotidos"),
    (r"amino|alanine|aspartate|arginine|proline|cysteine|glutamate|glycine|serine|threonine|histidine|lysine|methionine|phenylalanine|tyrosine|tryptophan|valine|leucine|isoleucine|urea|trna|dipeptide|ectoine", "Aminoacidos"),
    (r"lipid|fatty|glycerophospho|pha", "Lipidos y PHA"),
    (r"envelope|murein|peptidoglycan|lps|lipopolysacch|alginate|antigen|heptose|cellulose", "Envoltura celular"),
    (r"iron|metal|inorganic|sulfur|nitrogen|phosph|xenobiotic", "Iones, hierro y azufre"),
    (r"alternate|aromatic|starch|sucrose|butano|formaldehyde|ketone|propano|plant_growth", "Carbono alternativo y aromaticos"),
]


def categoria(subsistema):
    s = (subsistema or "").lower()
    for patron, cat in _REGLAS:
        if re.search(patron, s):
            return cat
    return "Sin asignar"


def anotar_rutas(model, referencias):
    """Hereda el subsistema de cada reaccion desde modelos de referencia con ids BiGG.
    referencias: lista de (nombre, modelo cobra). Devuelve DataFrame reaccion/subsistema/categoria/fuente."""
    filas = []
    for r in model.reactions:
        sub, fuente = "", ""
        if r.subsystem:
            sub, fuente = r.subsystem, "propio"
        else:
            # mismo id, o la variante periplasmica 'pp' (p. ej. ATPS4r -> ATPS4rpp en iJO1366)
            for nombre, ref in referencias:
                for cand in (r.id, r.id + "pp"):
                    if cand in ref.reactions and ref.reactions.get_by_id(cand).subsystem not in ("", "Unassigned"):
                        sub, fuente = ref.reactions.get_by_id(cand).subsystem, nombre
                        break
                if sub:
                    break
        cat = categoria(sub)
        if cat == "Sin asignar":
            if r.id.startswith("EX_"):
                cat, fuente = "Intercambio", fuente or "regla"
            elif r is model.reactions.get_by_id(objetivo(model)):
                cat, fuente = "Biomasa", fuente or "regla"
            elif len({m.compartment for m in r.metabolites}) > 1:
                cat, fuente = "Transporte", fuente or "regla"
        filas.append({"reaccion": r.id, "nombre": r.name, "subsistema": sub, "categoria": cat, "fuente": fuente})
    return pd.DataFrame(filas)


def objetivo(model):
    return [r.id for r in model.reactions if r.objective_coefficient != 0][0]


def leer_rutas(ruta_csv):
    df = pd.read_csv(ruta_csv)
    return dict(zip(df.reaccion, df.categoria))


def rid(nodo):
    return nodo.rsplit("_", 1)[0]


# ------------------------------------------------------------------ comparacion de particiones
def vi_normalizada(a, b):
    """Variacion de informacion normalizada por log(n) (Ec. 20 del articulo)."""
    a = np.asarray(a); b = np.asarray(b); n = len(a)
    if n < 2:
        return 0.0
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    C = sparse.coo_matrix((np.ones(n), (ia, ib))).toarray() / n
    pa = C.sum(1); pb = C.sum(0)
    def H(p):
        p = p[p > 0]; return -np.sum(p * np.log(p))
    Hab = H(C.ravel())
    return (2 * Hab - H(pa) - H(pb)) / np.log(n)


AZUL, NARANJA, GRIS, TINTA, TINTA2, REJILLA = "#2a78d6", "#eb6834", "#a8a69f", "#0b0b0b", "#52514e", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": TINTA2,
        "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": False, "grid.color": REJILLA,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    return plt


def guardar_fig(fig, ruta):
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


# ------------------------------------------------------------------ Markov Stability dirigida
try:
    from pygenstability.constructors import constructor_directed as _CD
except Exception:          # si PyGenStability no esta instalado, el resto de funciones sigue sirviendo
    _CD = object


class DirigidoCorregido(_CD):
    """Constructor 'directed' de PyGenStability con el manejo correcto de los nodos sin aristas
    de salida (en la version 0.2.5 la division deja memoria sin inicializar en esos nodos)."""

    def prepare(self, **kwargs):
        import scipy.sparse as sp
        a = kwargs.get("alpha", LAMBDA)
        A = self.graph.toarray()
        n = A.shape[0]
        out = A.sum(1)
        dang = out == 0
        dinv = np.divide(1.0, out, out=np.zeros_like(out), where=~dang)
        Mt = a * (dinv[:, None] * A) + ((1 - a) + a * dang)[:, None] * np.full((n, n), 1.0 / n)
        self.partial_quality_matrix = sp.csr_matrix(Mt - np.eye(n))
        pi = np.full(n, 1.0 / n)            # distribucion estacionaria (= PageRank)
        for _ in range(100000):
            pn = pi @ Mt
            if np.abs(pn - pi).sum() < 1e-14:
                break
            pi = pn
        self.partial_null_model = np.array([pn, pn])


def markov_stability(W, min_t=-1.0, max_t=1.5, n_t=30, n_tries=100, alpha=LAMBDA, n_workers=4, **kw):
    """Markov Stability dirigida con teletransportacion (Ecs. 16-22 del articulo).
    t (tiempo de Markov) va de 10**min_t a 10**max_t. Devuelve el diccionario de PyGenStability."""
    import warnings
    warnings.filterwarnings("ignore")
    import scipy.sparse as sp
    import pygenstability as pgs
    G = sp.csr_matrix(W)
    cons = DirigidoCorregido(G, exp_comp_mode="expm")
    return pgs.run(G, constructor=cons, constructor_kwargs={"alpha": alpha}, exp_comp_mode="expm",
                   min_scale=min_t, max_scale=max_t, n_scale=n_t, n_tries=n_tries,
                   n_workers=n_workers, tqdm_disable=True, result_file=os.devnull, **kw)


def componente_gigante(W, nodos):
    """Restringe a la componente debilmente conexa mas grande (Markov Stability la exige conexa)."""
    from scipy.sparse.csgraph import connected_components
    _, lab = connected_components(W, directed=True, connection="weak")
    k = np.bincount(lab).argmax()
    idx = np.where(lab == k)[0]
    return W[idx][:, idx].tocsr(), [nodos[i] for i in idx]


def mfg_condicion(model, D, extra, quitar=(), limites=None, metodo="pfba"):
    """Fija el medio, resuelve (pFBA por defecto) y devuelve (W, nodos, crecimiento, flujos, error).
    W y nodos ya vienen sin nodos inactivos ni autolazos. Si no hay crecimiento devuelve W=None."""
    with model:
        fijar_medio(model, extra=extra, quitar=quitar, limites=limites)
        mu, flujos = resolver(model, metodo)
    if flujos is None:
        return None, [], 0.0, None, np.nan
    M, err = D.mfg(flujos.values)
    W, nodos = submatriz_activa(M, D.nodos, umbral=1e-9)
    return W, nodos, mu, flujos, err


def ecoli_core():
    """Modelo central de E. coli (el del articulo) con sus rutas heredadas de iJO1366."""
    eco = cobra.io.load_model("textbook")
    an = anotar_rutas(eco, [("iJO1366 (E. coli)", cobra.io.load_model("iJO1366"))])
    return eco, dict(zip(an.reaccion, an.categoria))
