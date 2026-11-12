# Grafos de flujo de la red metabólica de *Pseudomonas* sp. RU47

Datos y código del trabajo de grado *"Grafos de flujo de la red metabólica a escala genómica de Pseudomonas sp. RU47: estructura, comunidades y comparación con Escherichia coli"* (\[autor\], \[programa\], Universidad de los Llanos, \[año\]).

El trabajo construye el modelo metabólico a escala genómica de *Pseudomonas* sp. RU47, lo representa con los grafos de flujo de Beguerisse-Díaz et al. (2018), el *Normalised Flow Graph* (NFG) y los *Mass Flow Graphs* (MFG), y analiza sus propiedades topológicas (Newman, 2010), su centralidad (PageRank) y sus comunidades (Markov Stability), en comparación con el modelo central de *E. coli*.

## Contenido

| Archivo o carpeta | Qué es |
|---|---|
| `RU47_model.xml` | Modelo metabólico de RU47 en formato SBML, reconstruido con CarveMe \[versión\] a partir del ensamblaje NCBI GCA_004011755.1 (descargado el 30 de agosto de 2026). El archivo tiene 3.152 reacciones; al leerlo, COBRApy añade 12 intercambios, para un total de 3.164 reacciones, 2.016 metabolitos y 1.777 genes. |
| `iJN1462.json` | Modelo de *P. putida* KT2440 (BiGG Models; Nogales et al., 2020). Solo se usa para asignar rutas a las reacciones de RU47. |
| `flujo.py` | Funciones comunes: medio de cultivo, desdoblamiento de reacciones, NFG, MFG, PageRank, anotación de rutas y Markov Stability. |
| `paso1_rutas.py` a `paso6_rag_y_red.py` | Análisis en el orden del documento (ver abajo). |
| `nfg_gephi.py` | Genera los archivos `.gexf` del NFG para Gephi. |
| `LEEME_analisis.txt` | Orden de ejecución, tiempos y decisiones de método. |
| `requirements.txt` | Librerías de Python y versiones usadas. |
| `resultados/` | Tablas (`.csv`), figuras (`.png`), grafos para Gephi (`.gexf`) y resultados de Markov Stability ya calculados (`.pkl`). |
| `resultados/figuras_gephi/` | Figuras de la red elaboradas en Gephi. |

## Cómo reproducir el análisis

1. Instalar Python 3.11 o superior y las librerías:
   ```
   pip install -r requirements.txt
   ```
2. Desde esta carpeta, ejecutar los pasos en orden:
   ```
   python paso1_rutas.py          # rutas de cada reacción y PageRank por ruta
   python paso2_comunidades.py    # comunidades con Markov Stability
   python paso3_condiciones.py    # MFG en ocho condiciones de crecimiento
   python paso4_rigor.py          # modelos nulos, distribuciones de grado y sensibilidad del FBA
   python paso5_robustez.py       # percolación y esencialidad
   python paso6_rag_y_red.py      # grafo de adyacencia de reacciones frente al NFG y figura de la red
   python nfg_gephi.py            # archivos .gexf del NFG para Gephi
   ```
   Los resultados se guardan en `resultados/`.

**Notas.**
- Las comunidades del NFG y del RAG de RU47 tardan horas en calcularse. Si `resultados/ms_RU47_NFG.pkl` y `resultados/ms_RU47_RAG.pkl` están presentes, los pasos 2 y 6 los leen y no repiten el cálculo.
- Los pasos 1 y 3 descargan el modelo iJO1366 de *E. coli* desde BiGG Models, por lo que necesitan conexión a internet. El modelo central de *E. coli* viene incluido en COBRApy.
- Markov Stability usa optimizaciones aleatorias, así que al repetirlo pueden cambiar ligeramente los tiempos de Markov seleccionados; en las pruebas hechas, el número de comunidades robustas se mantuvo.

## Parámetros principales

- Medio M9 mínimo con la fuente de carbono limitada a 18,5 mmol gDW⁻¹ h⁻¹.
- Flujos calculados con FBA parsimonioso (pFBA).
- PageRank y Markov Stability con λ = 0,85.
- Se desdobla toda reacción con cota inferior negativa y todo intercambio.

## Referencias principales

- Beguerisse-Díaz, M., Bosque, G., Oyarzún, D., Picó, J. y Barahona, M. (2018). Flux-dependent graphs for metabolic networks. *npj Systems Biology and Applications, 4*, 32. https://doi.org/10.1038/s41540-018-0067-y
- Newman, M. E. J. (2010). *Networks: An introduction*. Oxford University Press.
- Machado, D., Andrejev, S., Tramontano, M. y Patil, K. R. (2018). Fast automated reconstruction of genome-scale metabolic models for microbial species and communities. *Nucleic Acids Research, 46*(15), 7542–7553. https://doi.org/10.1093/nar/gky537
