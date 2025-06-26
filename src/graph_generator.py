import pandas as pd
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import typing as t

def is_generator(row: pd.Series) -> bool:
    """
    Identifica si un nodo es un generador basado en su capacidad y tensión.

    Un nodo se considera generador si su capacidad instalada (POT_INST) es
    superior a 100 MW y su tensión primaria (TENSION_1) está en una lista
    predefinida de tensiones de distribución/generación.

    Args:
        row (pd.Series): Una fila del DataFrame de nodos, que contiene
                         'POT_INST' y 'TENSION_1'.

    Returns:
        bool: True si se clasifica como generador, False en caso contrario.
    """
    # Esta es la lista original de tensiones que se consideraban para generadores.
    # Puede ser ajustada según un análisis más detallado de los datos.
    GENERATOR_TENSIONS = ['10 kV', '13,2 kV', '22,9 kV']

    try:
        # Se convierte POT_INST a float, ignorando errores.
        pot_inst = float(row['POT_INST'])
        return (pot_inst > 100) and (row['TENSION_1'] in GENERATOR_TENSIONS)
    except (ValueError, TypeError):
        return False

def create_graph_from_csv(
    file_path: str,
    num_nodes: t.Optional[int] = None,
    k_neighbors: int = 3,
    dbscan_eps: float = 0.08
) -> t.Tuple[nx.Graph, pd.DataFrame]:
    """
    Carga datos de subestaciones desde un CSV, los limpia, genera un grafo de red eléctrica
    y devuelve el grafo y el DataFrame de nodos procesado.

    Args:
        file_path (str): Ruta al archivo CSV que contiene los datos de las subestaciones.
        num_nodes (Optional[int]): Número opcional de nodos a muestrear del DataFrame.
                                    Si es None, se usan todos los nodos.
        k_neighbors (int): Número de vecinos a considerar para la conexión de nodos dentro de un clúster.
        dbscan_eps (float): El radio máximo de la vecindad para DBSCAN.

    Returns:
        Tuple[nx.Graph, pd.DataFrame]: Una tupla que contiene el grafo de NetworkX
                                       y el DataFrame de nodos procesado.
    """
    # 1. Carga y Limpieza de Datos
    cols_to_use = [
        'COD', 'LATITUD', 'LONGITUD', 'POT_INST', 'TENSION_1', 'TENSION_2',
        'PROPIEDAD', 'NOM_SIST', 'EMPRESA'
    ]
    nodes_df = pd.read_csv(file_path, encoding='utf-8', usecols=cols_to_use)

    # Limpieza de columnas numéricas clave para asegurar que sean floats/integers.
    for col in ['LATITUD', 'LONGITUD', 'POT_INST']:
        # Se eliminan caracteres no numéricos y se convierte a tipo numérico.
        # Los errores de conversión se reemplazan con NaN (Not a Number).
        nodes_df[col] = pd.to_numeric(nodes_df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')

    # Se eliminan filas con valores nulos en columnas críticas para la visualización y el análisis.
    nodes_df.dropna(subset=['LATITUD', 'LONGITUD', 'POT_INST'], inplace=True)
    nodes_df.reset_index(drop=True, inplace=True)

    # 2. Submuestreo de Nodos
    if num_nodes and num_nodes < len(nodes_df):
        # Se usa random_state para que el muestreo sea reproducible.
        nodes_df = nodes_df.sample(n=num_nodes, random_state=42).reset_index(drop=True)

    # 3. Clasificación de Nodos
    nodes_df['type'] = nodes_df.apply(is_generator, axis=1)

    # 4. Clustering Geográfico
    # Se usa DBSCAN para agrupar nodos geográficamente cercanos en "clústeres" o "ciudades".
    coords_all = nodes_df[['LATITUD', 'LONGITUD']].to_numpy()
    db = DBSCAN(eps=dbscan_eps, min_samples=1).fit(coords_all)
    nodes_df['city'] = db.labels_

    # 5. Construcción del Grafo
    # Se crea un grafo final conectando nodos dentro de cada clúster.
    G = nx.Graph()
    for city_id, group in nodes_df.groupby('city'):
        group = group.reset_index(drop=True)
        city_graph = nx.Graph() # Grafo temporal para el clúster actual.
        for _, r in group.iterrows():
            city_graph.add_node(r['COD'], pot_inst=float(r['POT_INST']),
                                pos=(r['LONGITUD'], r['LATITUD']), type=r['type'],
                                city=city_id, tension_1=r['TENSION_1'],
                                tension_2=r['TENSION_2'], propiedad=r['PROPIEDAD'],
                                sistema=r['NOM_SIST'], empresa=r['EMPRESA'])
        
        # Se conectan los nodos dentro del clúster a sus 'k' vecinos más cercanos.
        if len(group) > 1:
            coords = group[['LATITUD', 'LONGITUD']].to_numpy()
            k = min(k_neighbors + 1, len(group))
            nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
            dists, idx = nbrs.kneighbors(coords)
            for i, neigh in enumerate(idx):
                for j, dist in zip(neigh[1:], dists[i][1:]): # Se omite el primer vecino (el nodo mismo).
                    city_graph.add_edge(group.at[i, 'COD'], group.at[j, 'COD'], weight=dist)
            # Se usa un Árbol de Expansión Mínima (MST) para asegurar la conectividad básica sin ciclos redundantes.
            city_MST = nx.minimum_spanning_tree(city_graph, weight='weight')
        else:
            city_MST = city_graph

        # Se añade el grafo del clúster al grafo principal.
        G.add_nodes_from(city_MST.nodes(data=True))
        G.add_edges_from(city_MST.edges(data=True))

    return G, nodes_df