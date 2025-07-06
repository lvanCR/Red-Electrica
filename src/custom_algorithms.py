import numpy as np
import heapq
import math
import random
import typing as t
import networkx as nx
import pandas as pd

# Constante para la simulación de contingencia
CONTINGENCY_LINE_CAPACITY_MW = 75.0

def my_dijkstra_for_centrality(graph: nx.Graph, start_node: t.Any) -> t.Tuple[t.Dict, t.Dict, t.List]:
    """
    Implementación de Dijkstra para encontrar los caminos más cortos desde un nodo de inicio.
    Esta versión está adaptada para el cálculo de la centralidad de intermediación.

    Devuelve:
    - sigma: Un diccionario que mapea cada nodo al número de caminos más cortos desde start_node.
    - predecessors: Un diccionario que mapea cada nodo a una lista de sus predecesores en los caminos más cortos.
    - stack: Una pila de nodos en orden de distancia no creciente desde start_node.
    """
    sigma = {node: 0 for node in graph.nodes()}
    sigma[start_node] = 1
    
    distances = {node: float('inf') for node in graph.nodes()}
    distances[start_node] = 0
    
    predecessors = {node: [] for node in graph.nodes()}
    
    pq = [(0, start_node)] # (distancia, nodo)
    stack = []

    while pq:
        dist, u = heapq.heappop(pq)

        if dist > distances[u]:
            continue
        
        stack.append(u)

        for v, edge_data in graph[u].items():
            weight = edge_data.get('weight', 1)
            new_dist = distances[u] + weight

            if new_dist < distances[v]:
                distances[v] = new_dist
                sigma[v] = sigma[u]
                predecessors[v] = [u]
                heapq.heappush(pq, (new_dist, v))
            elif new_dist == distances[v]:
                sigma[v] += sigma[u]
                predecessors[v].append(u)

    return sigma, predecessors, stack

def my_betweenness_centrality(graph: nx.Graph) -> t.Dict[t.Any, float]:
    """
    Implementación desde cero del algoritmo de Brandes para calcular la centralidad de intermediación.
    """
    centrality = {node: 0.0 for node in graph.nodes()}

    for s in graph.nodes():
        sigma, predecessors, stack = my_dijkstra_for_centrality(graph, s)
        
        delta = {node: 0.0 for node in graph.nodes()}
        
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                if sigma[w] != 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                centrality[w] += delta[w]

    return centrality


def my_kmeans(points: np.ndarray, k: int, max_iters: int = 100) -> np.ndarray:
    """
    Implementación desde cero del algoritmo K-Means.

    Args:
        points (np.array): Un array de NumPy de puntos (coordenadas).
        k (int): El número de clústeres a encontrar.
        max_iters (int): Número máximo de iteraciones.

    Returns:
        np.array: Un array con las coordenadas de los centroides finales.
    """
    # 1. Inicializar centroides aleatoriamente de forma reproducible
    random.seed(42)
    random_indices = random.sample(range(len(points)), k)
    centroids = points[random_indices]

    for _ in range(max_iters):
        # 2. Asignar cada punto al centroide más cercano
        clusters = [[] for _ in range(k)]
        for point in points:
            distances = np.linalg.norm(point - centroids, axis=1)
            closest_centroid_idx = np.argmin(distances)
            clusters[closest_centroid_idx].append(point)

        new_centroids = np.zeros((k, points.shape[1]))
        for i, cluster in enumerate(clusters):
            if cluster:
                new_centroids[i] = np.mean(cluster, axis=0)
            else: # Si un clúster queda vacío, se re-inicializa
                new_centroids[i] = points[random.randint(0, len(points) - 1)]

        # 3. Comprobar convergencia
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids

    return centroids

def find_contingency_solution(
    subgraph: nx.Graph,
    nodes_df: pd.DataFrame,
    isolated_nodes: t.List[t.Any],
    healthy_nodes: t.Set[t.Any]
) -> t.Tuple[t.List[t.Tuple], t.Set[t.Tuple]]:
    """
    Orquesta la lógica para encontrar una solución de contingencia, usando my_kmeans.
    """
    suggested_connections = []
    re_energized_edges = set()

    if not healthy_nodes or not isolated_nodes:
        return suggested_connections, re_energized_edges

    isolated_nodes_df = nodes_df[nodes_df['COD'].isin(isolated_nodes)]
    total_isolated_power_mw = isolated_nodes_df['POT_INST'].sum()

    # Calcular el número de conexiones necesarias
    num_connections_needed = math.ceil(total_isolated_power_mw / CONTINGENCY_LINE_CAPACITY_MW) if total_isolated_power_mw > 0 else 1
    num_connections_needed = min(num_connections_needed, len(isolated_nodes))

    candidate_pool = list(healthy_nodes)
    candidate_coords = np.array([subgraph.nodes[n]['pos'] for n in candidate_pool])
    iso_coords = isolated_nodes_df[['LONGITUD', 'LATITUD']].to_numpy()

    # Usar K-Means para encontrar puntos de conexión estratégicos
    if num_connections_needed > 1 and len(isolated_nodes) >= num_connections_needed:
        centroids = my_kmeans(iso_coords, k=num_connections_needed)
        nodes_to_connect = []
        for center in centroids:
            distances_to_center = np.linalg.norm(iso_coords - center, axis=1)
            closest_node_idx = distances_to_center.argmin()
            nodes_to_connect.append(isolated_nodes[closest_node_idx])
    else: # Si solo se necesita 1 conexión, encontrar el puente más corto
        dist_matrix = np.linalg.norm(iso_coords[:, np.newaxis, :] - candidate_coords[np.newaxis, :, :], axis=2)
        min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        nodes_to_connect = [isolated_nodes[min_idx[0]]]

    # Para cada nodo estratégico, encontrar su conexión sana más cercana
    for node_code in set(nodes_to_connect):
        node_pos = np.array(subgraph.nodes[node_code]['pos'])
        distances_to_candidates = np.linalg.norm(candidate_coords - node_pos, axis=1)
        best_candidate_idx = distances_to_candidates.argmin()
        best_candidate = candidate_pool[best_candidate_idx]
        suggested_connections.append((node_code, best_candidate))

    # Las líneas internas se re-energizan en cualquier caso
    isolated_subgraph = subgraph.subgraph(isolated_nodes)
    re_energized_edges = set(isolated_subgraph.edges())

    return suggested_connections, re_energized_edges