import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import networkx as nx
import numpy as np
import os
from src.graph_generator import create_graph_from_csv
from src.custom_algorithms import my_betweenness_centrality, find_contingency_solution

# --- 1. CONFIGURACIÓN DE LA PÁGINA Y ESTILO ---
st.set_page_config(
    layout="wide",
    page_title="Análisis de Red Eléctrica de Perú",
    page_icon="⚡"
)

# Paleta de colores y estilo CSS para una interfaz más moderna
PRIMARY_COLOR = "#1f77b4" # Azul principal
ACCENT_COLOR = "#ff7f0e"  # Naranja para acentos
SUCCESS_COLOR = "#2ca02c" # Verde para éxito
DANGER_COLOR = "#d62728"  # Rojo para peligro/fallas
GRAY_COLOR = "#7f7f7f"    # Gris para elementos neutros/aislados

st.markdown(f"""
<style>
    .stButton>button {{
        background-color: {PRIMARY_COLOR};
        color: white;
        border-radius: 8px;
        border: 2px solid {PRIMARY_COLOR};
    }}
    .stButton>button:hover {{
        background-color: white;
        color: {PRIMARY_COLOR};
    }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align: center;">
    <h1>Grafo de la Red Eléctrica Peruana</h1>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align: left;">
    Una herramienta interactiva para el análisis de la red de subestaciones de distribución eléctrica del Perú, basada en datos públicos del <b>Organismo Supervisor de la Inversión en Energía y Minería (OSINERGMIN)</b>.
    <br>
    <hr>
    <br>
    <i>Nota: La información presentada es de carácter referencial, consolidada a partir de los reportes suministrados por las Empresas Distribuidoras de Electricidad.</i>
</div>
""", unsafe_allow_html=True)

# --- 2. FUNCIONES Y ESTADO DE LA APLICACIÓN ---
def reset_analysis_states():
    """Resetea todas las variables de estado de los análisis para evitar solapamientos."""
    st.session_state.shortest_path = None
    st.session_state.start_node = None
    st.session_state.end_node = None
    st.session_state.centrality = None
    st.session_state.failed_nodes = set() # Cambiado a un conjunto para fallas múltiples
    st.session_state.isolated_nodes = set()
    st.session_state.suggested_connections = []
    st.session_state.re_energized_edges = set()
    st.session_state.solution_calculated = False
    st.session_state.last_processed_click_coords = None

# @st.cache_data asegura que los datos se carguen solo una vez, mejorando el rendimiento.
@st.cache_data
def load_data(num_nodes):
    """
    Carga y procesa los datos del grafo. La ejecución de esta función se guarda en caché.
    """
    # Construir una ruta absoluta al archivo de datos para evitar errores de 'FileNotFound'
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'data', 'SED.csv')

    graph, nodes_df = create_graph_from_csv(
        file_path,
        num_nodes=num_nodes
    )
    return graph, nodes_df

# --- 3. BARRA LATERAL DE CONTROLES ---
st.sidebar.title("⚡ Panel de Control")

# Slider para seleccionar el número de nodos
num_nodes_selection = st.sidebar.slider(
    "Seleccione el número de nodos a analizar:",
    min_value=500,
    max_value=20000,
    value=2000,
    step=500,
    help="Un número menor de nodos acelera la carga. Un número mayor ofrece un análisis más completo."
)

# Cargar los datos basados en la selección del slider
graph, nodes_df = load_data(num_nodes_selection)

# Menú desplegable para filtrar por sistema eléctrico
sistemas = ['Todo el País'] + sorted(nodes_df['NOM_SIST'].unique().tolist())
selected_system = st.sidebar.selectbox(
    "Filtrar por Sistema Eléctrico:",
    options=sistemas,
    help="Seleccione un sistema para enfocar el mapa en una región específica."
)

st.sidebar.header("🎨 Opciones de Visualización")
map_theme = st.sidebar.selectbox("Tema del Mapa:", ["Claro (Positron)", "Oscuro (Dark Matter)"])

# Selección de Modo de Análisis
st.sidebar.header("🔬 Modo de Análisis")
analysis_mode = st.sidebar.radio(
    "Seleccione una herramienta de análisis:",
    ['Exploración de la Red', 'Análisis de Ruta Óptima', 'Análisis de Centralidad', 'Análisis de Vulnerabilidad'],
    on_change=reset_analysis_states # Resetea el estado al cambiar de modo
)

# Inicialización del estado de la sesión para guardar los resultados entre interacciones.
if 'shortest_path' not in st.session_state:
    st.session_state.shortest_path = None
    st.session_state.start_node = None
    st.session_state.end_node = None
if 'centrality' not in st.session_state:
    st.session_state.centrality = None
if 'failed_nodes' not in st.session_state: # Cambiado a plural
    st.session_state.failed_nodes = set() # Almacena los nodos en los que se ha simulado una falla
    st.session_state.isolated_nodes = set()
if 'suggested_connections' not in st.session_state:
    st.session_state.suggested_connections = []
if 're_energized_edges' not in st.session_state:
    st.session_state.re_energized_edges = set()
if 'solution_calculated' not in st.session_state:
    st.session_state.solution_calculated = False
if 'last_processed_click_coords' not in st.session_state:
    st.session_state.last_processed_click_coords = None


# --- 4. LÓGICA DE FILTRADO Y PREPARACIÓN DEL MAPA ---
if selected_system == 'Todo el País':
    filtered_nodes_df = nodes_df
    # Si se visualiza todo el país, el subgrafo de análisis es el grafo completo.
    subgraph = graph
else:
    filtered_nodes_df = nodes_df[nodes_df['NOM_SIST'] == selected_system]
    # Creamos un subgrafo que contiene solo los nodos del sistema seleccionado
    nodes_in_system = filtered_nodes_df['COD'].tolist()
    subgraph = graph.subgraph(nodes_in_system)

nodes_to_display = set(filtered_nodes_df['COD'])

# Centrar el mapa. Si no hay nodos filtrados, se usa una ubicación por defecto para Perú.
if not filtered_nodes_df.empty:
    map_center = [filtered_nodes_df['LATITUD'].mean(), filtered_nodes_df['LONGITUD'].mean()]
    zoom_start = 6 if selected_system == 'Todo el País' else 8
else:
    map_center = [-9.19, -75.01] # Centro de Perú
    zoom_start = 5

tile_layer = "cartodbpositron" if map_theme == "Claro (Positron)" else "cartodbdark_matter"
m = folium.Map(location=map_center, zoom_start=zoom_start, tiles=tile_layer)

# Lógica específica para cada modo de análisis en la barra lateral
if analysis_mode == 'Análisis de Ruta Óptima':
    st.sidebar.subheader("Configurar Ruta")
    
    # Crear una lista de nodos para los selectores
    node_list = sorted(filtered_nodes_df['COD'].tolist())
    
    start_node = st.sidebar.selectbox("Seleccione Origen:", options=node_list, index=0 if node_list else -1)
    end_node = st.sidebar.selectbox("Seleccione Destino:", options=node_list, index=len(node_list)-1 if node_list else -1)

    if st.sidebar.button("Calcular Ruta Óptima"):
        try:
            # Se usa el algoritmo de Dijkstra sobre el subgrafo filtrado para encontrar la ruta más corta.
            path = nx.shortest_path(subgraph, source=start_node, target=end_node, weight='weight') if start_node and end_node else []
            st.session_state.shortest_path = path
            st.session_state.start_node = start_node
            st.session_state.end_node = end_node
            st.sidebar.success(f"Ruta encontrada con {len(path)} nodos.")
        except nx.NetworkXNoPath:
            st.sidebar.error("No existe una ruta entre los nodos seleccionados.")
            st.session_state.shortest_path = None
            st.session_state.start_node = None
            st.session_state.end_node = None
        except nx.NodeNotFound:
            st.sidebar.error("Uno de los nodos no fue encontrado en el grafo.")
            st.session_state.shortest_path = None
            st.session_state.start_node = None
            st.session_state.end_node = None

elif analysis_mode == 'Análisis de Centralidad':
    st.sidebar.subheader("Puntos Críticos de la Red")
    if st.sidebar.button("Calcular Criticidad de la Red"):
        if len(subgraph.nodes) > 0:
            with st.spinner("Calculando centralidad... Esto puede tardar unos segundos."):
                # 1. Usamos nuestra implementación personalizada (que devuelve valores no normalizados)
                centrality = my_betweenness_centrality(subgraph)

                # 2. Normalizamos los resultados de 0 a 1 para una visualización consistente
                max_centrality = max(centrality.values()) if centrality else 0
                if max_centrality > 0:
                    normalized_centrality = {node: value / max_centrality for node, value in centrality.items()}
                else:
                    normalized_centrality = centrality # Evitar división por cero

                st.session_state.centrality = normalized_centrality
                st.sidebar.success("Cálculo de centralidad completado.")
        else:
            st.sidebar.warning("No hay nodos en la vista actual para analizar.")

elif analysis_mode == 'Análisis de Vulnerabilidad':
    st.sidebar.info("Haz clic en un nodo del mapa para simular su desconexión.")
    
    # El botón de solución solo aparece si hay una falla simulada y no se ha calculado una solución aún.
    if st.session_state.failed_nodes and not st.session_state.solution_calculated:
        if st.sidebar.button("💡 Buscar Solución"):
            isolated_nodes = list(st.session_state.isolated_nodes)
            healthy_nodes = set(subgraph.nodes()) - st.session_state.failed_nodes - st.session_state.isolated_nodes

            # Llamada a nuestra nueva función de orquestación que usa my_kmeans
            suggested_connections, re_energized_edges = find_contingency_solution(
                subgraph,
                nodes_df,
                isolated_nodes,
                healthy_nodes
            )
            st.session_state.suggested_connections = suggested_connections
            st.session_state.re_energized_edges = re_energized_edges
            
            st.session_state.solution_calculated = True
            st.rerun()

    if st.sidebar.button("Resetear Simulación"):
        reset_analysis_states()
        st.rerun() # Forzar la recarga de la app

# --- 5. DIBUJAR NODOS Y ARISTAS EN EL MAPA ---
if not filtered_nodes_df.empty:
    # Dibujar aristas (conexiones)
    for u, v, data in graph.edges(data=True):
        if u in nodes_to_display and v in nodes_to_display:
            edge_color = GRAY_COLOR
            edge_weight = 1.5
            edge_opacity = 0.7

            is_re_energized = (u, v) in st.session_state.re_energized_edges or (v, u) in st.session_state.re_energized_edges

            if st.session_state.solution_calculated and is_re_energized:
                edge_color = "#87CEEB"  # Celeste / Sky Blue
                edge_weight = 2.5
                edge_opacity = 0.9
            elif st.session_state.shortest_path:
                # Resaltar la ruta óptima si está calculada
                path_edges = list(zip(st.session_state.shortest_path, st.session_state.shortest_path[1:]))
                if (u, v) in path_edges or (v, u) in path_edges:
                    edge_color = PRIMARY_COLOR # Azul brillante
                    edge_weight = 4
                    edge_opacity = 1.0


            pos_u = graph.nodes[u]['pos']
            pos_v = graph.nodes[v]['pos']
            folium.PolyLine(
                locations=[(pos_u[1], pos_u[0]), (pos_v[1], pos_v[0])],
                color=edge_color,
                weight=edge_weight,
                opacity=edge_opacity
            ).add_to(m)

    # Dibujar nodos (subestaciones)
    for _, node in filtered_nodes_df.iterrows():
        tooltip_html = f"""
        <b>Subestación: {node['COD']}</b><br>
        --------------------<br>
        <b>Tipo:</b> {'Generador' if node['type'] else 'Distribuidor'}<br>
        <b>Sistema:</b> {node['NOM_SIST']}<br>
        <b>Capacidad:</b> {node['POT_INST']:.2f} MW<br>
        <b>Tensión Primaria:</b> {node['TENSION_1']}<br>
        <b>Empresa:</b> {node['EMPRESA']}
        """
        
        # --- Lógica de Estilo de Nodos ---
        # Estilos por defecto
        node_color = SUCCESS_COLOR # Para distribuidores
        icon_color = DANGER_COLOR   # Para generadores
        node_radius = 4
        
        # Diferenciación visual: Generadores tendrán un punto base naranja
        if node['type']: # Si es generador
            node_color = ACCENT_COLOR # Punto base para generadores
            icon_color = DANGER_COLOR # Color del rayo para generadores

        is_failed_node = analysis_mode == 'Análisis de Vulnerabilidad' and node['COD'] in st.session_state.failed_nodes

        # Estilo dinámico según el modo de análisis activo
        if analysis_mode == 'Análisis de Ruta Óptima' and st.session_state.shortest_path and node['COD'] in st.session_state.shortest_path:
            if node['COD'] == st.session_state.start_node:
                # Nodo de Origen: Azul para destacarlo
                node_color = PRIMARY_COLOR
                icon_color = 'blue'
                node_radius = 8
            elif node['COD'] == st.session_state.end_node:
                # Nodo de Destino: Rojo brillante
                node_color = DANGER_COLOR
                icon_color = DANGER_COLOR
                node_radius = 8
            else:
                # Nodos intermedios: Verde para indicar la ruta
                node_color = SUCCESS_COLOR
                icon_color = 'green'
                node_radius = 6
        
        elif analysis_mode == 'Análisis de Centralidad' and st.session_state.centrality:
            centrality_value = st.session_state.centrality.get(node['COD'], 0)
            # Escalar el radio y el color basado en la centralidad.
            # Los nodos más críticos se visualizan más grandes y de color más intenso (rojo). Ajustado para valores normalizados (0-1).
            node_radius = 4 + centrality_value * 15  # Escala el radio
            # Interpola el color de verde a rojo basado en la centralidad.
            if centrality_value > 0.1: # Umbral para destacar (ajustado para normalización 0-1)
                red_component = int(min(255, 255 * centrality_value * 20))
                green_component = int(max(0, 255 - (255 * centrality_value * 10)))
                node_color = f'#{red_component:02x}{green_component:02x}00'
                icon_color = ACCENT_COLOR if centrality_value < 0.1 else DANGER_COLOR
        
        elif analysis_mode == 'Análisis de Vulnerabilidad':
            if is_failed_node:
                # El nodo fallido se dibuja con un ícono especial y se omite el dibujo normal.
                pass
            elif node['COD'] in st.session_state.isolated_nodes:
                if st.session_state.solution_calculated:
                    # Nodo ahora considerado "restaurado"
                    node_color = PRIMARY_COLOR
                    icon_color = 'blue'
                else:
                    # Nodo aislado, sin solución aún
                    node_color = GRAY_COLOR
                    icon_color = 'lightgray'

        # --- Lógica de Dibujo en el Mapa ---
        if is_failed_node:
            folium.Marker(
                location=[node['LATITUD'], node['LONGITUD']],
                tooltip=f"FALLA SIMULADA: {node['COD']}",
                icon=folium.Icon(prefix='fa', icon='times-circle', color='black')
            ).add_to(m)
        else:
            # Se dibuja un CircleMarker como base para todos los nodos. Es fiable para clics.
            folium.CircleMarker(
                location=[node['LATITUD'], node['LONGITUD']],
                radius=node_radius,
                color=node_color,
                fill=True,
                fill_color=node_color,
                fill_opacity=0.7,
                tooltip=tooltip_html
            ).add_to(m)

            # Si el nodo es un generador, se añade el icono del rayo encima del círculo base.
            if node['type']:
                folium.Marker(
                    location=[node['LATITUD'], node['LONGITUD']],
                    icon=folium.Icon(prefix='fa', icon='bolt', color=icon_color),
                    interactive=False, # El icono es solo visual y no interfiere con los clics.
                    tooltip=tooltip_html # El tooltip se muestra al pasar el cursor por el icono
                ).add_to(m)

    # Dibujar conexiones de contingencia sugeridas
    if analysis_mode == 'Análisis de Vulnerabilidad' and st.session_state.suggested_connections:
        for u, v in st.session_state.suggested_connections:
            if u in graph.nodes and v in graph.nodes:
                pos_u = graph.nodes[u]['pos']
                pos_v = graph.nodes[v]['pos']
                folium.PolyLine(
                    locations=[(pos_u[1], pos_u[0]), (pos_v[1], pos_v[0])],
                    color='#00FFFF',  # Cian brillante para destacar
                    weight=3,
                    opacity=0.9,
                    dash_array='10, 5', # Línea punteada
                    tooltip=f"CONEXIÓN DE CONTINGENCIA: {u} a {v}"
                ).add_to(m)


# --- 6. MOSTRAR EL MAPA EN STREAMLIT ---
map_output = st_folium(m, width='100%', height=600, returned_objects=['last_clicked'])

# --- 7. LEYENDAS Y DATOS ---
st.markdown("---") # Separador visual
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Leyenda")
    # HTML para una leyenda más visual
    legend_html = f"""
    <ul style="list-style-type: none; padding-left: 0;">
        <li><span style='background-color:{ACCENT_COLOR}; border-radius:50%; display: inline-block; width: 12px; height: 12px; margin-right: 5px;'></span><i class='fa fa-bolt' style='color:{DANGER_COLOR};'></i> Nodo Generador</li>
        <li><span style='background-color:{SUCCESS_COLOR}; border-radius:50%; display: inline-block; width: 12px; height: 12px; margin-right: 5px;'></span> Nodo Distribuidor</li>
        <li><span style='background-color:{GRAY_COLOR}; border-radius:50%; display: inline-block; width: 12px; height: 12px; margin-right: 5px;'></span> Nodo Aislado</li>
        <li><i class='fa fa-times-circle' style='color:black; margin-left: -2px; margin-right: 5px;'></i> Falla Simulada</li>
        <hr style='margin: 5px 0;'>
        <li><div style='border-bottom: 4px dashed #00FFFF; width: 20px; display: inline-block; vertical-align: middle; margin-right: 5px;'></div> Conexión de Contingencia</li>
        <li><div style='background-color:#87CEEB; height: 4px; width: 20px; display: inline-block; vertical-align: middle; margin-right: 5px;'></div> Línea Re-energizada</li>
        <li><div style='background-color:{PRIMARY_COLOR}; height: 4px; width: 20px; display: inline-block; vertical-align: middle; margin-right: 5px;'></div> Línea de Ruta Óptima</li>
    </ul>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

with col2:
    st.subheader("ℹ️ Información de Análisis")
    if analysis_mode == 'Análisis de Centralidad' and st.session_state.centrality:
        st.info("Los nodos más grandes y rojizos son los más críticos en la red. Su falla tendría un impacto significativo en la conectividad.")
    elif analysis_mode == 'Análisis de Vulnerabilidad' and st.session_state.failed_nodes:
        info_message = (
            f"**Simulación Activa:**\n\n"
            f"- **Nodos con Fallo:** {len(st.session_state.failed_nodes)}\n"
            f"- **Nodos Aislados:** {len(st.session_state.isolated_nodes)}"
        )
        if st.session_state.solution_calculated:
            info_message += "\n\n---\n\n**Solución Propuesta:**\n\n"
            info_message += (f"Se sugieren **{len(st.session_state.suggested_connections)}** nueva(s) conexión(es) "
                             f"para re-energizar {len(st.session_state.re_energized_edges)} líneas internas.")
            st.success(info_message)
        elif st.session_state.failed_nodes:
            info_message += "\n\n*Presione 'Buscar Solución' para calcular una contingencia.*"
            st.warning(info_message)
    elif analysis_mode == 'Análisis de Ruta Óptima' and st.session_state.shortest_path:
        st.success(f"Mostrando ruta óptima entre **{st.session_state.start_node}** y **{st.session_state.end_node}**.")
    else:
        st.info("Seleccione un modo de análisis y ejecute una acción para ver los resultados aquí.")

with st.expander("📖 Ver Tabla de Datos de Nodos Filtrados"):
    # Mostrar una versión más limpia del dataframe
    display_df = filtered_nodes_df[['COD', 'type', 'NOM_SIST', 'POT_INST', 'TENSION_1', 'EMPRESA']].copy()
    display_df.rename(columns={
        'COD': 'Código', 'type': 'Tipo', 'NOM_SIST': 'Sistema',
        'POT_INST': 'Capacidad (MW)', 'TENSION_1': 'Tensión', 'EMPRESA': 'Empresa'
    }, inplace=True)
    display_df['Tipo'] = display_df['Tipo'].apply(lambda x: 'Generador' if x else 'Distribuidor')
    st.dataframe(display_df, use_container_width=True)

# --- 8. LÓGICA POST-RENDERIZADO PARA CLICS EN EL MAPA ---
if analysis_mode == 'Análisis de Vulnerabilidad' and map_output and map_output.get("last_clicked"):
    clicked_coords = map_output["last_clicked"]
    current_click_key = f"{clicked_coords['lat']},{clicked_coords['lng']}"

    # Solo se procesa si es un clic nuevo para evitar reprocesamiento por el rerender de Streamlit.
    if st.session_state.last_processed_click_coords != current_click_key:
        st.session_state.last_processed_click_coords = current_click_key

        # --- 1. Detectar Clic y Encontrar Nodo (Optimizado) ---
        if not filtered_nodes_df.empty:
            # Convertir coordenadas de nodos a un array de NumPy para cálculo vectorizado.
            node_coords = filtered_nodes_df[['LATITUD', 'LONGITUD']].to_numpy()
            click_coord = np.array([clicked_coords['lat'], clicked_coords['lng']])
            
            # Calcular la distancia de todos los nodos al punto de clic de una sola vez.
            distances = np.linalg.norm(node_coords - click_coord, axis=1)
            
            # Encontrar el índice del nodo más cercano.
            closest_node_idx = distances.argmin()
            min_dist = distances[closest_node_idx]

            CLICK_THRESHOLD = 0.02
            if min_dist < CLICK_THRESHOLD:
                closest_node = filtered_nodes_df.iloc[closest_node_idx]
                closest_node_cod = closest_node['COD']

                # --- 2. Simular Falla y Calcular Nodos Aislados ---
                if closest_node_cod not in st.session_state.failed_nodes:                    
                    # Resetear cualquier solución previa antes de simular una nueva falla
                    st.session_state.solution_calculated = False
                    st.session_state.suggested_connections = []
                    st.session_state.re_energized_edges = set()

                    st.session_state.failed_nodes.add(closest_node_cod)                    
                    
                    # Análisis de impacto localizado por clúster ('city').
                    city_id = closest_node['city']
                    nodes_in_city = set(filtered_nodes_df[filtered_nodes_df['city'] == city_id]['COD'])
                    city_subgraph = subgraph.subgraph(nodes_in_city).copy()
                    failed_nodes_in_city = st.session_state.failed_nodes.intersection(nodes_in_city)
                    city_subgraph.remove_nodes_from(failed_nodes_in_city)

                    if city_subgraph.number_of_nodes() > 0:
                        components = list(nx.connected_components(city_subgraph))
                        if components:
                            largest_component = max(components, key=len)
                            newly_isolated = (nodes_in_city - largest_component) - failed_nodes_in_city
                            st.session_state.isolated_nodes.update(newly_isolated)
                    st.rerun()
