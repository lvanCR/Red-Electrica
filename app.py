import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import networkx as nx
from src.graph_generator import create_graph_from_csv
import numpy as np
import os

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
    
    start_node = st.sidebar.selectbox("Seleccione Origen:", options=node_list, index=0)
    end_node = st.sidebar.selectbox("Seleccione Destino:", options=node_list, index=len(node_list)-1)

    if st.sidebar.button("Calcular Ruta Óptima"):
        try:
            # Se usa el algoritmo de Dijkstra sobre el subgrafo filtrado para encontrar la ruta más corta.
            path = nx.shortest_path(subgraph, source=start_node, target=end_node, weight='weight')
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
                # Se usa 'betweenness_centrality' para encontrar los nodos más críticos.
                centrality = nx.betweenness_centrality(subgraph, weight='weight', normalized=True)
                st.session_state.centrality = centrality
                st.sidebar.success("Cálculo de centralidad completado.")
        else:
            st.sidebar.warning("No hay nodos en la vista actual para analizar.")

elif analysis_mode == 'Análisis de Vulnerabilidad':
    st.sidebar.info("Haz clic en un nodo del mapa para simular su desconexión.")
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

            # Resaltar la ruta óptima si está calculada
            if st.session_state.shortest_path:
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
            # Los nodos más críticos se visualizan más grandes y de color más intenso (rojo).
            node_radius = 3 + centrality_value * 50  # Escala el radio
            # Interpola el color de verde a rojo basado en la centralidad.
            if centrality_value > 0.01: # Umbral para destacar
                red_component = int(min(255, 255 * centrality_value * 20))
                green_component = int(max(0, 255 - (255 * centrality_value * 10)))
                node_color = f'#{red_component:02x}{green_component:02x}00'
                icon_color = ACCENT_COLOR if centrality_value < 0.1 else DANGER_COLOR
        
        elif analysis_mode == 'Análisis de Vulnerabilidad':
            if is_failed_node:
                # El nodo fallido se dibuja con un ícono especial y se omite el dibujo normal.
                pass
            elif node['COD'] in st.session_state.isolated_nodes:
                # Nodos aislados se muestran en gris.
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
        <li><div style='background-color:{PRIMARY_COLOR}; height: 4px; width: 20px; display: inline-block; vertical-align: middle; margin-right: 5px;'></div> Línea de Ruta Óptima</li>
    </ul>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

with col2:
    st.subheader("ℹ️ Información de Análisis")
    if analysis_mode == 'Análisis de Centralidad' and st.session_state.centrality:
        st.info("Los nodos más grandes y rojizos son los más críticos en la red. Su falla tendría un impacto significativo en la conectividad.")
    elif analysis_mode == 'Análisis de Vulnerabilidad' and st.session_state.failed_nodes:
        st.warning(f"""
        **Simulación Activa:**
        - **Nodos con Fallo:** {len(st.session_state.failed_nodes)}
        - **Nodos Aislados:** {len(st.session_state.isolated_nodes)}
        """)
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
if analysis_mode == 'Análisis de Vulnerabilidad':
    if map_output and map_output.get("last_clicked"):
        clicked_coords = map_output["last_clicked"]
        lat, lon = clicked_coords['lat'], clicked_coords['lng']

        # Crear una clave única para el clic actual para evitar reprocesamiento
        current_click_key = f"{lat},{lon}"

        # Solo se procesa si es un clic nuevo (diferente al último procesado).
        if st.session_state.last_processed_click_coords != current_click_key:
            st.session_state.last_processed_click_coords = current_click_key

            # Encontrar el nodo del grafo más cercano a las coordenadas del clic.
            min_dist = float('inf')
            closest_node_cod = None

            # Iterar solo sobre los nodos actualmente visibles.
            if not filtered_nodes_df.empty:
                for _, node in filtered_nodes_df.iterrows():
                    # Calcular la distancia euclidiana entre el clic y cada nodo.
                    node_lat = node['LATITUD']
                    node_lon = node['LONGITUD']
                    dist = np.sqrt((node_lat - lat)**2 + (node_lon - lon)**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_node_cod = node['COD']
                
                # Umbral de distancia para considerar un clic válido sobre un nodo.
                CLICK_THRESHOLD = 0.02 # Reducido para mayor precisión
                
                if closest_node_cod and min_dist < CLICK_THRESHOLD:
                    # Solo se procesa la falla si el nodo no ha fallado previamente.
                    if closest_node_cod not in st.session_state.failed_nodes:
                        
                        # --- Análisis de Impacto Localizado ---
                        # 1. Identificar el clúster (ciudad) al que pertenece el nodo fallado.
                        city_id = filtered_nodes_df.loc[filtered_nodes_df['COD'] == closest_node_cod, 'city'].iloc[0]
                        nodes_in_city = set(filtered_nodes_df[filtered_nodes_df['city'] == city_id]['COD'])

                        # 2. Añadir el nodo al conjunto global de fallas para su visualización.
                        st.session_state.failed_nodes.add(closest_node_cod)
                        
                        # 3. Crear un subgrafo temporal que contenga solo los nodos del clúster afectado.
                        city_subgraph = subgraph.subgraph(nodes_in_city).copy()
                        
                        # 4. Simular la falla eliminando los nodos correspondientes DENTRO del subgrafo del clúster.
                        failed_nodes_in_city = st.session_state.failed_nodes.intersection(nodes_in_city)
                        city_subgraph.remove_nodes_from(failed_nodes_in_city)

                        # 5. Calcular los nodos aislados comparando los componentes restantes con el conjunto original del clúster.
                        if city_subgraph.number_of_nodes() > 0:
                            components = list(nx.connected_components(city_subgraph))
                            largest_component = max(components, key=len)
                            newly_isolated = (nodes_in_city - largest_component) - failed_nodes_in_city
                            st.session_state.isolated_nodes.update(newly_isolated)

                        st.rerun()
