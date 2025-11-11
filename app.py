import os
import time
from flask import Flask, render_template, request, redirect, url_for
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import heapq

app = Flask(__name__)

# Variable global para almacenar el grafo
G = None
pos = None
shortest_path_info = {}

def crear_grafo_de_ejemplo():
    """Crea un grafo de ejemplo no dirigido y ponderado."""
    global G, pos
    G = nx.Graph()
    nodos = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    G.add_nodes_from(nodos)
    aristas = [
        ('A', 'B', 5), ('A', 'C', 10),
        ('B', 'D', 8), ('B', 'E', 15),
        ('C', 'D', 4), ('C', 'F', 20),
        ('D', 'E', 6), ('D', 'F', 9),
        ('E', 'F', 12),
        ('F', 'G', 7), ('G', 'H', 3),
        ('H', 'I', 11), ('I', 'J', 14),
        ('J', 'A', 18), ('G', 'D', 5),
        ('H', 'E', 10)
    ]
    G.add_weighted_edges_from(aristas)
    # Posiciones para la visualización
    pos = nx.spring_layout(G, seed=42)

def dijkstra(graph, start):
    """Algoritmo de Dijkstra para encontrar el camino más corto desde un nodo de inicio."""
    distances = {node: float('infinity') for node in graph.nodes}
    distances[start] = 0
    previous_nodes = {node: None for node in graph.nodes}
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, data in graph[current_node].items():
            weight = data['weight']
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances, previous_nodes

def obtener_camino(previous_nodes, start, end):
    """Reconstruye el camino más corto desde el diccionario de nodos previos."""
    path = []
    current_node = end
    while current_node is not None:
        path.append(current_node)
        current_node = previous_nodes[current_node]
    path.reverse()
    if path[0] == start:
        return path
    return None

def dibujar_grafo(path=None, path_color='r'):
    """Dibuja el grafo y resalta un camino si se proporciona."""
    if G is None:
        if os.path.exists('static/graph.png'):
            os.remove('static/graph.png')
        return

    if not os.path.exists('static'):
        os.makedirs('static')

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Dibuja el grafo base
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=15, font_weight='bold', ax=ax)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)

    # Si hay un camino para resaltar
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color=path_color, node_size=2000, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color=path_color, width=2.5, ax=ax)

    plt.title("Grafo Ponderado No Dirigido")
    plt.savefig('static/graph.png')
    plt.close(fig)

@app.route('/')
def inicio():
    global shortest_path_info
    
    # Dibuja el grafo si existe
    if G:
        path_total = shortest_path_info.get('path')
        dibujar_grafo(path=path_total)

    existe_imagen = G is not None
    marca_de_tiempo = int(time.time()) if existe_imagen else 0
    url_imagen = f'static/graph.png?t={marca_de_tiempo}' if existe_imagen else None
    
    return render_template('index.html', 
                           url_imagen=url_imagen,
                           path_info=shortest_path_info,
                           nodos=list(G.nodes()) if G else [])

@app.route('/cargar_grafo', methods=['POST'])
def cargar_grafo():
    global shortest_path_info
    crear_grafo_de_ejemplo()
    shortest_path_info = {}  # Limpiar la información de la ruta anterior
    dibujar_grafo()
    return redirect(url_for('inicio'))

@app.route('/calcular_ruta', methods=['POST'])
def calcular_ruta():
    global shortest_path_info
    
    inicio_nodo = request.form.get('inicio').upper()
    intermedio_nodo = request.form.get('intermedio').upper()
    fin_nodo = request.form.get('fin').upper()

    if not G:
        shortest_path_info = {'error': 'El grafo no está cargado.'}
        return redirect(url_for('inicio'))

    nodos_validos = all(n in G.nodes for n in [inicio_nodo, intermedio_nodo, fin_nodo])
    if not nodos_validos:
        shortest_path_info = {'error': 'Uno o más nodos no existen en el grafo.'}
        return redirect(url_for('inicio'))

    # Calcular ruta de inicio a intermedio
    dist1, prev1 = dijkstra(G, inicio_nodo)
    path1 = obtener_camino(prev1, inicio_nodo, intermedio_nodo)
    
    # Calcular ruta de intermedio a fin
    dist2, prev2 = dijkstra(G, intermedio_nodo)
    path2 = obtener_camino(prev2, intermedio_nodo, fin_nodo)

    if path1 and path2:
        # Combinar caminos y distancia
        distancia_total = dist1[intermedio_nodo] + dist2[fin_nodo]
        # El nodo intermedio se repite, lo eliminamos una vez
        path_total = path1 + path2[1:]
        
        shortest_path_info = {
            'path': path_total,
            'distance': distancia_total,
            'start': inicio_nodo,
            'intermediate': intermedio_nodo,
            'end': fin_nodo
        }
    else:
        shortest_path_info = {'error': 'No se pudo encontrar una ruta que conecte los tres puntos.'}

    return redirect(url_for('inicio'))

@app.route('/borrar_grafo', methods=['POST'])
def borrar_grafo():
    global G, pos, shortest_path_info
    G = None
    pos = None
    shortest_path_info = {}
    if os.path.exists('static/graph.png'):
        os.remove('static/graph.png')
    return redirect(url_for('inicio'))

if __name__ == '__main__':
    app.run(debug=True)