import os
import time
from flask import Flask, render_template, request, redirect, url_for
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import heapq

app = Flask(__name__)

# Variables globales para almacenar el grafo y su estado
grafo = None
posiciones = None
info_ruta_mas_corta = {}

def crear_grafo_de_ejemplo():
    """Crea un grafo de ejemplo no dirigido y ponderado."""
    global grafo, posiciones
    grafo = nx.Graph()
    nodos = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'] #, 'K', 'L']
    grafo.add_nodes_from(nodos)
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
    grafo.add_weighted_edges_from(aristas)
    # Posiciones para la visualización, 'k' aumenta la separación para una mejor visualización
    posiciones = nx.spring_layout(grafo, seed=42, k=0.8)

def dijkstra(g, inicio):
    """Algoritmo de Dijkstra para encontrar el camino más corto desde un nodo de inicio."""
    distancias = {nodo: float('infinity') for nodo in g.nodes}
    distancias[inicio] = 0
    nodos_previos = {nodo: None for nodo in g.nodes}
    cola_prioridad = [(0, inicio)]

    while cola_prioridad:
        distancia_actual, nodo_actual = heapq.heappop(cola_prioridad)

        if distancia_actual > distancias[nodo_actual]:
            continue

        for vecino, datos in g[nodo_actual].items():
            peso = datos['weight']
            distancia = distancia_actual + peso
            if distancia < distancias[vecino]:
                distancias[vecino] = distancia
                nodos_previos[vecino] = nodo_actual
                heapq.heappush(cola_prioridad, (distancia, vecino))
    
    return distancias, nodos_previos

def obtener_camino(nodos_previos, inicio, fin):
    """Reconstruye el camino más corto desde el diccionario de nodos previos."""
    camino = []
    nodo_actual = fin
    while nodo_actual is not None:
        camino.append(nodo_actual)
        nodo_actual = nodos_previos[nodo_actual]
    camino.reverse()
    if camino and camino[0] == inicio:
        return camino
    return None

def dibujar_grafo(camino=None, color_camino='r'):
    """Dibuja el grafo y resalta un camino si se proporciona."""
    if grafo is None:
        if os.path.exists('static/graph.png'):
            os.remove('static/graph.png')
        return

    if not os.path.exists('static'):
        os.makedirs('static')

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Dibuja el grafo base
    nx.draw(grafo, posiciones, with_labels=True, node_color='skyblue', node_size=2000, font_size=15, font_weight='bold', ax=ax)
    etiquetas = nx.get_edge_attributes(grafo, 'weight')
    nx.draw_networkx_edge_labels(grafo, posiciones, edge_labels=etiquetas, ax=ax)

    # Si hay un camino para resaltar
    if camino:
        aristas_camino = list(zip(camino, camino[1:]))
        nx.draw_networkx_nodes(grafo, posiciones, nodelist=camino, node_color=color_camino, node_size=2000, ax=ax)
        nx.draw_networkx_edges(grafo, posiciones, edgelist=aristas_camino, edge_color=color_camino, width=2.5, ax=ax)

    plt.title("Grafo Ponderado No Dirigido")
    plt.savefig('static/graph.png')
    plt.close(fig)

@app.route('/')
def inicio():
    global info_ruta_mas_corta
    
    # Dibuja el grafo si existe
    if grafo:
        camino_total = info_ruta_mas_corta.get('camino')
        dibujar_grafo(camino=camino_total)

    existe_imagen = grafo is not None
    marca_de_tiempo = int(time.time()) if existe_imagen else 0
    url_imagen = f'static/graph.png?t={marca_de_tiempo}' if existe_imagen else None
    
    return render_template('index.html', 
                           url_imagen=url_imagen,
                           path_info=info_ruta_mas_corta,
                           nodos=list(grafo.nodes()) if grafo else [])

@app.route('/cargar_grafo', methods=['POST'])
def cargar_grafo():
    global info_ruta_mas_corta
    crear_grafo_de_ejemplo()
    info_ruta_mas_corta = {}  # Limpiar la información de la ruta anterior
    dibujar_grafo()
    return redirect(url_for('inicio'))

@app.route('/calcular_ruta', methods=['POST'])
def calcular_ruta():
    global info_ruta_mas_corta
    
    inicio_nodo = request.form.get('inicio').upper()
    intermedio_nodo = request.form.get('intermedio').upper()
    fin_nodo = request.form.get('fin').upper()

    if not grafo:
        info_ruta_mas_corta = {'error': 'El grafo no está cargado.'}
        return redirect(url_for('inicio'))

    nodos_validos = all(n in grafo.nodes for n in [inicio_nodo, intermedio_nodo, fin_nodo])
    if not nodos_validos:
        info_ruta_mas_corta = {'error': 'Uno o más nodos no existen en el grafo.'}
        return redirect(url_for('inicio'))

    # Calcular ruta de inicio a intermedio
    distancias1, previos1 = dijkstra(grafo, inicio_nodo)
    camino1 = obtener_camino(previos1, inicio_nodo, intermedio_nodo)
    
    # Calcular ruta de intermedio a fin
    distancias2, previos2 = dijkstra(grafo, intermedio_nodo)
    camino2 = obtener_camino(previos2, intermedio_nodo, fin_nodo)

    if camino1 and camino2:
        # Combinar caminos y distancia
        distancia_total = distancias1[intermedio_nodo] + distancias2[fin_nodo]
        # El nodo intermedio se repite, lo eliminamos una vez
        camino_total = camino1 + camino2[1:]
        
        info_ruta_mas_corta = {
            'camino': camino_total,
            'distancia': distancia_total,
            'inicio': inicio_nodo,
            'intermedio': intermedio_nodo,
            'fin': fin_nodo
        }
    else:
        info_ruta_mas_corta = {'error': 'No se pudo encontrar una ruta que conecte los tres puntos.'}

    return redirect(url_for('inicio'))

@app.route('/borrar_grafo', methods=['POST'])
def borrar_grafo():
    global grafo, posiciones, info_ruta_mas_corta
    grafo = None
    posiciones = None
    info_ruta_mas_corta = {}
    if os.path.exists('static/graph.png'):
        os.remove('static/graph.png')
    return redirect(url_for('inicio'))

if __name__ == '__main__':
    app.run(debug=True)
