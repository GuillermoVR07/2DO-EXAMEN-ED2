

import os
import time
from flask import Flask, render_template, request, redirect, url_for
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Clase Nodo ---
class Nodo:
    def __init__(self, valor):
        self.valor = valor
        self.hijo_izquierdo = None
        self.hijo_derecho = None

    def obtener_hijo_izquierdo(self):
        return self.hijo_izquierdo

    def obtener_hijo_derecho(self):
        return self.hijo_derecho
    
    def establecer_hijo_izquierdo(self, nodo):
        self.hijo_izquierdo = nodo

    def establecer_hijo_derecho(self, nodo):
        self.hijo_derecho = nodo

    def obtener_valor(self):
        return self.valor
    
    def establecer_valor(self, valor):
        self.valor = valor

    def es_hoja(self):
        return self.hijo_izquierdo is None and self.hijo_derecho is None

# --- Clase ArbolBinario ---
class ArbolBinario:
    def __init__(self):
        self.raiz = None

    def insertar(self, valor):
        self.raiz = self._insertar_recursivo(self.raiz, valor)

    def _insertar_recursivo(self, raiz_aux, valor):
        if raiz_aux is None:
            return Nodo(valor)
        else:
            if valor < raiz_aux.obtener_valor():
                raiz_aux.establecer_hijo_izquierdo(self._insertar_recursivo(raiz_aux.obtener_hijo_izquierdo(), valor))
            else:
                raiz_aux.establecer_hijo_derecho(self._insertar_recursivo(raiz_aux.obtener_hijo_derecho(), valor))
            return raiz_aux
    
    def verificar_hoja(self, raiz_aux):
        return raiz_aux.obtener_hijo_izquierdo() is None and raiz_aux.obtener_hijo_derecho() is None

    def altura(self):
        return self._altura_recursivo(self.raiz)

    def _altura_recursivo(self, raiz_aux):
        if raiz_aux is None:
            return -1
        
        altura_izquierda = self._altura_recursivo(raiz_aux.obtener_hijo_izquierdo())
        altura_derecha = self._altura_recursivo(raiz_aux.obtener_hijo_derecho())
        
        return 1 + max(altura_izquierda, altura_derecha)

    def contar_filas_espejo(self, altura_maxima_usuario):
        if self.raiz is None:
            return 0
        
        altura_real_arbol = self.altura()
        limite_verificacion = min(altura_maxima_usuario, altura_real_arbol)

        contador = 0
        for nivel in range(1, limite_verificacion + 1):
            if self.son_nodos_en_el_nivel_reflejado(self.raiz.obtener_hijo_izquierdo(), self.raiz.obtener_hijo_derecho(), nivel - 1):
                contador += 1
            else:
                break
        return contador

    def son_nodos_en_el_nivel_reflejado(self, nodo1, nodo2, nivel):
        if nivel == 0:
            return (nodo1 is None) == (nodo2 is None)

        if (nodo1 is None) != (nodo2 is None):
            return False
        
        if nodo1 is None:
            return True

        return (self.son_nodos_en_el_nivel_reflejado(nodo1.obtener_hijo_izquierdo(), nodo2.obtener_hijo_derecho(), nivel - 1) and
                self.son_nodos_en_el_nivel_reflejado(nodo1.obtener_hijo_derecho(), nodo2.obtener_hijo_izquierdo(), nivel - 1))

    def contar_nodos_espejo(self, altura):
        if self.raiz is None:
            return 0
        
        altura_real = self.altura()
        altura_a_verificar = min(altura, altura_real)

        return self._contar_nodos_espejo_recursivo(self.raiz.obtener_hijo_izquierdo(), self.raiz.obtener_hijo_derecho(), 1, altura_a_verificar)

    def _contar_nodos_espejo_recursivo(self, nodo1, nodo2, nivel_actual, altura_maxima):
        if nivel_actual > altura_maxima:
            return 0

        if nodo1 is None or nodo2 is None:
            return 0

        contador = 2
        
        contador += self._contar_nodos_espejo_recursivo(
            nodo1.obtener_hijo_izquierdo(), nodo2.obtener_hijo_derecho(), nivel_actual + 1, altura_maxima
        )
        contador += self._contar_nodos_espejo_recursivo(
            nodo1.obtener_hijo_derecho(), nodo2.obtener_hijo_izquierdo(), nivel_actual + 1, altura_maxima
        )
        
        return contador

# --- Aplicación Flask ---
app = Flask(__name__)
arbol = ArbolBinario()

def obtener_nodos_y_aristas(nodo, nodos=None, aristas=None):
    if nodos is None:
        nodos = []
    if aristas is None:
        aristas = []
    
    if nodo is not None:
        nodos.append(nodo)
        if nodo.obtener_hijo_izquierdo() is not None:
            aristas.append((nodo, nodo.obtener_hijo_izquierdo()))
            obtener_nodos_y_aristas(nodo.obtener_hijo_izquierdo(), nodos, aristas)
        if nodo.obtener_hijo_derecho() is not None:
            aristas.append((nodo, nodo.obtener_hijo_derecho()))
            obtener_nodos_y_aristas(nodo.obtener_hijo_derecho(), nodos, aristas)
    return nodos, aristas

def obtener_posiciones(raiz):
    posiciones = {}
    next_x = [0]

    def _obtener_posiciones_inorden(nodo, nivel=0):
        if nodo is None:
            return
        
        _obtener_posiciones_inorden(nodo.obtener_hijo_izquierdo(), nivel + 1)
        
        posiciones[nodo.obtener_valor()] = (next_x[0], -nivel)
        next_x[0] += 1.2

        _obtener_posiciones_inorden(nodo.obtener_hijo_derecho(), nivel + 1)

    _obtener_posiciones_inorden(raiz)
    return posiciones

def dibujar_arbol(raiz):
    if raiz is None:
        if os.path.exists('static/tree.png'):
            os.remove('static/tree.png')
        return

    if not os.path.exists('static'):
        os.makedirs('static')

    posiciones = obtener_posiciones(raiz)
    nodos, aristas = obtener_nodos_y_aristas(raiz)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')

    for nodo_padre, nodo_hijo in aristas:
        pos_padre = posiciones.get(nodo_padre.obtener_valor())
        pos_hijo = posiciones.get(nodo_hijo.obtener_valor())
        if pos_padre and pos_hijo:
            ax.plot([pos_padre[0], pos_hijo[0]], [pos_padre[1], pos_hijo[1]], 'k-', lw=1.5, zorder=1)

    for valor_nodo, (x, y) in posiciones.items():
        circulo = plt.Circle((x, y), 0.3, color='skyblue', zorder=2)
        ax.add_patch(circulo)
        ax.text(x, y, str(valor_nodo), ha='center', va='center', zorder=3, fontsize=28)
    
    ax.relim()
    ax.autoscale_view()

    plt.title("Árbol Binario de Búsqueda")
    plt.savefig('static/tree.png')
    plt.close(fig)

@app.route('/')
def inicio():
    resultado_filas = request.args.get('resultado_espejo', None)
    resultado_nodos_espejo = request.args.get('resultado_nodos_espejo', None)
    
    dibujar_arbol(arbol.raiz)
    existe_imagen = arbol.raiz is not None
    
    marca_de_tiempo = int(time.time()) if existe_imagen else 0
    url_imagen = f'static/tree.png?t={marca_de_tiempo}' if existe_imagen else None
    
    return render_template('index.html', 
                           url_imagen=url_imagen, 
                           resultado_espejo=resultado_filas,
                           resultado_nodos_espejo=resultado_nodos_espejo)

@app.route('/insertar', methods=['POST'])
def insertar():
    valor = request.form.get('valor')
    if valor:
        try:
            valor_int = int(valor)
            arbol.insertar(valor_int)
        except ValueError:
            pass
    return redirect(url_for('inicio'))

@app.route('/contar_espejo', methods=['POST'])
def contar_espejo():
    resultado = None
    try:
        altura = int(request.form['altura_maxima'])
        resultado = arbol.contar_filas_espejo(altura)
    except (ValueError, TypeError):
        resultado = "Error: La altura debe ser un número entero."
    return redirect(url_for('inicio', resultado_espejo=resultado))

@app.route('/contar_columnas_espejo', methods=['POST'])
def contar_columnas_espejo():
    resultado_nodos = None
    try:
        altura = int(request.form['altura_columnas'])
        resultado_nodos = arbol.contar_nodos_espejo(altura)
    except (ValueError, TypeError):
        resultado_nodos = "Error: La altura debe ser un número entero."
    return redirect(url_for('inicio', resultado_nodos_espejo=resultado_nodos))

@app.route('/borrar_arbol', methods=['POST'])
def borrar_arbol():
    global arbol
    arbol = ArbolBinario()
    if os.path.exists('static/tree.png'):
        os.remove('static/tree.png')
    return redirect(url_for('inicio'))

if __name__ == '__main__':
    app.run(debug=True)
