#  Hecho por: 
# Hector Alejandro Ortega Gacria
# Registro: 21310248.
# Grupo: 6E
#----------------------------------------------------------------------------------------------
import networkx as nx
import matplotlib.pyplot as plt

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

def kruskal(graph, max_tree=False):
    """
    Implementación del algoritmo de Kruskal para encontrar el MST o el MaxST.

    :param graph: Un objeto de NetworkX que representa el grafo.
    :param max_tree: Booleano, si es True encuentra el MaxST, si es False encuentra el MST.
    :return: Lista de aristas que forman el MST/MaxST y el costo total.
    """
    # Mapear nodos a índices
    node_index = {node: idx for idx, node in enumerate(graph.nodes)}
    index_node = {idx: node for node, idx in node_index.items()}
    
    # Crear una lista de aristas y sus pesos
    edges = [(node_index[u], node_index[v], d['weight']) for u, v, d in graph.edges(data=True)]
    # Ordenar las aristas por peso, ascendente para MST, descendente para MaxST
    edges.sort(key=lambda x: x[2], reverse=max_tree)
    
    # Inicializar Union-Find
    uf = UnionFind(len(graph.nodes))
    
    # Listas para las aristas del MST/MaxST y el costo total
    tree_edges = []
    total_cost = 0
    
    # Recorrer las aristas ordenadas
    for u, v, weight in edges:
        # Encontrar los conjuntos a los que pertenecen los nodos
        if uf.find(u) != uf.find(v):
            # Si están en diferentes conjuntos, unirlos y añadir la arista al árbol
            uf.union(u, v)
            tree_edges.append((index_node[u], index_node[v], weight))
            total_cost += weight
    
    return tree_edges, total_cost

def draw_graph(graph, tree_edges, title):
    """
    Dibuja el grafo con el árbol resaltado.

    :param graph: Un objeto de NetworkX que representa el grafo.
    :param tree_edges: Lista de aristas del árbol (MST o MaxST).
    :param title: Título del gráfico.
    """
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(10, 7))
    
    # Dibujar el grafo original
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=700, font_size=15, font_weight='bold')
    
    # Resaltar las aristas del árbol
    tree = nx.Graph()
    tree.add_edges_from([(u, v) for u, v, w in tree_edges])
    nx.draw_networkx_edges(graph, pos, edgelist=tree.edges(), width=2.5, edge_color='r')
    
    # Dibujar las etiquetas de peso de las aristas
    edge_labels = {(u, v): w for u, v, w in tree_edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    
    plt.title(title)
    plt.show()

# Crear el grafo de ejemplo
G = nx.Graph()
edges = [
    ('A', 'B', 2), ('A', 'C', 3), ('A', 'D', 1),
    ('B', 'C', 1), ('B', 'E', 4),
    ('C', 'D', 5), ('C', 'E', 6),
    ('D', 'F', 7),
    ('E', 'F', 8)
]
G.add_weighted_edges_from(edges)

# Ejecutar el algoritmo de Kruskal para encontrar el MST y el MaxST
mst_edges, mst_cost = kruskal(G, max_tree=False)
maxst_edges, maxst_cost = kruskal(G, max_tree=True)

# Imprimir los resultados
print("Árbol de Expansión Mínima (MST):", mst_edges)
print("Costo Total del MST:", mst_cost)

print("Árbol de Expansión Máxima (MaxST):", maxst_edges)
print("Costo Total del MaxST:", maxst_cost)

# Dibujar los grafos
draw_graph(G, mst_edges, "Árbol de Expansión Mínima (MST)")
draw_graph(G, maxst_edges, "Árbol de Expansión Máxima (MaxST)")