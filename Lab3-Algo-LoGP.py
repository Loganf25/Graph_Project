#CSCI 3330 Project 3
#Logan Flora
#Joe Thomas
#Jack Chiolino

#If Error
#Use terminal to install: type "pip install networkx[default]"
import networkx as nx
import matplotlib.pyplot as plt
from itertools import groupby
from collections import defaultdict
 
# Iterative DFS function
def dfs_iterative(graph, start):
    visited, stack = set(), [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(set(graph.neighbors(node)) - visited)
    return visited

# BFS function to find a single path between two nodes
def bfs_path(graph, start, goal):
    if start == goal:
        return [start]
    visited = {start}
    queue = [(start, [])]
    while queue:
        current, path = queue.pop(0)
        for neighbor in graph.neighbors(current):
            if neighbor == goal:
                return path + [neighbor]
            elif neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None


#Dijkstra Algorithm (For Shortest Path)
def dijkstra(graph, initial):
  visited = {initial: 0}
  path = {}

  nodes = set(graph.nodes)

  while nodes: 
    min_node = None
    for node in nodes:
      if node in visited:
        if min_node is None:
          min_node = node
        elif visited[node] < visited[min_node]:
          min_node = node

    if min_node is None:
      break

    nodes.remove(min_node)
    current_weight = visited[min_node]
    
    for neighbor in graph.neighbors(min_node):
      weight = current_weight + graph[min_node][neighbor]['weight']
      if neighbor not in visited or weight < visited[neighbor]:
        visited[neighbor] = weight
        path[neighbor] = min_node

  return visited, path

#Kruskall Algo for MST
def Kruskal(graph):
    #Helper functions 
    def find(parent, i):
        if parent[i] == i:
            return i
        return find(parent, parent[i])
    def union(parent, rank, x, y):
        xroot = find(parent, x)
        yroot = find(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank [yroot]:
            parent [yroot] = xroot
        else:
            parent [yroot] = xroot
            rank[xroot] += 1
 
    result = nx.Graph() #This will store the resultant MST
 
    i = 0 # An index variable, used for sorted edges
    e = 0 # An index variable, used for result[]
 
    #Step 1:  Sort all the edges in non-decreasing order of their
    # weight.  If we are not allowed to change the given graph, we
    # can create a copy of graph
    sorted_graph =  sorted(graph.edges(data=True), key = lambda t: t[2].get('weight', 1))
    #print self.graph
 
    parent = {} ; rank = {}
 
    # Create V subsets with single elements
    for node in graph.nodes():
        parent[node] = node
        rank[node] = 0 
     
    # Number of edges to be taken is equal to V-1
    while e < graph.number_of_nodes() - 1:
 
        # Step 2: Pick the smallest edge and increment the index
        # for next iteration
        u,v,w =  sorted_graph[i]
        i = i + 1
        x = find(parent, u)
        y = find(parent ,v)
 
        # If including this edge does't cause cycle, include it
        # in result and increment the index of result for next edge
        if x != y:
            e = e + 1  
            result.add_edge(u, v, weight=w['weight'])
            union(parent, rank, x, y)          
        # Else discard the edge v5
    return result

# Extra stuff for Scc function and scc function for question 2.a.

class Tracker(object):
    """Keeps track of the current time, current source, component leader,
    finish time of each node and the explored nodes.
    
    'self.leader' is informs of {node: leader, ...}."""

    def __init__(self):
        self.current_time = 0
        self.current_source = None
        self.leader = {}
        self.finish_time = {}
        self.explored = set()

def dfs(graph, node, tracker):
    """Inner loop explores all nodes in a SCC. Graph represented as a dict,
    {tail: [head_list], ...}. Depth first search runs recursively and keeps
    track of the parameters"""

    tracker.explored.add(node)
    tracker.leader[node] = tracker.current_source
    for head in graph.neighbors(node):
        if head not in tracker.explored:
            dfs(graph, head, tracker)
    tracker.current_time += 1
    tracker.finish_time[node] = tracker.current_time


def dfs_loop(graph_dict, nodes, tracker):
    """Outer loop checks out all SCCs. Current source node changes when one
    SCC inner loop finishes."""

    for node in nodes:
        if node not in tracker.explored:
            tracker.current_source = node
            dfs(graph_dict, node, tracker)

def graph_reverse(graph):
    """Given a directed graph in forms of {tail:[head_list], ...}, compute
    a reversed directed graph, in which every edge changes direction."""

    reversed_graph = graph.reverse(copy=True)
    return reversed_graph

# Scc for question 2.a to find strongly connected components of the meta graph.
def scc(graph):
    out = defaultdict(list)
    tracker1 = Tracker()
    tracker2 = Tracker()
    nodes = graph.nodes()
    reversed_graph = graph_reverse(graph)
    nodes = sorted(list(nodes), reverse=True)
    dfs_loop(reversed_graph, nodes, tracker1)
    sorted_nodes = sorted(tracker1.finish_time,
                          key=tracker1.finish_time.get, reverse=True)
    dfs_loop(graph, sorted_nodes, tracker2)
    for lead, vertex in groupby(sorted(tracker2.leader, key=tracker2.leader.get),
                                key=tracker2.leader.get):
        out[lead] = list(vertex)
    return out

# Create graph for question 2.b
def create_meta_graph(Dir_Edges, post_scc):
    meta_graph = nx.DiGraph()
    scc_nodes = {}

    # Iterate through nodes in the SCC graph
    for component in nx.strongly_connected_components(Dir_Edges):
        leader = min(component)  # Choose a representative node 
        scc_nodes[leader] = component  
            
    for u in Dir_Edges.nodes():
        for v in Dir_Edges.neighbors(u):
            leader_u = next((leader for leader, comp in scc_nodes.items() if u in comp), None)
            leader_v = next((leader for leader, comp in scc_nodes.items() if v in comp), None)
            if leader_u != leader_v:
                meta_graph.add_edge(leader_u, leader_v)
    return meta_graph

# Dfs for question 2.c to represent the meta graph as a DAG, linearized in its topological order.
def dfs_tpl_order(graph,start,path,n_holder):
    path = path + [start]
    for edge in graph[start]: 
        if edge not in path:
            path = dfs_tpl_order(graph, edge, path, n_holder)
    print (n_holder[0], start)
    n_holder[0] -= 1
    return path

#This Function does the logic for Question 1
def unDirGraph():
    print("\nQuestion 1 Results")
    #Implement Graph for Questions (The Picture)
    G = nx.Graph()


    G.add_edges_from([
        ('A', 'B'), ('A', 'E'), ('A', 'F'), ('B', 'C'), ('B', 'F'), ('C', 'D'), ('C', 'G'), ('D', 'H'), ('E', 'I'), ('F', 'G'), ('F', 'J'), ('G', 'H'), ('G', 'K'),  ('H', 'L'),  ('I', 'J'),  ('I', 'M'),  ('J', 'K'),  ('J', 'N'),  ('K', 'L'),  ('K', 'O'),  ('M', 'N'),  ('N', 'O'),  ('O', 'P') 
    ])

    # Convert the NetworkX graph to a dictionary of adjacency lists for DFS and BFS
    graph_dict = {node: set(G.neighbors(node)) for node in G.nodes()}

    #Code for Part a (Implement DFS and BFS to check
    #If they find all connected components of undir graph)
    #DFS Code in Chapter 3, BFS in Chapter 4 

    start_node = 'A'  # Example starting node, replace as needed
    dfs_result = dfs_iterative(G, start_node)
    bfs_result = bfs_path(G, start_node, 'I')  # Example path from A to I

    # Output results
    print("Part A Results")
    print("DFS visited nodes:", dfs_result)
    print("BFS path from", start_node, "to I:", bfs_result)

    #Code for Part b (Determine if either can find path between two given nodes)
    #Using Previously Implemented BFS and DFS
    #Code for DFS Paths in Chapter 3, DFS in Chapter 4

    u, v = 'A', 'I'  # Example nodes, will replace with actual nodes later
    dfs_path = dfs_iterative(G, u)
    bfs_path_result = bfs_path(G, u, v)
    
    print("\n Part B Results")
    print("Path found by DFS:", dfs_path)
    print("Path found by BFS:", bfs_path_result)

    #Code for Part c (Provided u and v have a path b/w (found from above)
    #Determine if the path is always the same for DFS and BFS)
    #Prob using same code as above multiple times or sm (May be no code)

    same_path = dfs_path == bfs_path_result
    print("\nPart C Results")
    print("Do DFS and BFS find the same path?", same_path)

    return 0

#This Function does the logic for Question 2
def dirDigraph():
    print('Question 2 Results')
    #Implement Digraph for Questions (The Picture)
    Dir_Edges = nx.DiGraph() 
    
    #Code for Part a (Create graph and find strongly connected components)
    #Code in Chapter 3 Folder
    # Set graph as depicted in instructions (A = 1, B = 2, etc.).
    Dir_Edges.add_edges_from([
       (1, 3), (2, 1), (3, 2), (3, 5), (4, 1), (4, 2), (4, 12),
        (5, 6), (5, 8), (6, 7), (6, 8), (6, 10), (7, 10), (8, 9),
        (8, 10), (9, 5), (9, 11), (10, 9), (10, 11), (11, 12)
    ])

    # Run scc.py on graph
    post_scc = scc(Dir_Edges)
    post_scct2 = nx.strongly_connected_components(Dir_Edges)
    print('Strongly connected components are:')

    for key in post_scc:
        print(post_scc[key])
    print('\n')
    for key in post_scct2:
        print(list[key])
    #Part b is on Paper (Draw Meta Graph of Strong Components)
    #Use output of Part a to draw graph (Like in Chapter 3 Slides near end)
    #(May be no code)
    meta_graph = nx.DiGraph()
    meta_graph = create_meta_graph(Dir_Edges, post_scc)

    #Create Image of MST
    pos1 = nx.spring_layout(meta_graph)
    plt.figure()
    nx.draw(
       meta_graph, pos1, edge_color='black', width=1, linewidths=1,
       node_size=500, node_color='red', alpha=0.9,
       labels={node: node for node in meta_graph.nodes()}
    )
    plt.savefig('spt_q2p2.png')
    plt.show()

    #Code for Part c (Represent the drawn graph as a DAG in topological order)
    #Code also in Chapter 3 Sample Code Folder
    n_holder = [len(Dir_Edges)]
    print('the meta graph as a DAG and linearized in its topological order is:')
    #c = dfs_tpl_order(meta_graph, 1, [],n_holder)
    #print(c)
    #print('\n')
    c1 = list(nx.topological_sort(meta_graph))
    print(c1)

    return 0

#This Function does the logic for Question 3
def unDirWeighted():
    print("Question 3 Results")
    #Implementing Undirected Weighted Graph
    #Create an Undirected Graph
    G = nx.Graph()
    
    #Add All nodes into the graph
    nodes = ['A', 'B', 'C', 'E', 'F', 'G', 'H', 'I']
    G.add_nodes_from(nodes)

    #Create the Weighted Edges in Between Each Node
    weighted_edges = [('A', 'B', {"weight": 22}), ('A', 'C', {"weight": 9}), ('A', 'D', {"weight": 12}),
                     ('B', 'C', {"weight": 35}), ('B', 'F', {"weight": 36}), ('B', 'H', {"weight": 34}),
                     ('C', 'F', {"weight": 42}), ('C', 'E', {"weight": 65}), ('C', 'D', {"weight": 4}),
                     ('D', 'E', {"weight": 33}), ('D', 'I', {"weight": 30}), ('E', 'F', {"weight": 18}),
                     ('E', 'G', {"weight": 23}), ('F', 'G', {"weight": 39}), ('F', 'H', {"weight": 24}),
                     ('G', 'H', {"weight": 25}), ('G', 'I', {"weight": 21}), ('H', 'I', {"weight": 19})]

    G.add_edges_from(weighted_edges)

    #Code for Part a (Implement graph and Use Dijksreas algo to produce
    #Shortest path three, test with node A)
    initial_Node = 'A'
    visited, path = dijkstra(G, initial_Node)


    #Create Shortest Path Tree From Dijkstra's Algo Results
    spt = nx.Graph()
    spt.add_node(initial_Node)

    #Add all nodes from the paths into new graph
    for node, parent in path.items():
       edge_weight = G[parent][node]['weight']
       spt.add_edge(parent, node, weight = edge_weight)

    #Print Results (Answer to Question)
    print("Part A Results")
    print("Shortest Path Tree of Graph, Edges and Weights: ")
    print(spt)
    #data tells what we are trying to get from each edge 
    print(spt.edges(data=True))
    print(visited)

    #Creates image of Sortest Path Tree (Cudos to NetworkX and plt)
    pos = nx.spring_layout(spt)
    plt.figure()
    nx.draw(
       spt, pos, edge_color='black', width=1, linewidths=1,
       node_size=500, node_color='red', alpha=0.9,
       labels={node: node for node in spt.nodes()}
    )
    plt.savefig('spt_q3p1.png')
    plt.show()

    #Code for Part b (Produce min Spanning tree for graph)
    MST = nx.Graph()
    MST = Kruskal(G)

    print("\nPart B Results")
    print("Minimum Spanning Tree of Graph, Edges and Weights: ")
    print(MST)
    print(MST.edges(data=True))

    #Create Image of MST
    pos1 = nx.spring_layout(MST)
    plt.figure()
    nx.draw(
       MST, pos1, edge_color='black', width=1, linewidths=1,
       node_size=500, node_color='red', alpha=0.9,
       labels={node: node for node in MST.nodes()}
    )
    plt.savefig('spt_q3p2.png')
    plt.show()

    #Code for Part c (Are Shortest Path Tree and Min Span Tree
    #Usually the Same?) (May be no code)
    print("\nPart C Results")
    print("As seen by the graphs created above, the SPT and MST are different from one another")
    print("But, they are the same each time either is found on the given graph.")
    print("Test edges equality: ", spt.edges == MST.edges)


    #Coe for Part d (If - weighted edges occur, can Dijkstra's Ago find
    #Shortest path three still?) (May be no code)
    print('\nPart D Results')
    print("It is not possible to use Dijkstra's Algo for - Weighted Edges. Example: ")

    #Test Dijkstra with - weight
    G.add_node('J')
    print('Added Node: J')
    G.add_edge('J', 'A', weight = -5)
    visitedT, pathT = dijkstra(G, initial_Node)
    Testspt = nx.Graph()
    Testspt.add_node(initial_Node)
    for node, parent in path.items():
       edge_weight = G[parent][node]['weight']
       Testspt.add_edge(parent, node, weight = edge_weight)


    print('Added Edge with wieght of -5 and preform Dijkstra\'s Algo')
    print(Testspt.edges(data=True))
    print('As shown by the new spt, there are no edges connecting Node J, so it cannot add J to the Graph')


    #Try to create Graph with one of its edges negative 
    return 0


def main():
    #Results for Q 1
    unDirGraph()
    print("\n\n")
    #Results for Q 2
    dirDigraph()
    print('\n\n')
    #Results for Q 3 
    unDirWeighted()


main()
