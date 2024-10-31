#CSCI 3330 Project 3
#Logan Flora
#Joe Thomas
#Jack Chiolino

#If Error
#Use terminal to install: type "pip install networkx[default]"
import networkx as nx
import matplotlib.pyplot as plt

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

#This Function does the logic for Question 1
def unDirGraph():
    #Implement Graph for Questions (The Picture)
    G = nx.Graph()

    # Example nodes and edges (Adding just as a template for now, will replace with actual nodes later)
    G.add_edges_from([
        ('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'E'), ('D', 'F'),
        ('E', 'F'), ('F', 'G'), ('G', 'H'), ('H', 'I'), ('I', 'A')
    ])

    # Convert the NetworkX graph to a dictionary of adjacency lists for DFS and BFS
    graph_dict = {node: set(G.neighbors(node)) for node in G.nodes()}

    #Code for Part a (Implement DFS and BFS to check
    #If they find all connected components of undir graph)
    #DFS Code in Chapter 3, BFS in Chapter 4 

    start_node = 'A'  # Example starting node, replace as needed
    dfs_result = dfs_iterative(graph_dict, start_node)
    bfs_result = bfs_path(graph_dict, start_node, 'I')  # Example path from A to I

    # Output results
    print("DFS visited nodes:", dfs_result)
    print("BFS path from", start_node, "to I:", bfs_result)

    #Code for Part b (Determine if either can find path between two given nodes)
    #Using Previously Implemented BFS and DFS
    #Code for DFS Paths in Chapter 3, DFS in Chapter 4

    u, v = 'A', 'I'  # Example nodes, will replace with actual nodes later
    dfs_path = dfs_iterative(graph_dict, u)
    bfs_path_result = bfs_path(graph_dict, u, v)
    
    print("Path found by DFS:", dfs_path)
    print("Path found by BFS:", bfs_path_result)

    #Code for Part c (Provided u and v have a path b/w (found from above)
    #Determine if the path is always the same for DFS and BFS)
    #Prob using same code as above multiple times or sm (May be no code)

    same_path = dfs_path == bfs_path_result
    print("Do DFS and BFS find the same path?", same_path)

    return 0

#This Function does the logic for Question 2
def dirDigraph():
    #Implement Digraph for Questions (The Picture)
    DiG = nx.DiGraph() 

    #Code for Part a (Create graph and find strongly connected components)
    #Code in Chapter 3 Folder

    #Part b is on Paper (Draw Meta Graph of Strong Components)
    #Use output of Part a to draw graph (Like in Chapter 3 Slides near end)
    #(May be no code)

    #Code for Part c (Represent the drawn graph as a DAG in topological order)
    #Code also in Chapter 3 Sample Code Folder
    return 0

#This Function does the logic for Question 3
def unDirWeighted():
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
    print("Shortest Path Tree of Graph, Edges and Weights: ")
    print(spt)
    #data tells what we are trying to get from each edge 
    for u, v, weight in spt.edges(data='weight'):
       print(f"Edge: {u}--{v} has Weight of: {weight}")
    print("\nShortest distances from node:", initial_Node)
    print(visited)

    #Creates image of Sortest Path Tree (Cudos to NetworkX and plt)
    pos = nx.spring_layout(spt)
    plt.figure()
    nx.draw(
       spt, pos, edge_color='black', width=1, linewidths=1,
       node_size=500, node_color='red', alpha=0.9,
       labels={node: node for node in spt.nodes()}
    )
    plt.savefig('spt_q3.png')
    plt.show()

    #Code for Part b (Produce min Spanning tree for graph)

    #Code for Part c (Are Shortest Path Tree and Min Span Tree
    #Usually the Same?) (May be no code)

    #Coe for Part d (If - weighted edges occur, can Dijkstra's Ago find
    #Shortest path three still?) (May be no code)
    return 0


def main():
    print ("Hello World!")
    unDirWeighted()


main()
