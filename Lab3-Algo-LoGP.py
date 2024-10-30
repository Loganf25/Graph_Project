#CSCI 3330 Project 3
#Logan Flora
#Joe Thomas
#Jack Chiolino

#If Error
#Use terminal to install: type "pip install networkx[default]"
import networkx as nx

# Iterative DFS function
def dfs_iterative(graph, start):
    visited, stack = set(), [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)
    return visited

# BFS function to find a single path between two nodes
def bfs_path(graph, start, goal):
    if start == goal:
        return [start]
    visited = {start}
    queue = [(start, [])]
    while queue:
        current, path = queue.pop(0)
        for neighbor in graph[current] - set(path):
            if neighbor == goal:
                return path + [current, neighbor]
            elif neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [current]))
    return None



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
    G = nx.Graph()

    #Code for Part a (Implement graph and Use Dijksreas algo to produce
    #Shortest path three, test with node A)

    #Code for Part b (Produce min Spanning tree for graph)

    #Code for Part c (Are Shortest Path Tree and Min Span Tree
    #Usually the Same?) (May be no code)

    #Coe for Part d (If - weighted edges occur, can Dijkstra's Ago find
    #Shortest path three still?) (May be no code)
    return 0


def main():
    print ("Hello World!")


main()
