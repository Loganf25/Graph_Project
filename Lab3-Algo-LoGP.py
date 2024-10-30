#CSCI 3330 Project 3
#Logan Flora
#Joe Thomas
#Jack Chiolino

#If Error
#Use terminal to install: type "pip install networkx[default]"
import networkx as nx

#This Function does the logic for Question 1
def unDirGraph():
    #Implement Graph for Questions (The Picture)
    G = nx.Graph()

    #Code for Part a (Implement DFS and BFS to check
    #If they find all connected components of undir graph)
    #DFS Code in Chapter 3, BFS in Chapter 4 

    #Code for Part b (Determine if either can find path between two given nodes)
    #Using Previously Implemented BFS and DFS
    #Code for DFS Paths in Chapter 3, DFS in Chapter 4

    #Code for Part c (Provided u and v have a path b/w (found from above)
    #Determine if the path is always the same for DFS and BFS)
    #Prob using same code as above multiple times or sm (May be no code)

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
