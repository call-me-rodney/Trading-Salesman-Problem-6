"""
Implements an undirected graph using an adjacency list.

Key points:
1. Each node is stored as a key in a dictionary (`adjacency_list`).
2. The dictionary value is a list of neighboring nodes.
3. The graph tracks the total number of nodes (`number_of_nodes`).
"""
import heapq

class Graph:
    def __init__(self):
        """
        Constructor initializes an empty adjacency list (a dictionary)
        and a node count of zero.
        """
        self.number_of_nodes = 0
        self.adjacency_list = {}

    def insert_node(self, node):
        """
        Adds a new node to the adjacency list as a key 
        with an empty list (no edges yet).
        Also increments the total number of nodes by 1.
        """
        if node not in self.adjacency_list:
            self.adjacency_list[node] = []
            self.number_of_nodes += 1
        else:
            return "Node already in graph"

    def insert_edge(self, vertex1, vertex2, distance):
        """
        Adds an undirected edge between vertex1 and vertex2.
        (Adds vertex2 to the list of vertex1's neighbors and vice versa.)

        If vertex2 is already in vertex1's adjacency list, 
        it indicates that the edge already exists.
        """
        if not any(neighbor == vertex2 for neighbor, _ in self.adjacency_list[vertex1]):
            self.adjacency_list[vertex1].append((vertex2, distance))
            self.adjacency_list[vertex2].append((vertex1, distance))
        else:
            return "Edge already exists"

    def show_connections(self):
        """
        Prints each node in the graph along with the
        nodes it is directly connected to and the distances.
        """
        for node in self.adjacency_list:
            connections = ', '.join(f"{neighbor} (distance: {distance})" for neighbor, distance in self.adjacency_list[node])
            print(f"{node} -->> {connections}")


class TSPSolver:
    def __init__(self, graph):
        self.graph = graph
        self.best_cost = float('inf')
        self.best_path = []

    def tsp(self, start_node):
        # Priority queue for exploring paths
        pq = []
        # Initial path with cost 0
        heapq.heappush(pq, (0, [start_node], set([start_node])))

        while pq:
            current_cost, current_path, visited = heapq.heappop(pq)

            # If all nodes are visited, check if returning to start is better
            if len(visited) == self.graph.number_of_nodes:
                return_cost = self.get_edge_cost(current_path[-1], start_node)
                total_cost = current_cost + return_cost
                if total_cost < self.best_cost:
                    self.best_cost = total_cost
                    self.best_path = current_path + [start_node]
                continue

            # Explore neighbors
            last_node = current_path[-1]
            for neighbor, distance in self.graph.adjacency_list[last_node]:
                if neighbor not in visited:
                    new_cost = current_cost + distance
                    # Use a bounding function to prune paths
                    if new_cost < self.best_cost:
                        new_path = current_path + [neighbor]
                        new_visited = visited | {neighbor}
                        heapq.heappush(pq, (new_cost, new_path, new_visited))

        return self.best_path, self.best_cost

    def get_edge_cost(self, node1, node2):
        for neighbor, distance in self.graph.adjacency_list[node1]:
            if neighbor == node2:
                return distance
        return float('inf')

#graph must be created properly in order for the tsp solver to work
# Example usage:
my_graph = Graph()
my_graph.insert_node(1)
my_graph.insert_node(2)
my_graph.insert_node(3)
my_graph.insert_node(4)
my_graph.insert_node(5)
my_graph.insert_node(6)
my_graph.insert_node(7)
my_graph.insert_edge(1, 2, 12)
my_graph.insert_edge(1, 3, 10)
my_graph.insert_edge(1, 7, 12)
my_graph.insert_edge(2, 3, 8)
my_graph.insert_edge(2, 4, 12)
my_graph.insert_edge(3, 4, 11)
my_graph.insert_edge(3, 5, 3)
my_graph.insert_edge(3, 7, 9)
my_graph.insert_edge(4, 5, 11)
my_graph.insert_edge(4, 6, 10)
my_graph.insert_edge(5, 6, 6)
my_graph.insert_edge(5, 7, 7)
my_graph.insert_edge(6, 7, 9)
# my_graph.show_connections()  # Displays each node's neighbors

# """
# Expected output:
# 1 -->> 2 3
# 2 -->> 1 3
# 3 -->> 1 2
# """

# print(my_graph.adjacency_list)  
# print(my_graph.number_of_nodes) 

solver = TSPSolver(my_graph)
best_path, best_cost = solver.tsp(1)
print(f"Best path: {best_path} with cost: {best_cost}")