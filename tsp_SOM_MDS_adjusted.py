import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

class Graph:
    def __init__(self):
        self.number_of_nodes = 0
        self.adjacency_list = {}

    def insert_node(self, data):
        if data not in self.adjacency_list:
            self.adjacency_list[data] = []
            self.number_of_nodes += 1

    def insert_edge(self, vertex1, vertex2, weight):
        if not any(neighbor == vertex2 for neighbor, _ in self.adjacency_list[vertex1]):
            self.adjacency_list[vertex1].append((vertex2, weight))
            self.adjacency_list[vertex2].append((vertex1, weight))

    def show_connections(self):
        for node in self.adjacency_list:
            connections = ', '.join(f"{neighbor} (weight: {weight})" for neighbor, weight in self.adjacency_list[node])
            print(f"{node} -->> {connections}")

class SOM:
    def __init__(self, num_neurons, num_iterations, learning_rate, neighborhood_radius):
        self.num_neurons = num_neurons
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.neighborhood_radius = neighborhood_radius
        self.neurons = np.random.rand(num_neurons, 2)  # Initialize neurons randomly in 2D space

    def train(self, city_coords):
        for iteration in range(self.num_iterations):
            for city in city_coords:
                winner_idx = self.find_winner(city)
                self.update_weights(city, winner_idx, iteration)

    def find_winner(self, city):
        distances = np.linalg.norm(self.neurons - city, axis=1)
        return np.argmin(distances)

    def update_weights(self, city, winner_idx, iteration):
        learning_rate = self.learning_rate * np.exp(-iteration / self.num_iterations)
        neighborhood_radius = self.neighborhood_radius * np.exp(-iteration / self.num_iterations)

        for i in range(self.num_neurons):
            distance_to_winner = np.linalg.norm(i - winner_idx)
            if distance_to_winner < neighborhood_radius:
                influence = np.exp(-distance_to_winner**2 / (2 * (neighborhood_radius**2)))
                self.neurons[i] += influence * learning_rate * (city - self.neurons[i])

    def get_tour(self):
        return np.argsort(self.neurons[:, 0])

def plot_tour(city_coords, tour):
    plt.figure(figsize=(8, 6))
    plt.scatter(city_coords[:, 0], city_coords[:, 1], c='red', label='Cities')
    plt.plot(city_coords[tour, 0], city_coords[tour, 1], 'b-', label='Tour')
    plt.scatter(city_coords[tour[0], 0], city_coords[tour[0], 1], c='green', label='Start/End')
    plt.legend()
    plt.show()

def calculate_tour_cost(tour, graph):
    total_cost = 0
    for i in range(len(tour) - 1):
        current_city = tour[i]
        next_city = tour[i + 1]
        for neighbor, weight in graph.adjacency_list[current_city]:
            if neighbor == next_city:
                total_cost += weight
                break
    start_city = tour[0]
    end_city = tour[-1]
    for neighbor, weight in graph.adjacency_list[end_city]:
        if neighbor == start_city:
            total_cost += weight
            break
    return total_cost

def filter_invalid_edges(tour, graph):
    valid_tour = [tour[0]]
    for i in range(1, len(tour)):
        current_city = valid_tour[-1]
        next_city = tour[i]
        if any(neighbor == next_city for neighbor, _ in graph.adjacency_list[current_city]):
            valid_tour.append(next_city)
        else:
            # Find the shortest path to a valid next city
            for neighbor, _ in graph.adjacency_list[current_city]:
                if neighbor not in valid_tour:
                    valid_tour.append(neighbor)
                    break
    return valid_tour

# Create the graph
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

num_nodes = my_graph.number_of_nodes
large_value = 1e6  # A large finite value to represent non-existent edges
distance_matrix = np.full((num_nodes, num_nodes), large_value)

# Fill the distance matrix with weights from the graph
for node in my_graph.adjacency_list:
    for neighbor, weight in my_graph.adjacency_list[node]:
        distance_matrix[node-1][neighbor-1] = weight
        distance_matrix[neighbor-1][node-1] = weight

# Use MDS to generate 2D coordinates
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
city_coords = mds.fit_transform(distance_matrix)

# Train the SOM
som = SOM(num_neurons=7, num_iterations=2000, learning_rate=0.1, neighborhood_radius=5)
som.train(city_coords)
preliminary_tour = som.get_tour()

# Map the tour to city identities
city_ids = list(my_graph.adjacency_list.keys())
tour_with_ids = [city_ids[i] for i in preliminary_tour]

# Ensure the tour starts with city 1
if tour_with_ids[0] != 1:
    idx = tour_with_ids.index(1)
    tour_with_ids = tour_with_ids[idx:] + tour_with_ids[:idx]

# Filter invalid edges
valid_tour = filter_invalid_edges(tour_with_ids, my_graph)

# Calculate the initial cost
initial_cost = calculate_tour_cost(valid_tour, my_graph)

print(f"Initial Tour: {valid_tour}")
print(f"Initial Tour Cost: {initial_cost}")

# Plot the tour
plot_tour(city_coords, preliminary_tour)

#adjust the tour cost calculation function to be more accurate
#ask if the adjust how the neorons are given identities for better clarity