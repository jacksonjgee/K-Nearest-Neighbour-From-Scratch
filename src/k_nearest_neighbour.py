import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k):
        self.k = k
        self.points = {
            "Red": [[1, 2], [3, 2], [1, 1], [4, 2], [2, 4]],
            "Blue": [[7, 8], [9, 6], [8, 6], [9, 8], [7.5, 8]]
        }

    def get_data(self):
        return self.points

    def euclidean_distance(self, p, q):
        return np.sqrt(np.sum(np.square(np.array(p) - np.array(q))))

    def fit(self, new_point):
        distances = []

        for category in self.points:
            for point in self.points[category]:
                distance = self.euclidean_distance(point, new_point)
                distances.append([distance, category])

        nearest_neighbours = sorted(distances, key=lambda x: x[0])[:self.k]

        categories = [neighbour[1] for neighbour in nearest_neighbours]

        result = Counter(categories).most_common(1)

        return 1 if result[0][0] == "Red" else 0

    def predict(self, X):
        predictions = []

        for point in X:
            prediction = self.fit(point)
            predictions.append(prediction)

        return np.array(predictions)