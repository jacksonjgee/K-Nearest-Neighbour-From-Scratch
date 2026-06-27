import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k):
        self.k = k
        self.points = {
            "Red": [
                [1.0, 1.2], [1.5, 3.8], [1.8, 7.2],
                [2.4, 2.5], [2.7, 5.9], [3.1, 8.1],
                [3.6, 1.7], [4.0, 4.6], [4.3, 7.4],
                [5.0, 2.9], [5.4, 6.8], [5.9, 4.1],
                [6.4, 1.5], [6.8, 5.5], [7.2, 8.0],
                [7.7, 3.2], [8.1, 6.4], [8.6, 2.1],
                [9.0, 4.8], [9.3, 7.5]
            ],

            "Blue": [
                [1.2, 6.4], [1.7, 2.1], [2.1, 8.4],
                [2.6, 4.7], [3.0, 1.3], [3.4, 6.9],
                [3.9, 3.1], [4.5, 8.2], [4.8, 5.2],
                [5.2, 1.8], [5.7, 7.5], [6.1, 3.8],
                [6.6, 6.2], [7.0, 2.4], [7.5, 4.9],
                [8.0, 1.1], [8.3, 7.1], [8.8, 3.6],
                [9.1, 6.0], [9.5, 2.8]
            ]
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