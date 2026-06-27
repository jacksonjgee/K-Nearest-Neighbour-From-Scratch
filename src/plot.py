import numpy as np
import matplotlib.pyplot as plt

class Plot:
    def __init__(self, model):
        self.points = model.get_data()

    def plot_points(self , new_point):
        red = np.array(self.points["Red"])
        blue = np.array(self.points["Blue"])

        plt.figure(figsize=(8, 6))
        plt.scatter(red[:, 0], red[:, 1], s=75, color="red", marker="o", label="Red Points", zorder=3)
        plt.scatter(blue[:, 0], blue[:, 1], s=75, color="blue", marker="o", label="Blue Points", zorder=3)
        plt.scatter(new_point[0], new_point[1], s=150s, color="grey", marker="*", label="New Point", zorder=3)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("KNN Training Points")
        plt.legend()
        plt.grid(True, zorder = 0)
        plt.show()