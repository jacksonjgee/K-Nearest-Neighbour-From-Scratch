import os
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

class Plot:
    def __init__(self, model):
        self.model = model
        self.points = model.get_data()

    def plot_points(self, new_point):
        red = np.array(self.points["Red"])
        blue = np.array(self.points["Blue"])

        os.makedirs("assets", exist_ok=True)

        plt.figure(figsize=(8, 6))

        plt.scatter(red[:, 0], red[:, 1], s=75, color="red", marker="o", label="Red Points", zorder=3)
        plt.scatter(blue[:, 0], blue[:, 1], s=75, color="blue", marker="o", label="Blue Points", zorder=3)
        plt.scatter(new_point[0], new_point[1], s=150, color="grey", marker="*", label="New Point", zorder=3)

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("KNN Training Points")
        plt.grid(True, zorder=0)

        plt.savefig("assets/knn-training-points-simple.png", dpi=300, bbox_inches="tight")
        plt.show()

    def plot_decision_boundary(self, new_point=None):
        red = np.array(self.points["Red"])
        blue = np.array(self.points["Blue"])

        X = np.vstack((red, blue))

        y = np.array(
            [1] * len(red) +
            [0] * len(blue)
        )

        os.makedirs("assets", exist_ok=True)

        plt.figure(figsize=(8, 6))

        plot_decision_regions(
            X=X,
            y=y,
            clf=self.model,
            legend=0
        )

        plt.scatter(
            red[:, 0],
            red[:, 1],
            s=75,
            color="red",
            marker="o",
            edgecolor="black",
            zorder=3
        )

        plt.scatter(
            blue[:, 0],
            blue[:, 1],
            s=75,
            color="blue",
            marker="o",
            edgecolor="black",
            zorder=3
        )

        if new_point is not None:
            new_point_plot = plt.scatter(
                new_point[0],
                new_point[1],
                s=150,
                color="grey",
                marker="*",
                edgecolor="black",
                zorder=4
            )

            plt.legend(
                handles=[new_point_plot],
                labels=["New Point"]
            )

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("KNN Decision Boundary: k = 3")
        plt.grid(True, alpha=0.3, zorder=0)

        plt.savefig("assets/knn-decision-boundary-k3.png", dpi=300, bbox_inches="tight")
        plt.show()