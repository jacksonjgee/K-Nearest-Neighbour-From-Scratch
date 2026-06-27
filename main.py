from src.k_nearest_neighbour import KNN
from src.plot import Plot

model = KNN(k=3)

new_point = [4.7,5.6]

plot = Plot(model)

plot.plot_points(new_point)
plot.plot_decision_boundary(new_point)