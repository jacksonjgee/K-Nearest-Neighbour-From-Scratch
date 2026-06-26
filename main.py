from src.k_nearest_neighbour import KNN
from src.plot import Plot

new_point = [5,5]
k = 3
model = KNN(3)
print(model.fit(new_point))
plot = Plot(model)