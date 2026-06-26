from src.k_nearest_neighbour import KNN

new_point = [5,5]
k = 3
model = KNN(3)
print(model.fit(new_point))