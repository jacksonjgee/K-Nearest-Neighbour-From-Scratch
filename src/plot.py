import matplotlib.pyplot as plt

class Plot:
    def __init__(self, model):
        self.points = model.get_data()