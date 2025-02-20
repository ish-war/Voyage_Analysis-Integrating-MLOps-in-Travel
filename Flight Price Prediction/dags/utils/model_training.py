from sklearn.ensemble import RandomForestRegressor

class RandomForestModel:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def random_forest(self):
        model = RandomForestRegressor()
        model.fit(self.X, self.Y)
        return model
