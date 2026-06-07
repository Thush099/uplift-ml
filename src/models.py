import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

class UpliftModel:
    def __init__(self, model_type: str):
        self.model_type = model_type

    def train(self, data):
        if self.model_type == 'S-learner':
            X = data.drop(['outcome', 'treatment'], axis=1)
            y = data['outcome']
            self.model = RandomForestRegressor()
            self.model.fit(X, y)
        elif self.model_type == 'T-learner':
            X = data.drop(['outcome', 'treatment'], axis=1)
            y = data['outcome']
            self.model = RandomForestRegressor()
            self.model.fit(X, y)
        elif self.model_type == 'X-learner':
            X = data.drop(['outcome', 'treatment'], axis=1)
            y = data['outcome']
            self.model = RandomForestRegressor()
            self.model.fit(X, y)

    def evaluate(self, data):
        X = data.drop(['outcome', 'treatment'], axis=1)
        y = data['outcome']
        treatment_effect = self.model.predict(X)
        return treatment_effect.mean()
