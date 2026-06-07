import numpy as np


class Ensemble:

    def __init__(self, xgb_model, gnn_model):

        self.xgb = xgb_model
        self.gnn = gnn_model

    def predict(self, x_tab, x_graph):

        p1 = self.xgb.predict_proba(x_tab)[:, 1]
        p2 = self.gnn.predict(x_graph).detach().numpy()

        return 0.4 * p1 + 0.6 * p2