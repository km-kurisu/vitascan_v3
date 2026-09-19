class BaselineModelsEvaluator:
    def predict(self, normalized_biomarkers):
        return {
            "iron": {"gat": 0.75, "logistic_regression": 0.68, "random_forest": 0.72, "xgboost": 0.74}
        }
