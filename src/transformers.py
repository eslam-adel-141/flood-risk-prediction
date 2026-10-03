import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FloodFeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain-driven composite features for flood risk assessment."""

    ENGINEERED_NAMES = [
        "TotalRiskScore", "MeanRiskScore", "StdRiskScore",
        "HumanFactorScore", "InfrastructureScore", "ClimateNaturalScore",
        "GovernanceScore", "DrainageEfficiency", "MonsoonDeforestInteraction",
    ]

    def __init__(self, feature_names=None):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_eng = np.asarray(X, dtype=float).copy()
        i = {name: k for k, name in enumerate(self.feature_names)}

        human = (X_eng[:, i["Deforestation"]] + X_eng[:, i["Urbanization"]] +
                 X_eng[:, i["Encroachments"]] + X_eng[:, i["AgriculturalPractices"]] +
                 X_eng[:, i["PopulationScore"]] + X_eng[:, i["WetlandLoss"]])
        infra = (X_eng[:, i["TopographyDrainage"]] + X_eng[:, i["RiverManagement"]] +
                 X_eng[:, i["DamsQuality"]] + X_eng[:, i["DrainageSystems"]] +
                 X_eng[:, i["DeterioratingInfrastructure"]])
        climate = (X_eng[:, i["MonsoonIntensity"]] + X_eng[:, i["ClimateChange"]] +
                   X_eng[:, i["CoastalVulnerability"]] + X_eng[:, i["Landslides"]] +
                   X_eng[:, i["Watersheds"]] + X_eng[:, i["Siltation"]])
        governance = (X_eng[:, i["IneffectiveDisasterPreparedness"]] +
                      X_eng[:, i["InadequatePlanning"]] + X_eng[:, i["PoliticalFactors"]])

        total_score = X_eng.sum(axis=1)
        mean_score = X_eng.mean(axis=1)
        std_score = X_eng.std(axis=1)

        drainage_eff = X_eng[:, i["DrainageSystems"]] / (X_eng[:, i["MonsoonIntensity"]] + 1e-8)
        monsoon_deforest = X_eng[:, i["MonsoonIntensity"]] * X_eng[:, i["Deforestation"]]

        return np.column_stack([
            X_eng, total_score, mean_score, std_score,
            human, infra, climate, governance,
            drainage_eff, monsoon_deforest,
        ])

    def get_feature_names(self):
        return list(self.feature_names) + self.ENGINEERED_NAMES


class OutlierHandler(BaseEstimator, TransformerMixin):
    """Clip outliers to IQR-based bounds instead of removing rows."""

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        self.lower_bounds_, self.upper_bounds_ = [], []
        for k in range(X.shape[1]):
            q1, q3 = np.percentile(X[:, k], [25, 75])
            iqr = q3 - q1
            self.lower_bounds_.append(q1 - self.factor * iqr)
            self.upper_bounds_.append(q3 + self.factor * iqr)
        return self

    def transform(self, X):
        X_t = X.copy()
        for k in range(X.shape[1]):
            X_t[:, k] = np.clip(X_t[:, k], self.lower_bounds_[k], self.upper_bounds_[k])
        return X_t
