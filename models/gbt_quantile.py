
# models/gbt_quantile.py
# Wrapper around scikit-learn gradient boosting for multiple quantiles.

class GBTQuantileStack:
    """
    Train one boosted tree model per quantile level (tau).
    Uses HistGradientBoostingRegressor if available, else GradientBoostingRegressor.
    Adds early-stopping support to speed up training.
    """
    def __init__(self, taus, n_estimators=600, learning_rate=0.05, max_depth=3,
                 min_samples_leaf=20, random_state=42,
                 early_stopping=True, n_iter_no_change=30, validation_fraction=0.1):
        self.taus = sorted(list(taus))
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth) if max_depth is not None else None
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = int(random_state)
        self.early_stopping = bool(early_stopping)
        self.n_iter_no_change = int(n_iter_no_change)
        self.validation_fraction = float(validation_fraction)
        self.models = []
        self._impl = None  # 'hist' or 'gb'

        try:
            from sklearn.ensemble import HistGradientBoostingRegressor  # noqa
            self._impl = 'hist'
        except Exception:
            self._impl = 'gb'

    def _new_est(self, alpha):
        if self._impl == 'hist':
            from sklearn.ensemble import HistGradientBoostingRegressor
            return HistGradientBoostingRegressor(
                loss="quantile",
                quantile=float(alpha),
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                early_stopping=self.early_stopping,
                n_iter_no_change=self.n_iter_no_change,
                validation_fraction=self.validation_fraction,
                random_state=self.random_state
            )
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                loss="quantile",
                alpha=float(alpha),
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                subsample=1.0,
                random_state=self.random_state,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change
            )

    def fit(self, X, y):
        X = [list(map(float, row)) for row in X]
        y = [float(v) for v in y]
        self.models = []
        for tau in self.taus:
            est = self._new_est(tau)
            est.fit(X, y)
            self.models.append(est)
        return self

    def predict(self, X):
        if not self.models: return []
        X = [list(map(float, row)) for row in X]
        preds_per_tau = [m.predict(X) for m in self.models]
        n = len(X); k = len(self.taus)
        out = []
        for i in range(n):
            row = [preds_per_tau[j][i] for j in range(k)]
            out.append(row)
        return out
