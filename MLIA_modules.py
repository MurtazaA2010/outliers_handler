
class MLIA():
    def __init__(self, X_train=None, X_val=None, y_train=None, y_val=None,
                 features=None, base_model=LinearRegression, metric=mean_squared_error):
        self.X_train = X_train.copy()
        self.X_val = X_val.copy()
        self.y_train = y_train
        self.y_val = y_val
        self.features = features if features else []
        self.base_model = base_model
        self.metric = metric
        self.X_cols = [f for f in self.X_train.columns if f not in self.features]

    # --- IQR method ---
    def IQR(self):
        Q1 = self.X_train[self.features].quantile(0.25)
        Q3 = self.X_train[self.features].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        X_train_clean = self.X_train.copy()
        X_val_clean = self.X_val.copy()
        X_train_clean[self.features] = X_train_clean[self.features].clip(lower, upper, axis=1)
        X_val_clean[self.features] = X_val_clean[self.features].clip(lower, upper, axis=1)
        return X_train_clean, X_val_clean

    # --- Z-score method ---
    def z_score(self, threshold=3):
        X_train_clean = self.X_train.copy()
        X_val_clean = self.X_val.copy()
        for col in self.features:
            col_z = np.abs(stats.zscore(X_train_clean[col]))
            X_train_clean[col] = X_train_clean[col].where(col_z <= threshold, np.nan)
            mean_col = X_train_clean[col].mean()
            std_col = X_train_clean[col].std()
            X_val_clean[col] = ((X_val_clean[col] - mean_col)/std_col).where(
                np.abs((X_val_clean[col] - mean_col)/std_col) <= threshold, np.nan)
        return X_train_clean, X_val_clean

    # --- Winsorization ---
    def winsorize(self, lower_pct=0.01, upper_pct=0.99):
        X_train_clean = self.X_train.copy()
        X_val_clean = self.X_val.copy()
        for col in self.features:
            lower = self.X_train[col].quantile(lower_pct)
            upper = self.X_train[col].quantile(upper_pct)
            X_train_clean[col] = X_train_clean[col].clip(lower, upper)
            X_val_clean[col] = X_val_clean[col].clip(lower, upper)
        return X_train_clean, X_val_clean

    # --- Median Imputation ---
    def median_impute(self):
        X_train_clean = self.X_train.copy()
        X_val_clean = self.X_val.copy()
        imputer = SimpleImputer(strategy="median")
        X_train_clean[self.features] = imputer.fit_transform(X_train_clean[self.features])
        X_val_clean[self.features] = imputer.transform(X_val_clean[self.features])
        return X_train_clean, X_val_clean

    # --- Mean Imputation ---
    def mean_impute(self):
        X_train_clean = self.X_train.copy()
        X_val_clean = self.X_val.copy()
        imputer = SimpleImputer(strategy="mean")
        X_train_clean[self.features] = imputer.fit_transform(X_train_clean[self.features])
        X_val_clean[self.features] = imputer.transform(X_val_clean[self.features])
        return X_train_clean, X_val_clean

    # --- Isolation Forest ---
    def isolation_forest(self, contamination=0.05):
        iso = IsolationForest(contamination=contamination, random_state=42)
        X_train_clean = self.X_train.copy()
        X_val_clean = self.X_val.copy()
        iso.fit(X_train_clean[self.X_cols + self.features])
        outliers_train = iso.predict(X_train_clean[self.X_cols + self.features]) == -1
        outliers_val = iso.predict(X_val_clean[self.X_cols + self.features]) == -1
        X_train_clean.loc[outliers_train, self.features] = np.nan
        X_val_clean.loc[outliers_val, self.features] = np.nan
        for col in self.features:
            median_val = X_train_clean[col].median()
            X_train_clean[col].fillna(median_val, inplace=True)
            X_val_clean[col].fillna(median_val, inplace=True)
        return X_train_clean, X_val_clean

    # --- Box-Cox / Yeo-Johnson Transformation ---
    def boxcox_transform(self):
        pt = PowerTransformer(method='yeo-johnson')
        X_train_clean = self.X_train.copy()
        X_val_clean = self.X_val.copy()
        X_train_clean[self.features] = pt.fit_transform(X_train_clean[self.features])
        X_val_clean[self.features] = pt.transform(X_val_clean[self.features])
        return X_train_clean, X_val_clean

    # --- KNN replacement ---
    def knn_method(self):
        if not self.X_cols:
            warnings.warn("All Features have outliers. KNN method couldn't work.")
            return self.X_train.copy(), self.X_val.copy()
        knn_model = KNeighborsRegressor(n_neighbors=5, weights='distance')
        knn_model.fit(self.X_train[self.X_cols], self.X_train[self.features])
        X_train_clean = self.X_train.copy()
        X_val_clean = self.X_val.copy()
        X_train_clean[self.features] = knn_model.predict(self.X_train[self.X_cols])
        X_val_clean[self.features] = knn_model.predict(self.X_val[self.X_cols])
        return X_train_clean, X_val_clean

    # --- Forward evaluation ---
    def forward(self):
        methods = {
            "IQR": self.IQR,
            "Z_score": self.z_score,
            "Winsorize": self.winsorize,
            "Median_Impute": self.median_impute,
            "Mean_Impute": self.mean_impute,
            "IsolationForest": self.isolation_forest,
            "BoxCox": self.boxcox_transform,
            "KNN": self.knn_method,
            "Original": lambda: (self.X_train.copy(), self.X_val.copy())
        }

        results = []
        best_score = np.inf
        best_X_train = self.X_train.copy()
        best_X_val = self.X_val.copy()
        best_method = "Original"

        for name, method in methods.items():
            try:
                X_tr, X_vl = method()
                model = self.base_model()
                model.fit(X_tr, self.y_train)
                score = self.metric(self.y_val, model.predict(X_vl))
                results.append({"Method": name, "Score": score})
                # lower is better
                if score < best_score:
                    best_score = score
                    best_X_train = X_tr
                    best_X_val = X_vl
                    best_method = name
            except Exception as e:
                warnings.warn(f"Method {name} failed: {e}")

        performance_df = pd.DataFrame(results).sort_values("Score").reset_index(drop=True)
        return best_X_train, best_X_val, best_method, performance_df
