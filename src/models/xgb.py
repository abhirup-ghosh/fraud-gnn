import xgboost as xgb
from sklearn.model_selection import train_test_split


def train_xgb(df, features):

    X = df[features]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05
    )

    model.fit(X_train, y_train)

    return model