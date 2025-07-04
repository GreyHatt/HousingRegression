from utils import load_data, train_test_split, evaluate_model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(df)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=13),
        "Random Forest": RandomForestRegressor(random_state=13)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        mse, r2 = evaluate_model(model, X_test, y_test)
        results[name] = {"MSE": mse, "R2": r2}
        print(f"{name} - MSE: {mse:.2f}, R2: {r2:.2f}")

    # Optionally, save models
    # import joblib
    # for name, model in models.items():
    #     joblib.dump(model, f"{name.replace(' ', '_')}.joblib")

if __name__ == "__main__":
    main()