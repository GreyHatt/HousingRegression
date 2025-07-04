from utils import (
    load_data, 
    train_test_split, 
    evaluate_model,
    grid_search_cv,
    comparison_graph
)
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

    # perform hyperparameter tuning
    X_train, X_test, y_train, y_test = train_test_split(df)
    models = {
        "Linear Regression": (
            LinearRegression(),
            {
                'fit_intercept': [True, False],
                'copy_X': [True, False]
            }
        ),
        "Decision Tree": (
            DecisionTreeRegressor(random_state=13),
            {
                'max_depth': [2, 4, 6, 8, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=13),
            {
                'n_estimators': [50, 100],
                'max_depth': [4, 8, 16, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        )
    }
    results = []
    for name, (model, param_grid) in models.items():
        print(f"\nTuning {name}...")
        grid = grid_search_cv(model, param_grid, X_train, y_train)
        best_model = grid.best_estimator_
        mse, r2 = evaluate_model(best_model, X_test, y_test)
        print(f"{name}: Best Params: {grid.best_params_}")
        print(f"{name}: MSE: {mse:.2f}, R2: {r2:.2f}")
        results.append({
            "Model": name,
            "MSE": mse,
            "R2": r2,
            "Best Params": grid.best_params_
        })
    
    comparison_graph(results)

if __name__ == "__main__":
    main()