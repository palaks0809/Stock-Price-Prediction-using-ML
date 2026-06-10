# ============================================================
# ADVANCED AAPL STOCK PRICE FORECASTING PROJECT
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

# ============================================================
# 1. PROBLEM DEFINITION
# ============================================================

print("=" * 80)
print("ADVANCED STOCK PRICE FORECASTING USING MACHINE LEARNING")
print("=" * 80)

print("""
OBJECTIVE

Forecast future stock prices using:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Elastic Net
5. Random Forest
6. AdaBoost
7. XGBoost

The dataset contains more than 60 technical
indicators used to identify trend, momentum,
volatility and price strength.
""")

TARGET = "Close_forcast"

# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading Dataset...")

try:
    df = pd.read_csv("AAPL.csv")
except Exception as e:
    print("Error Loading Dataset")
    print(e)
    exit()

print("Dataset Shape:", df.shape)

# ============================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print("\nFirst 5 Records")
print(df.head())

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

# ============================================================
# MISSING VALUES
# ============================================================

print("\nMissing Values")

missing = df.isnull().sum()

missing_df = pd.DataFrame({
    "Column": missing.index,
    "Missing Values": missing.values
})

print(
    missing_df[
        missing_df["Missing Values"] > 0
    ]
)

# ============================================================
# DUPLICATES
# ============================================================

duplicates = df.duplicated().sum()

print("\nDuplicate Records:", duplicates)

# ============================================================
# CORRELATION HEATMAP
# ============================================================

numeric_df = df.select_dtypes(
    include=np.number
)

plt.figure(figsize=(18,14))

sns.heatmap(
    numeric_df.corr(),
    cmap="coolwarm",
    center=0
)

plt.title(
    "Feature Correlation Matrix"
)

plt.tight_layout()

plt.savefig(
    "correlation_heatmap.png",
    dpi=300
)

plt.close()

# ============================================================
# FEATURE DISTRIBUTION
# ============================================================

numeric_df.hist(
    figsize=(18,14),
    bins=20
)

plt.tight_layout()

plt.savefig(
    "feature_distribution.png",
    dpi=300
)

plt.close()

# ============================================================
# 4. DATA ENGINEERING
# ============================================================

print("\n" + "=" * 80)
print("DATA ENGINEERING")
print("=" * 80)

if "Close" in df.columns:

    # Lag Features

    df["Close_Lag_1"] = (
        df["Close"].shift(1)
    )

    df["Close_Lag_3"] = (
        df["Close"].shift(3)
    )

    df["Close_Lag_5"] = (
        df["Close"].shift(5)
    )

    # Daily Return

    df["Daily_Return"] = (
        df["Close"]
        .pct_change()
    )

    # Rolling Means

    df["Rolling_Mean_5"] = (
        df["Close"]
        .rolling(5)
        .mean()
    )

    df["Rolling_Mean_10"] = (
        df["Close"]
        .rolling(10)
        .mean()
    )

    # Rolling STD

    df["Rolling_STD_5"] = (
        df["Close"]
        .rolling(5)
        .std()
    )

    # Momentum

    df["Momentum_10"] = (
        df["Close"]
        -
        df["Close"].shift(10)
    )

    # Volatility

    df["Volatility_10"] = (
        df["Daily_Return"]
        .rolling(10)
        .std()
    )

print("Feature Engineering Completed")

# ============================================================
# 5. DATA PREPARATION
# ============================================================

print("\n" + "=" * 80)
print("DATA PREPARATION")
print("=" * 80)

if "Date" in df.columns:

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df.sort_values(
        by="Date",
        inplace=True
    )

# Remove Duplicates

df.drop_duplicates(
    inplace=True
)

# Handle Missing Values

df = df.ffill().bfill()

print(
    "Remaining Missing Values:",
    df.isnull().sum().sum()
)

print(
    "Shape After Cleaning:",
    df.shape
)

# ============================================================
# 6. FEATURE SELECTION
# ============================================================

print("\n" + "=" * 80)
print("FEATURE SELECTION")
print("=" * 80)

corr_matrix = df.select_dtypes(
    include=np.number
).corr()

if TARGET not in corr_matrix.columns:

    print(
        f"\nERROR: {TARGET} not found"
    )

    print(
        corr_matrix.columns.tolist()
    )

    exit()

target_corr = (
    corr_matrix[TARGET]
    .abs()
    .sort_values(
        ascending=False
    )
)

print("\nTop 20 Correlated Features")

print(
    target_corr.head(20)
)

# ============================================================
# TOP 20 FEATURE HEATMAP
# ============================================================

top_features = (
    target_corr
    .head(20)
    .index
)

plt.figure(figsize=(12,10))

sns.heatmap(
    df[top_features].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title(
    f"Top 20 Features Correlated With {TARGET}"
)

plt.tight_layout()

plt.savefig(
    "top20_correlation_heatmap.png",
    dpi=300
)

plt.close()

print(
    "Saved: top20_correlation_heatmap.png"
)

# ============================================================
# 7. DATA TRANSFORMATION
# ============================================================

print("\n" + "=" * 80)
print("DATA TRANSFORMATION")
print("=" * 80)

drop_columns = [TARGET]

if "Date" in df.columns:
    drop_columns.append("Date")

X = df.drop(
    columns=drop_columns,
    errors="ignore"
)

X = X.select_dtypes(
    include=np.number
)

y = df[TARGET]

print("\nFeature Matrix Shape")
print(X.shape)

print("\nTarget Shape")
print(y.shape)

# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

print(
    "\nFeature Scaling Completed"
)

# ============================================================
# 8. TRAIN TEST SPLIT
# ============================================================

print("\n" + "=" * 80)
print("TRAIN TEST SPLIT")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42
)

print(
    "Training Shape:",
    X_train.shape
)

print(
    "Testing Shape:",
    X_test.shape
)

# ============================================================
# 9. MODEL BUILDING
# ============================================================

print("\n" + "=" * 80)
print("MODEL BUILDING")
print("=" * 80)

models = {

    "Linear Regression":
        LinearRegression(),

    "Ridge Regression":
        Ridge(alpha=1.0),

    "Lasso Regression":
        Lasso(alpha=0.001),

    "Elastic Net":
        ElasticNet(
            alpha=0.001,
            l1_ratio=0.5
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

    "AdaBoost":
        AdaBoostRegressor(
            n_estimators=100,
            random_state=42
        )
}

# ============================================================
# XGBOOST
# ============================================================

try:

    from xgboost import XGBRegressor

    models["XGBoost"] = XGBRegressor(

        n_estimators=100,

        learning_rate=0.05,

        max_depth=4,

        objective='reg:squarederror',

        random_state=42
    )

    print(
        "XGBoost Loaded Successfully"
    )

except:

    print(
        "XGBoost Not Installed"
    )

print("\nModels Ready")

for model_name in models.keys():

    print(
        f"✓ {model_name}"
    )

# ============================================================
# 10. MODEL EVALUATION
# ============================================================

results = []

best_model = None

best_r2 = -999

best_predictions = None

print("\n" + "=" * 80)
print("MODEL TRAINING")
print("=" * 80)

for name, model in models.items():

    print(
        f"\nTraining {name}"
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    mape = (
        mean_absolute_percentage_error(
            y_test,
            predictions
        )
    )

    results.append([

        name,

        mae,

        rmse,

        r2,

        mape
    ])

    print(
        f"MAE={mae:.4f} | "
        f"RMSE={rmse:.4f} | "
        f"R²={r2:.6f} | "
        f"MAPE={mape:.6f}"
    )

    if r2 > best_r2:

        best_r2 = r2

        best_model = model

        best_predictions = predictions

        # ============================================================
# 11. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

results_df = pd.DataFrame(

    results,

    columns=[

        "Model",

        "MAE",

        "RMSE",

        "R2",

        "MAPE"
    ]
)

results_df = results_df.sort_values(

    "R2",

    ascending=False
)

print(results_df)

results_df.to_csv(

    "model_comparison.csv",

    index=False
)

# ============================================================
# IMPROVED MODEL COMPARISON CHART
# ============================================================

plt.figure(figsize=(10,6))

comparison = results_df.sort_values(

    "R2",

    ascending=True
)

bars = plt.barh(

    comparison["Model"],

    comparison["R2"]
)

for bar in bars:

    width = bar.get_width()

    plt.text(

        width,

        bar.get_y() + bar.get_height()/2,

        f"{width:.6f}",

        va="center"
    )

plt.xlabel("R² Score")

plt.ylabel("Model")

plt.title("Machine Learning Model Comparison")

plt.xlim(

    comparison["R2"].min() - 0.001,

    1.0
)

plt.tight_layout()

plt.savefig(

    "model_comparison.png",

    dpi=300
)

plt.close()

print("Saved: model_comparison.png")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE")
print("=" * 80)

if hasattr(best_model, "feature_importances_"):

    importance_df = pd.DataFrame({

        "Feature":
            X.columns,

        "Importance":
            best_model.feature_importances_
    })

    importance_df = (

        importance_df

        .sort_values(

            "Importance",

            ascending=False
        )
    )

    print(
        importance_df.head(20)
    )

    importance_df.to_csv(

        "feature_importance.csv",

        index=False
    )

    top15 = importance_df.head(15)

    plt.figure(figsize=(10,6))

    sns.barplot(

        data=top15,

        x="Importance",

        y="Feature"
    )

    for idx, value in enumerate(
            top15["Importance"]):

        plt.text(

            value,

            idx,

            f"{value:.3f}"
        )

    plt.title(
        "Top 15 Important Features"
    )

    plt.tight_layout()

    plt.savefig(

        "feature_importance.png",

        dpi=300
    )

    plt.close()

    print(
        "Saved: feature_importance.png"
    )

# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

print("\nGenerating Actual vs Predicted Plot...")

plt.figure(figsize=(8,6))

plt.scatter(

    y_test,

    best_predictions,

    alpha=0.5
)

plt.xlabel("Actual Price")

plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted Stock Price")

plt.tight_layout()

plt.savefig(

    "actual_vs_predicted.png",

    dpi=300
)

plt.close()

print(
    "Saved: actual_vs_predicted.png"
)

# ============================================================
# FORECAST NEXT 7 DAYS
# ============================================================

print("\n" + "=" * 80)
print("NEXT WEEK FORECAST")
print("=" * 80)

future_data = X_scaled.tail(7)

forecast = best_model.predict(
    future_data
)

forecast_df = pd.DataFrame({

    "Forecast_Day":

        [1,2,3,4,5,6,7],

    "Forecast_Close":

        forecast
})

print(forecast_df)

forecast_df.to_csv(

    "AAPL_1Week_Forecast.csv",

    index=False
)

# ============================================================
# FORECAST VISUALIZATION
# ============================================================

plt.figure(figsize=(8,5))

plt.plot(

    forecast_df["Forecast_Day"],

    forecast_df["Forecast_Close"],

    marker="o"
)

plt.title(
    "Next 7-Day Stock Price Forecast"
)

plt.xlabel("Day")

plt.ylabel("Forecasted Price")

plt.grid(True)

plt.tight_layout()

plt.savefig(

    "forecast_next_week.png",

    dpi=300
)

plt.close()

print(
    "Saved: forecast_next_week.png"
)

# ============================================================
# PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("PROJECT COMPLETED")
print("=" * 80)

print(f"\nBest Model: {results_df.iloc[0]['Model']}")

print(f"Best R² Score: {results_df.iloc[0]['R2']:.6f}")

print("""
OUTPUT FILES GENERATED

1. correlation_heatmap.png
2. feature_distribution.png
3. top20_correlation_heatmap.png
4. model_comparison.png
5. feature_importance.png
6. actual_vs_predicted.png
7. forecast_next_week.png

8. model_comparison.csv
9. feature_importance.csv
10. AAPL_1Week_Forecast.csv

PROJECT STEPS COMPLETED

✓ Problem Definition
✓ Exploratory Data Analysis
✓ Data Engineering
✓ Data Preparation
✓ Feature Selection
✓ Data Transformation
✓ Model Building
✓ Model Evaluation
✓ Feature Importance
✓ Stock Forecasting
✓ Project Summary
""")