import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("fnb_poc_dataset .csv")

# -----------------------------
# PREPROCESSING
# -----------------------------
df['date'] = pd.to_datetime(df['date'])

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------
df = df.dropna(subset=['quantity_used_kg', 'spoilage_flag'])
df = df.fillna(0)
df['spoilage_flag'] = df['spoilage_flag'].astype(int)

# -----------------------------
# ENCODING
# -----------------------------
cat_cols = ['product_name', 'category', 'supplier_name', 
            'weather_condition', 'storage_type', 'location']

encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# -----------------------------
# FEATURES
# -----------------------------
features = [
    'warehouse_id',
    'product_id',
    'category',
    'supplier_id',
    'quantity_ordered_kg',
    'unit_cost_inr',
    'shelf_life_days',
    'temperature_celsius',
    'humidity_percent',
    'storage_capacity_kg',
    'day_of_week',
    'month'
]

X = df[features]

# =========================================================
# 1. DEMAND MODEL (OPTIMIZED)
# =========================================================
y_demand = df['quantity_used_kg']

X_train, X_test, y_train, y_test = train_test_split(
    X, y_demand, test_size=0.2, random_state=42
)

demand_model = RandomForestRegressor(
    n_estimators=30,        # ↓ drastically reduced
    max_depth=8,            # ↓ control tree growth
    min_samples_split=10,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)

demand_model.fit(X_train, y_train)

pred_demand = demand_model.predict(X_test)
print("\n📈 Demand Model MAE:", mean_absolute_error(y_test, pred_demand))

# =========================================================
# 2. SPOILAGE MODEL (OPTIMIZED)
# =========================================================
y_spoil = df['spoilage_flag']

X_train, X_test, y_train, y_test = train_test_split(
    X, y_spoil, test_size=0.2, random_state=42
)

spoil_model = RandomForestClassifier(
    n_estimators=30,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)

spoil_model.fit(X_train, y_train)

pred_spoil = spoil_model.predict(X_test)
print("⚠️ Spoilage Model Accuracy:", accuracy_score(y_test, pred_spoil))

# =========================================================
# SAVE MODELS (COMPRESSED PROPERLY)
# =========================================================
joblib.dump(demand_model, "demand_model.pkl", compress=5)
joblib.dump(spoil_model, "spoilage_model.pkl", compress=5)
joblib.dump(encoders, "encoders.pkl", compress=3)

print("\n✅ Models saved successfully (compressed!)")
