#!/usr/bin/env python3
"""
generar_modelo_prueba.py
========================
Genera un dataset de entregas y un modelo joblib COMPLETO con métricas,
compatible con la app de clasificación.

El joblib incluye:
  - pipeline entrenado (sklearn)
  - label_encoder
  - num_features, cat_features, label_classes
  - metrics: accuracy, cv_mean, cv_std, train_samples, test_samples, report
  - feature_importance
  - charts: confusion_matrix, feature_importance, class_distribution (base64)

Uso:
    python generar_modelo_prueba.py

Salida:
    - dataset_entregas_prueba.csv
    - modelo_entregas_prueba.joblib
"""
import os
import io
import base64
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

np.random.seed(42)
N = 500

# ═══════════════════════════════════════════════════════════════
# 1. GENERAR DATASET
# ═══════════════════════════════════════════════════════════════

data = {
    'distancia_km': np.random.uniform(1, 100, N),
    'peso_kg': np.random.uniform(0.5, 50, N),
    'volumen_m3': np.random.uniform(0.001, 2, N),
    'temperatura_c': np.random.uniform(-5, 40, N),
    'hora_dia': np.random.randint(0, 24, N),
    'dia_semana': np.random.choice(
        ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo'], N
    ),
    'tipo_vehiculo': np.random.choice(['Moto', 'Auto', 'Camioneta', 'Furgon'], N),
    'zona_entrega': np.random.choice(['Urbana', 'Suburbana', 'Rural', 'Industrial'], N),
    'prioridad': np.random.choice(['Baja', 'Media', 'Alta', 'Urgente'], N),
    'experiencia_conductor_anos': np.random.uniform(0, 30, N),
    'trafico': np.random.choice(['Bajo', 'Medio', 'Alto', 'Critico'], N),
    'clima': np.random.choice(['Soleado', 'Nublado', 'Lluvia', 'Tormenta', 'Nieve'], N),
}

df = pd.DataFrame(data)


def clasificar_entrega(row):
    """Reglas lógicas para clasificar cada entrega."""
    score = 0
    if row['distancia_km'] > 70: score += 2
    if row['peso_kg'] > 30: score += 1
    if row['hora_dia'] in [0, 1, 2, 3, 4, 5, 22, 23]: score += 2
    if row['dia_semana'] in ['Sabado', 'Domingo']: score += 1
    if row['zona_entrega'] == 'Rural': score += 2
    if row['prioridad'] == 'Urgente': score -= 1
    if row['experiencia_conductor_anos'] < 2: score += 2
    if row['trafico'] in ['Alto', 'Critico']: score += 2
    if row['clima'] in ['Tormenta', 'Nieve']: score += 3
    if row['temperatura_c'] > 35 or row['temperatura_c'] < 0: score += 1

    if score <= 2:
        return 'Exitosa'
    elif score <= 5:
        return 'Demorada'
    else:
        return 'Fallida'


df['estado_entrega'] = df.apply(clasificar_entrega, axis=1)

# Guardar CSV
csv_path = 'dataset_entregas_prueba.csv'
df.to_csv(csv_path, index=False)
print(f"✓ Dataset guardado: {csv_path} ({len(df)} filas x {len(df.columns)} columnas)")
print(f"  Distribución de clases:")
for clase, count in df['estado_entrega'].value_counts().items():
    print(f"    {clase}: {count}")

# ═══════════════════════════════════════════════════════════════
# 2. ENTRENAR MODELO + CALCULAR MÉTRICAS
# ═══════════════════════════════════════════════════════════════

target_col = 'estado_entrega'
num_features = [c for c in df.columns
                if c != target_col and df[c].dtype in ('int64', 'float64')]
cat_features = [c for c in df.columns
              if c != target_col and df[c].dtype == 'object']

X = df[num_features + cat_features].copy()
y_raw = df[target_col].astype(str)

le = LabelEncoder()
y = le.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_features),
    ("cat", categorical_transformer, cat_features),
], remainder="drop")

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
])

pipeline.fit(X_train, y_train)

# ── Métricas ──
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

# Feature importance
clf = pipeline.named_steps["classifier"]
all_features = num_features + cat_features
importances = clf.feature_importances_
feat_importance = sorted(
    zip(all_features, importances),
    key=lambda x: x[1], reverse=True
)
feat_importance = [{"feature": f, "importance": round(float(i), 4)} for f, i in feat_importance]

# Classification report limpio
def clean_report(report, classes):
    rows = []
    for cls in classes:
        key = str(cls)
        if key in report:
            r = report[key]
            rows.append({
                "clase":     cls,
                "precision": round(r["precision"], 3),
                "recall":    round(r["recall"], 3),
                "f1":        round(r["f1-score"], 3),
                "soporte":   int(r["support"]),
            })
    return rows

report_rows = clean_report(report, le.classes_)

print(f"✓ Modelo entrenado — Accuracy en test: {acc:.2%}")
print(f"  CV: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

# ═══════════════════════════════════════════════════════════════
# 3. GRÁFICOS BASE64 (mismo estilo que la app)
# ═══════════════════════════════════════════════════════════════

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# Confusion matrix
fig, ax = plt.subplots(figsize=(5.5, 4.5))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")
sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
    xticklabels=le.classes_, yticklabels=le.classes_,
    linewidths=.5, linecolor="#1e2230", ax=ax, cbar_kws={"shrink": .75})
ax.set_xlabel("Predicción", color="#9ca3af", fontsize=10)
ax.set_ylabel("Real", color="#9ca3af", fontsize=10)
ax.set_title("Matriz de Confusión", color="#f97316", fontsize=12, pad=12)
ax.tick_params(colors="#9ca3af")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", color="#d1d5db")
plt.setp(ax.get_yticklabels(), rotation=0, color="#d1d5db")
fig.tight_layout()
chart_cm = fig_to_b64(fig)

# Feature importance
if feat_importance:
    top = feat_importance[:12]
    feats = [f["feature"] for f in top]
    imps = [f["importance"] for f in top]
    fig, ax = plt.subplots(figsize=(6, max(3, len(feats) * 0.38)))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    colors = [f"#f97316{hex(int(255 * (1 - i/len(imps))))[2:].zfill(2)}" for i in range(len(imps))]
    bars = ax.barh(feats[::-1], imps[::-1], color=colors[::-1], edgecolor="none", height=0.65)
    ax.set_xlabel("Importancia", color="#9ca3af", fontsize=9)
    ax.set_title("Variables más influyentes", color="#f97316", fontsize=12, pad=10)
    ax.tick_params(colors="#d1d5db", labelsize=9)
    ax.spines[:].set_visible(False)
    ax.xaxis.grid(True, color="#1e2230", linestyle="--", linewidth=.7)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, imps[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", color="#9ca3af", fontsize=8)
    fig.tight_layout()
    chart_fi = fig_to_b64(fig)
else:
    chart_fi = ""

# Class distribution
counts = y_raw.value_counts()
fig, ax = plt.subplots(figsize=(5, 3.5))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")
palette = ["#f97316", "#fb923c", "#fdba74", "#fed7aa"]
ax.bar(counts.index, counts.values, color=palette[:len(counts)], edgecolor="none", width=0.55)
ax.set_title("Distribución de clases", color="#f97316", fontsize=12, pad=10)
ax.set_ylabel("Registros", color="#9ca3af", fontsize=9)
ax.tick_params(colors="#d1d5db", labelsize=9)
ax.spines[:].set_visible(False)
ax.yaxis.grid(True, color="#1e2230", linestyle="--", linewidth=.7)
ax.set_axisbelow(True)
fig.tight_layout()
chart_dist = fig_to_b64(fig)

print(f"✓ Gráficos generados (3 imágenes base64)")

# ═══════════════════════════════════════════════════════════════
# 4. GUARDAR JOBLIB COMPLETO
# ═══════════════════════════════════════════════════════════════

payload = {
    "pipeline":      pipeline,
    "label_encoder": le,
    "num_features":  num_features,
    "cat_features":  cat_features,
    "label_classes": le.classes_.tolist(),
    "model_key":     "random_forest",
    "target_col":    target_col,
    # ── Métricas de entrenamiento ──
    "metrics": {
        "accuracy":      round(float(acc), 4),
        "cv_mean":       round(float(cv_scores.mean()), 4),
        "cv_std":        round(float(cv_scores.std()), 4),
        "train_samples": len(X_train),
        "test_samples":  len(X_test),
        "report":        report_rows,
    },
    "feature_importance": feat_importance,
    "charts": {
        "confusion_matrix":    chart_cm,
        "feature_importance":  chart_fi,
        "class_distribution":  chart_dist,
    },
}

joblib_path = 'modelo_entregas_prueba.joblib'
joblib.dump(payload, joblib_path)
print(f"✓ Modelo guardado: {joblib_path}")

# Verificación
loaded = joblib.load(joblib_path)
print(f"Verificación de carga:")
print(f"    num_features: {len(loaded['num_features'])}")
print(f"    cat_features: {len(loaded['cat_features'])}")
print(f"    label_classes: {loaded['label_classes']}")
print(f"    metrics.accuracy: {loaded['metrics']['accuracy']}")
print(f"    metrics.report: {len(loaded['metrics']['report'])} clases")
print(f"    feature_importance: {len(loaded['feature_importance'])} features")
print(f"    charts: {list(loaded['charts'].keys())}")

# Prueba de predicción
test_row = X_test.iloc[0:1]
pred = loaded['pipeline'].predict(test_row)
pred_label = loaded['label_encoder'].inverse_transform(pred)
print(f"Predicción de prueba: {pred_label[0]}")

print(f"{'='*50}")
print("Puedes subir ambos archivos a la app:")
print("  1. dataset_entregas_prueba.csv   → para predict_batch")
print("  2. modelo_entregas_prueba.joblib → para cargar modelo propio")
print("     (incluye métricas, gráficos y feature importance)")