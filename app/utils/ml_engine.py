"""
app/utils/ml_engine.py — Motor de Machine Learning
====================================================
Centraliza toda la lógica de entrenamiento y predicción.

Para agregar un nuevo modelo:
  1. Abre este archivo
  2. Añade una entrada en AVAILABLE_MODELS con tu clasificador
  3. Reinicia la app

Ejemplo:
    "svm": {
        "label": "SVM",
        "description": "Máquinas de soporte vectorial.",
        "class": SVC,
        "params": {"kernel": "rbf", "C": 1.0, "random_state": 42},
    }
"""
import os
import io
import base64
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.svm              import SVC
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.tree             import DecisionTreeClassifier
from sklearn.naive_bayes      import GaussianNB
from sklearn.pipeline         import Pipeline
from sklearn.compose          import ColumnTransformer
from sklearn.preprocessing    import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute            import SimpleImputer
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.metrics           import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)


class MLError(Exception):
    pass


# ═══════════════════════════════════════════════════════════════
# MODELOS DISPONIBLES — Edita aquí para agregar más
# ═══════════════════════════════════════════════════════════════

AVAILABLE_MODELS = {
    "random_forest": {
        "label": "Random Forest",
        "description": "Robusto, maneja datos mixtos bien, muestra feature importance.",
        "class": RandomForestClassifier,
        "params": {"n_estimators": 100, "max_depth": None,
                   "random_state": 42, "n_jobs": -1},
    },
    "gradient_boosting": {
        "label": "Gradient Boosting",
        "description": "Alta precisión, más lento de entrenar.",
        "class": GradientBoostingClassifier,
        "params": {"n_estimators": 100, "learning_rate": 0.1,
                   "max_depth": 3, "random_state": 42},
    },
    "logistic_regression": {
        "label": "Regresión Logística",
        "description": "Rápido e interpretable, ideal para relaciones lineales.",
        "class": LogisticRegression,
        "params": {"max_iter": 1000, "random_state": 42},
    },
    "extra_trees": {
        "label": "Extra Trees",
        "description": "Más rápido que Random Forest, menos propenso al overfitting.",
        "class": ExtraTreesClassifier,
        "params": {"n_estimators": 100, "max_depth": None,
                   "random_state": 42, "n_jobs": -1},
    },
    "svm": {
        "label": "SVM (RBF)",
        "description": "Máquinas de soporte vectorial con kernel radial. Bueno para fronteras complejas.",
        "class": SVC,
        "params": {"kernel": "rbf", "C": 1.0, "random_state": 42, "probability": True},
    },
    "knn": {
        "label": "K-Nearest Neighbors",
        "description": "Clasifica según los vecinos más cercanos. Simple y efectivo.",
        "class": KNeighborsClassifier,
        "params": {"n_neighbors": 5, "weights": "uniform"},
    },
    "decision_tree": {
        "label": "Árbol de Decisión",
        "description": "Muy interpretable. Ideal para entender reglas de decisión.",
        "class": DecisionTreeClassifier,
        "params": {"max_depth": None, "random_state": 42},
    },
    "naive_bayes": {
        "label": "Naive Bayes",
        "description": "Extremadamente rápido. Asume independencia entre features.",
        "class": GaussianNB,
        "params": {},
    },
}


# ── Preprocesamiento ───────────────────────────────────────────

def detect_feature_types(df: pd.DataFrame, target_col: str) -> tuple[list, list]:
    """
    Separa las columnas en numéricas y categóricas,
    excluyendo el target y columnas con demasiados valores únicos (IDs).
    """
    feat_cols = [c for c in df.columns if c != target_col]
    num_feats, cat_feats = [], []

    for col in feat_cols:
        if df[col].dtype in ("int64", "float64"):
            num_feats.append(col)
        else:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio <= 0.8:
                cat_feats.append(col)

    return num_feats, cat_feats


def build_pipeline(model_key: str, num_features: list, cat_features: list) -> Pipeline:
    """
    Construye un Pipeline de scikit-learn con preprocesamiento + clasificador.
    """
    if model_key not in AVAILABLE_MODELS:
        raise MLError(f"Modelo '{model_key}' no disponible.")

    cfg = AVAILABLE_MODELS[model_key]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ], remainder="drop")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   cfg["class"](**cfg["params"])),
    ])

    return pipeline


# ── Entrenamiento ──────────────────────────────────────────────

def train_model(
    df: pd.DataFrame,
    target_col: str,
    model_key: str = "random_forest",
    test_size: float = 0.2,
) -> dict:
    """
    Entrena el modelo y retorna un dict con:
      - pipeline, label_encoder, metrics, feature_importance, charts, etc.
    """
    # ── Validaciones ──
    if target_col not in df.columns:
        raise MLError(f"La columna target '{target_col}' no existe en el CSV.")

    target_vals = df[target_col].dropna().unique()
    if len(target_vals) < 2:
        raise MLError("El target debe tener al menos 2 clases distintas.")
    if len(target_vals) > 20:
        raise MLError(f"El target tiene {len(target_vals)} clases únicas.")

    if len(df) < 20:
        raise MLError("El dataset tiene menos de 20 filas.")

    num_feats, cat_feats = detect_feature_types(df, target_col)
    if not num_feats and not cat_feats:
        raise MLError("No se encontraron columnas de características válidas.")

    # ── Preparar X e y ──
    X = df[num_feats + cat_feats].copy()
    y_raw = df[target_col].astype(str)

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    # ── Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # ── Entrenar ──
    pipeline = build_pipeline(model_key, num_feats, cat_feats)
    pipeline.fit(X_train, y_train)

    # ── Evaluar ──
    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred)

    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")

    # ── Feature importance ──
    feat_importance = _get_feature_importance(pipeline, num_feats, cat_feats)

    # ── Gráficos ──
    chart_cm    = _plot_confusion_matrix(cm, le.classes_)
    chart_fi    = _plot_feature_importance(feat_importance)
    chart_dist  = _plot_class_distribution(y_raw)

    return {
        "pipeline":          pipeline,
        "label_encoder":     le,
        "num_features":      num_feats,
        "cat_features":      cat_feats,
        "label_classes":     le.classes_.tolist(),
        "metrics": {
            "accuracy":      round(float(acc), 4),
            "cv_mean":       round(float(cv_scores.mean()), 4),
            "cv_std":        round(float(cv_scores.std()), 4),
            "train_samples": len(X_train),
            "test_samples":  len(X_test),
            "report":        _clean_report(report, le.classes_),
        },
        "feature_importance": feat_importance,
        "charts": {
            "confusion_matrix":    chart_cm,
            "feature_importance":  chart_fi,
            "class_distribution":  chart_dist,
        },
    }


# ── Feature importance ─────────────────────────────────────────

def _get_feature_importance(pipeline: Pipeline,
                             num_feats: list,
                             cat_feats: list) -> list:
    """
    Extrae feature importance del clasificador.
    Funciona con tree-based models y LogisticRegression.
    """
    clf = pipeline.named_steps["classifier"]
    all_features = num_feats + cat_feats

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_).mean(axis=0)
    else:
        return []

    pairs = sorted(
        zip(all_features, importances),
        key=lambda x: x[1], reverse=True
    )
    return [{"feature": f, "importance": round(float(i), 4)} for f, i in pairs]


# ── Predicción ─────────────────────────────────────────────────

def predict_single(pipeline, label_encoder, num_feats, cat_feats, input_data: dict) -> dict:
    """
    Predice la clase de una sola entrega a partir de un dict de valores.
    """
    all_feats = num_feats + cat_feats
    row = {}
    for f in all_feats:
        val = input_data.get(f)
        if val is None or val == "":
            row[f] = np.nan
        else:
            try:
                row[f] = float(val) if f in num_feats else str(val)
            except ValueError:
                row[f] = np.nan

    X_input = pd.DataFrame([row])[all_feats]
    pred_idx = pipeline.predict(X_input)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]

    proba = {}
    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        proba_arr = pipeline.predict_proba(X_input)[0]
        proba = {
            label_encoder.inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(proba_arr)
        }

    return {"prediccion": pred_label, "probabilidades": proba}


def predict_batch(pipeline, label_encoder, num_feats, cat_feats,
                  df: pd.DataFrame) -> pd.DataFrame:
    """
    Predice la clase para todas las filas de un DataFrame.
    """
    all_feats = [f for f in num_feats + cat_feats if f in df.columns]
    X = df[all_feats].copy()
    preds = pipeline.predict(X)
    df = df.copy()
    df["prediccion"] = label_encoder.inverse_transform(preds)
    return df


# ── Serialización ──────────────────────────────────────────────

def save_model(result: dict, models_folder: str, model_key: str) -> str:
    """Guarda el pipeline y metadata con joblib. Retorna la ruta del archivo."""
    os.makedirs(models_folder, exist_ok=True)
    filename = f"model_{model_key}.joblib"
    path     = os.path.join(models_folder, filename)
    payload  = {
        "pipeline":      result["pipeline"],
        "label_encoder": result["label_encoder"],
        "num_features":  result["num_features"],
        "cat_features":  result["cat_features"],
        "label_classes": result["label_classes"],
        "model_key":     model_key,
    }
    joblib.dump(payload, path)
    return path


def load_model(models_folder: str, model_key: str) -> dict:
    """Carga el pipeline desde disco. Lanza MLError si no existe."""
    path = os.path.join(models_folder, f"model_{model_key}.joblib")
    if not os.path.exists(path):
        raise MLError("No hay modelo entrenado. Entrena primero el modelo.")
    return joblib.load(path)


# ── Gráficos ───────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    """Convierte una figura matplotlib a string base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _plot_confusion_matrix(cm: np.ndarray, classes: list) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=classes, yticklabels=classes,
        linewidths=.5, linecolor="#1e2230",
        ax=ax, cbar_kws={"shrink": .75}
    )
    ax.set_xlabel("Predicción", color="#9ca3af", fontsize=10)
    ax.set_ylabel("Real",       color="#9ca3af", fontsize=10)
    ax.set_title("Matriz de Confusión", color="#f97316", fontsize=12, pad=12)
    ax.tick_params(colors="#9ca3af")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", color="#d1d5db")
    plt.setp(ax.get_yticklabels(), rotation=0, color="#d1d5db")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_feature_importance(feat_imp: list, top_n: int = 12) -> str:
    if not feat_imp:
        return ""
    top    = feat_imp[:top_n]
    feats  = [f["feature"] for f in top]
    imps   = [f["importance"] for f in top]
    colors = [f"#f97316{hex(int(255 * (1 - i/len(imps))))[2:].zfill(2)}" for i in range(len(imps))]

    fig, ax = plt.subplots(figsize=(6, max(3, len(feats) * 0.38)))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    bars = ax.barh(feats[::-1], imps[::-1], color=colors[::-1],
                   edgecolor="none", height=0.65)
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
    return _fig_to_b64(fig)


def _plot_class_distribution(y_raw: pd.Series) -> str:
    counts = y_raw.value_counts()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    palette = ["#f97316", "#fb923c", "#fdba74", "#fed7aa"]
    ax.bar(counts.index, counts.values,
           color=palette[:len(counts)], edgecolor="none", width=0.55)
    ax.set_title("Distribución de clases", color="#f97316", fontsize=12, pad=10)
    ax.set_ylabel("Registros", color="#9ca3af", fontsize=9)
    ax.tick_params(colors="#d1d5db", labelsize=9)
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color="#1e2230", linestyle="--", linewidth=.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── Utilidades ─────────────────────────────────────────────────

def _clean_report(report: dict, classes) -> list:
    """Convierte el dict de classification_report a lista de filas para tabla."""
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