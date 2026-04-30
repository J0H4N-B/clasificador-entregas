"""
app/blueprints/model.py — Entrenamiento y predicción
======================================================
Endpoints:
    POST /api/model/train        → entrena el modelo con el CSV cargado
    GET  /api/model/results      → retorna métricas y gráficos del último entrenamiento
    POST /api/model/predict      → predice una sola entrega (JSON body)
    POST /api/model/predict_batch → predice todas las filas del CSV cargado
    GET  /api/model/models       → lista modelos disponibles
"""
import os
from flask import Blueprint, request, jsonify, session, current_app
from app.utils.file_handler import read_csv_safe, FileValidationError
from app.utils.ml_engine import (
    train_model, predict_single, predict_batch,
    save_model, load_model, AVAILABLE_MODELS, MLError,
)

model_bp = Blueprint("model", __name__)


def _require_csv():
    path = session.get("csv_path")
    if not path:
        raise ValueError("No hay CSV cargado. Sube un archivo primero.")
    return read_csv_safe(path)


@model_bp.route("/models", methods=["GET"])
def list_models():
    """Lista los modelos disponibles para elegir desde el frontend."""
    return jsonify({
        key: {"label": v["label"], "description": v["description"]}
        for key, v in AVAILABLE_MODELS.items()
    })


@model_bp.route("/train", methods=["POST"])
def train():
    """
    Entrena el modelo con el CSV de la sesión.

    Body JSON:
        target_col  — columna a predecir (ej: "estado_entrega")
        model_key   — algoritmo: random_forest | gradient_boosting | logistic_regression | ...
        test_size   — fracción de test (default: 0.2)
    """
    try:
        df = _require_csv()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    body       = request.get_json(silent=True) or {}
    target_col = body.get("target_col", "")
    model_key  = body.get("model_key", "random_forest")
    test_size  = float(body.get("test_size", 0.2))

    # Validaciones de entrada
    if not target_col:
        return jsonify({"error": "Debes seleccionar una columna target."}), 422
    if model_key not in AVAILABLE_MODELS:
        return jsonify({"error": f"Modelo '{model_key}' no disponible."}), 422
    if not (0.1 <= test_size <= 0.4):
        return jsonify({"error": "test_size debe estar entre 0.1 y 0.4."}), 422

    try:
        result = train_model(df, target_col, model_key, test_size)
    except MLError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        current_app.logger.error(f"Train error: {e}")
        return jsonify({"error": "Error durante el entrenamiento."}), 500

    # Guardar modelo en disco
    model_path = save_model(result, current_app.config["MODELS_FOLDER"], model_key)

    # Guardar metadata en sesión para predict
    session["model_key"]     = model_key
    session["target_col"]    = target_col
    session["num_features"]  = result["num_features"]
    session["cat_features"]  = result["cat_features"]
    session["label_classes"] = result["label_classes"]

    return jsonify({
        "ok":                True,
        "model_key":         model_key,
        "target_col":        target_col,
        "label_classes":     result["label_classes"],
        "num_features":      result["num_features"],
        "cat_features":      result["cat_features"],
        "metrics":           result["metrics"],
        "feature_importance": result["feature_importance"][:10],
        "charts":            result["charts"],
    })


@model_bp.route("/results", methods=["GET"])
def results():
    """Retorna los resultados del último entrenamiento desde la sesión."""
    model_key = session.get("model_key")
    if not model_key:
        return jsonify({"error": "No hay modelo entrenado."}), 400
    return jsonify({
        "model_key":    model_key,
        "target_col":  session.get("target_col"),
        "num_features": session.get("num_features"),
        "cat_features": session.get("cat_features"),
        "label_classes": session.get("label_classes"),
    })


@model_bp.route("/predict", methods=["POST"])
def predict():
    """
    Predice la clase de UNA entrega.

    Body JSON: { "campo1": valor, "campo2": valor, ... }
    Retorna:   { "prediccion": "Exitosa", "probabilidades": {...} }
    """
    model_key = session.get("model_key")
    if not model_key:
        return jsonify({"error": "No hay modelo entrenado."}), 400

    input_data = request.get_json(silent=True) or {}
    if not input_data:
        return jsonify({"error": "Body vacío."}), 422

    try:
        payload = load_model(current_app.config["MODELS_FOLDER"], model_key)
        result  = predict_single(
            payload["pipeline"],
            payload["label_encoder"],
            payload["num_features"],
            payload["cat_features"],
            input_data,
        )
        return jsonify(result)
    except MLError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Predict error: {e}")
        return jsonify({"error": "Error en la predicción."}), 500


@model_bp.route("/predict_batch", methods=["POST"])
def predict_batch_endpoint():
    """
    Predice la clase para TODAS las filas del CSV cargado.
    Retorna las primeras 200 filas con la predicción añadida.
    """
    model_key = session.get("model_key")
    if not model_key:
        return jsonify({"error": "No hay modelo entrenado."}), 400

    try:
        df      = _require_csv()
        payload = load_model(current_app.config["MODELS_FOLDER"], model_key)
        df_pred = predict_batch(
            payload["pipeline"],
            payload["label_encoder"],
            payload["num_features"],
            payload["cat_features"],
            df,
        )
        # Solo retornar preview (200 filas) para no saturar el frontend
        preview = df_pred.head(200).fillna("").to_dict(orient="records")
        return jsonify({
            "ok":            True,
            "total_rows":    len(df_pred),
            "preview_rows":  len(preview),
            "records":       preview,
            "columns":       df_pred.columns.tolist(),
        })
    except (ValueError, MLError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        current_app.logger.error(f"Batch predict error: {e}")
        return jsonify({"error": "Error en predicción por lotes."}), 500