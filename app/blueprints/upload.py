"""
app/blueprints/upload.py — Carga del CSV
=========================================
Endpoints:
    POST /api/upload/csv    → sube el CSV, retorna columnas disponibles
    GET  /api/upload/sample → descarga el CSV de ejemplo
"""
import os
from flask import Blueprint, request, jsonify, session, current_app, send_file
from app.utils.file_handler import (
    save_uploaded_file, read_csv_safe,
    cleanup_old_uploads, FileValidationError,
)

upload_bp = Blueprint("upload", __name__)


@upload_bp.route("/csv", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "No se encontró el campo 'file'."}), 400

    file = request.files["file"]
    try:
        filepath = save_uploaded_file(file)
        cleanup_old_uploads(current_app.config["UPLOAD_FOLDER"])

        df = read_csv_safe(filepath)
        session["csv_path"] = filepath
        session["csv_name"] = file.filename

        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        return jsonify({
            "ok":       True,
            "filename": file.filename,
            "rows":     len(df),
            "cols":     len(df.columns),
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "all_cols": df.columns.tolist(),
            "preview":  df.head(5).to_dict(orient="records"),
        })
    except FileValidationError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        current_app.logger.error(f"Upload error: {e}")
        return jsonify({"error": "Error interno."}), 500


@upload_bp.route("/sample", methods=["GET"])
def download_sample():
    path = os.path.abspath(
        os.path.join(current_app.root_path, "..", "data", "samples", "entregas_ejemplo.csv")
    )
    if not os.path.exists(path):
        return jsonify({"error": "Archivo no encontrado."}), 404
    return send_file(path, as_attachment=True, download_name="entregas_ejemplo.csv")