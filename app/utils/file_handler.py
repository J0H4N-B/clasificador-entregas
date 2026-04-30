"""
app/utils/file_handler.py — Manejo seguro de archivos CSV
"""
import os, uuid
import pandas as pd
from flask import current_app
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage


class FileValidationError(Exception):
    pass


def allowed_file(filename: str) -> bool:
    allowed = current_app.config.get("ALLOWED_EXTENSIONS", {"csv"})
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def save_uploaded_file(file: FileStorage) -> str:
    if not file or file.filename == "":
        raise FileValidationError("No se recibió ningún archivo.")
    if not allowed_file(file.filename):
        raise FileValidationError("Solo se aceptan archivos .csv")
    safe     = secure_filename(file.filename)
    unique   = f"{uuid.uuid4().hex}_{safe}"
    filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], unique)
    file.save(filepath)
    return filepath


def read_csv_safe(filepath: str, max_rows: int = 50_000) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileValidationError("Archivo no encontrado.")
    df = None
    for enc in ("utf-8", "latin-1", "utf-8-sig"):
        try:
            df = pd.read_csv(filepath, nrows=max_rows, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise FileValidationError(f"Error al leer el CSV: {e}")
    if df is None:
        raise FileValidationError("No se pudo decodificar el CSV.")
    if df.empty:
        raise FileValidationError("El CSV está vacío.")
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(" ", "_").str.replace(r"[^\w]", "", regex=True))
    return df


def cleanup_old_uploads(folder: str, max_files: int = 50) -> None:
    try:
        files = sorted(
            [os.path.join(folder, f) for f in os.listdir(folder)
             if os.path.isfile(os.path.join(folder, f))],
            key=os.path.getctime)
        while len(files) > max_files:
            os.remove(files.pop(0))
    except Exception:
        pass