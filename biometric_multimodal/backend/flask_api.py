from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from flask import Flask, request, jsonify
from flask_cors import CORS
from app.service import AuthService
from app.config import load_settings

settings = load_settings("configs/settings.yaml")
app = Flask(__name__)
CORS(app)
service = AuthService("configs/settings.yaml")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/verify", methods=["POST"])
def verify():
    payload = request.get_json(force=True)
    subject_id = payload.get("subject_id")
    face_path = payload.get("face_path")
    fingerprint_path = payload.get("fingerprint_path")
    palm_path = payload.get("palm_path")

    if not subject_id:
        return jsonify({"error": "subject_id is required"}), 400

    try:
        result = service.verify(subject_id, face_path, fingerprint_path, palm_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host=settings["server"]["host"], port=settings["server"]["port"], debug=True)
