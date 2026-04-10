from __future__ import annotations

from flask import Flask, jsonify, request
from flask_cors import CORS
from app.service import VerificationService

app = Flask(__name__)
CORS(app)
service = VerificationService()


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/verify")
def verify():
    payload = request.get_json(force=True)
    required = ["subject_id", "enrollment", "probe"]
    for k in required:
        if k not in payload:
            return jsonify({"error": f"Missing field: {k}"}), 400

    result = service.verify(
        enrollment=payload["enrollment"],
        probe=payload["probe"],
        claimed_subject_id=payload["subject_id"],
    )
    return jsonify(result)


@app.post("/verify/face")
def verify_face():
    payload = request.get_json(force=True)
    result = service.verify_face(
        enrollment_path=payload["enrollment"],
        probe_path=payload["probe"],
        claimed_subject_id=payload.get("subject_id", "unknown"),
    )
    return jsonify(result)


@app.post("/verify/fingerprint")
def verify_fingerprint():
    payload = request.get_json(force=True)
    result = service.verify_fingerprint(
        enrollment_path=payload["enrollment"],
        probe_path=payload["probe"],
        claimed_subject_id=payload.get("subject_id", "unknown"),
    )
    return jsonify(result)


@app.post("/verify/palm")
def verify_palm():
    payload = request.get_json(force=True)
    result = service.verify_palm(
        enrollment_path=payload["enrollment"],
        probe_path=payload["probe"],
        claimed_subject_id=payload.get("subject_id", "unknown"),
    )
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
