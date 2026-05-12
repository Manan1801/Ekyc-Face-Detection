"""
AI-Based eKYC + Face Verification System
Improved Backend Version
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import hashlib
import datetime
import uuid
import mediapipe as mp

# ─────────────────────────────────────────────
# DeepFace Import
# ─────────────────────────────────────────────
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace loaded successfully")
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("⚠️ DeepFace not available — using OpenCV fallback")

# ─────────────────────────────────────────────
# Flask App
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

sessions = {}

# ─────────────────────────────────────────────
# MediaPipe Setup
# ─────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh

# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def decode_base64_image(b64_string):
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]

    img_bytes = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_bytes, np.uint8)

    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def encode_image_to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


# ─────────────────────────────────────────────
# Improved Face Extraction
# ─────────────────────────────────────────────
def extract_face(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None, None

    # largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # More padding improves embeddings
    padding = int(0.35 * w)

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)

    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)

    face_crop = image[y1:y2, x1:x2]

    # Resize for consistency
    face_crop = cv2.resize(face_crop, (224, 224))

    return face_crop, (x1, y1, x2, y2)


# ─────────────────────────────────────────────
# SHA256 Verification Hash
# ─────────────────────────────────────────────
def generate_verification_hash(session_id, match_score, timestamp):
    data = f"{session_id}:{match_score}:{timestamp}"
    return hashlib.sha256(data.encode()).hexdigest()


# ─────────────────────────────────────────────
# Fallback Comparison
# ─────────────────────────────────────────────
def fallback_face_compare(img1, img2):

    face1, _ = extract_face(img1)
    face2, _ = extract_face(img2)

    if face1 is None or face2 is None:
        return 0.0, "No Face"

    face1 = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
    face2 = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)

    hist1 = cv2.calcHist([face1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([face2], [0], None, [256], [0, 256])

    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    similarity = max(0, min(100, score * 100))

    return similarity, "OpenCV Histogram"


# ─────────────────────────────────────────────
# Eye Aspect Ratio
# ─────────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices, image_w, image_h):

    points = []

    for idx in eye_indices:
        lm = landmarks[idx]
        points.append((lm.x * image_w, lm.y * image_h))

    A = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    B = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    C = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    ear = (A + B) / (2.0 * C) if C != 0 else 0

    return ear


# ─────────────────────────────────────────────
# Upload ID
# ─────────────────────────────────────────────
@app.route("/upload-id", methods=["POST"])
def upload_id():

    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        id_image = decode_base64_image(data["image"])

        if id_image is None:
            return jsonify({"error": "Invalid image"}), 400

        face_crop, bbox = extract_face(id_image)

        if face_crop is None:
            return jsonify({
                "success": False,
                "error": "No face detected in ID"
            }), 400

        session_id = str(uuid.uuid4())

        id_path = os.path.join(
            UPLOAD_FOLDER,
            f"{session_id}_id.jpg"
        )

        face_path = os.path.join(
            UPLOAD_FOLDER,
            f"{session_id}_face.jpg"
        )

        cv2.imwrite(id_path, id_image)
        cv2.imwrite(face_path, face_crop)

        sessions[session_id] = {
            "id_image": id_path,
            "id_face": face_path,
            "liveness_passed": False,
            "created_at": datetime.datetime.now().isoformat()
        }

        annotated = id_image.copy()

        if bbox:
            x1, y1, x2, y2 = bbox

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 100),
                3
            )

        return jsonify({
            "success": True,
            "session_id": session_id,
            "face_image": encode_image_to_base64(face_crop),
            "annotated_id": encode_image_to_base64(annotated),
            "message": "Face extracted successfully"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Liveness Detection
# ─────────────────────────────────────────────
@app.route("/liveness-check", methods=["POST"])
def liveness_check():

    try:
        data = request.get_json()

        session_id = data.get("session_id")
        frames = data.get("frames", [])

        if not frames:
            return jsonify({"error": "No frames"}), 400

        LEFT_EYE = [362, 385, 387, 263, 373, 380]
        RIGHT_EYE = [33, 160, 158, 133, 153, 144]

        EAR_THRESHOLD = 0.22

        blink_count = 0
        min_ear = 1.0

        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:

            for frame_b64 in frames:

                frame = decode_base64_image(frame_b64)

                if frame is None:
                    continue

                h, w = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:

                    lms = results.multi_face_landmarks[0].landmark

                    left_ear = eye_aspect_ratio(
                        lms,
                        LEFT_EYE,
                        w,
                        h
                    )

                    right_ear = eye_aspect_ratio(
                        lms,
                        RIGHT_EYE,
                        w,
                        h
                    )

                    avg_ear = (left_ear + right_ear) / 2.0

                    min_ear = min(min_ear, avg_ear)

                    if avg_ear < EAR_THRESHOLD:
                        blink_count += 1

        liveness_passed = blink_count >= 1

        if session_id in sessions:
            sessions[session_id]["liveness_passed"] = liveness_passed

        return jsonify({
            "success": True,
            "liveness_passed": liveness_passed,
            "blink_frames": blink_count,
            "min_ear": round(min_ear, 4),
            "message": "Liveness PASSED"
            if liveness_passed
            else "Liveness FAILED"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Face Verification
# ─────────────────────────────────────────────
@app.route("/verify-face", methods=["POST"])
def verify_face():

    try:
        data = request.get_json()

        session_id = data.get("session_id")
        selfie_b64 = data.get("selfie")

        if session_id not in sessions:
            return jsonify({
                "error": "Invalid session"
            }), 400

        session = sessions[session_id]

        if not session.get("liveness_passed"):
            return jsonify({
                "error": "Liveness not completed"
            }), 400

        selfie_img = decode_base64_image(selfie_b64)

        if selfie_img is None:
            return jsonify({
                "error": "Invalid selfie"
            }), 400

        selfie_face, _ = extract_face(selfie_img)

        if selfie_face is None:
            return jsonify({
                "error": "No face detected in selfie"
            }), 400

        selfie_path = os.path.join(
            UPLOAD_FOLDER,
            f"{session_id}_selfie.jpg"
        )

        selfie_face_path = os.path.join(
            UPLOAD_FOLDER,
            f"{session_id}_selfie_face.jpg"
        )

        cv2.imwrite(selfie_path, selfie_img)
        cv2.imwrite(selfie_face_path, selfie_face)

        id_face_path = session["id_face"]

        # ─────────────────────────
        # DeepFace Verification
        # ─────────────────────────
        match_score = 0.0
        verified = False
        method_used = ""

        if DEEPFACE_AVAILABLE:

            try:
                result = DeepFace.verify(
                    img1_path=id_face_path,
                    img2_path=selfie_face_path,
                    model_name="ArcFace",
                    detector_backend="retinaface",
                    enforce_detection=False
                )

                distance = float(result.get("distance", 1.0))

                # Better score conversion
                match_score = max(
    0,
    min(100, (1 - distance) * 100 + 15)
)

                match_score = round(match_score, 2)

                # Better threshold
                verified = distance < 0.45

                method_used = "DeepFace ArcFace + RetinaFace"

                print("\n──────── VERIFICATION ────────")
                print(f"Distance     : {distance}")
                print(f"Match Score  : {match_score}")
                print(f"Verified     : {verified}")
                print("──────────────────────────────\n")

            except Exception as df_error:

                print("DeepFace Error:", df_error)

                id_img = cv2.imread(id_face_path)

                match_score, method_used = fallback_face_compare(
                    id_img,
                    selfie_img
                )

                verified = match_score >= 70

        else:

            id_img = cv2.imread(id_face_path)

            match_score, method_used = fallback_face_compare(
                id_img,
                selfie_img
            )

            verified = match_score >= 70

        # ─────────────────────────
        # Verification Hash
        # ─────────────────────────
        timestamp = datetime.datetime.now().isoformat()

        verification_hash = generate_verification_hash(
            session_id,
            match_score,
            timestamp
        )

        result_data = {
            "session_id": session_id,
            "verified": verified,
            "match_score": round(match_score, 2),
            "method": method_used,
            "timestamp": timestamp,
            "verification_hash": verification_hash,
            "liveness_passed": True,
            "selfie_face": encode_image_to_base64(selfie_face)
        }

        sessions[session_id]["verification_result"] = result_data

        return jsonify({
            "success": True,
            **result_data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# Get Result
# ─────────────────────────────────────────────
@app.route("/result/<session_id>", methods=["GET"])
def get_result(session_id):

    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    return jsonify({
        "success": True,
        "result": sessions[session_id].get(
            "verification_result"
        )
    })


# ─────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "running",
        "deepface_available": DEEPFACE_AVAILABLE,
        "version": "2.0"
    })


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)