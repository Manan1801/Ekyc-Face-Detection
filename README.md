# 🔐 AI-Based eKYC + Face Verification System

> A prototype AI-powered identity verification workflow demonstrating face matching, liveness detection, and blockchain-style audit trails — built for the IDS interview.

---

## 🎯 What This Project Does

```
Upload ID Card
    → OpenCV extracts face from ID
    → MediaPipe liveness check (blink detection)
    → Capture live selfie
    → DeepFace compares ID face vs selfie
    → SHA-256 verification hash (blockchain simulation)
    → Final verified / rejected result
```

---

## 🏗️ Architecture

```
Frontend (HTML + Vanilla JS)
         ↕  REST API (JSON)
Flask Backend (Python)
         ↕
    ┌────────────────────────────────┐
    │  OpenCV       – Face Detection  │
    │  DeepFace     – Face Matching   │
    │  MediaPipe    – Liveness (EAR) │
    │  hashlib      – SHA-256 Hash   │
    └────────────────────────────────┘
```

---

## 📁 Folder Structure

```
ekyc-project/
├── backend/
│   ├── app.py              ← Flask API (all routes)
│   └── requirements.txt    ← Python dependencies
├── frontend/
│   └── index.html          ← Complete UI (single file)
├── uploads/                ← Stored ID/selfie images (auto-created)
└── README.md
```

---

## ⚡ Quick Setup (5 Minutes)

### Prerequisites
- Python 3.9 or 3.10 (recommended)
- pip
- Modern browser (Chrome/Firefox)

---

### Step 1 — Clone / Download
```bash
# If using git:
git clone <your-repo-url>
cd ekyc-project

# Or just unzip the folder and cd into it
```

### Step 2 — Install Python Dependencies
```bash
cd backend

# Option A: with DeepFace (full AI, takes ~5 min to install)
pip install flask flask-cors opencv-python mediapipe numpy pillow deepface tf-keras

# Option B: minimal (faster, uses OpenCV fallback)
pip install flask flask-cors opencv-python mediapipe numpy pillow
```

> 💡 DeepFace downloads ~500MB of model weights on first run. It's worth it for accuracy.

### Step 3 — Start Flask Backend
```bash
# From inside backend/ folder:
python app.py
```
You should see:
```
🚀 eKYC Backend Starting...
   DeepFace: ✅ Available
   Server: http://localhost:5000
```

### Step 4 — Open the Frontend
```bash
# Simply open in browser — no server needed:
open ../frontend/index.html

# Or on Windows:
start ../frontend/index.html
```

> 🔑 The frontend talks to `http://localhost:5000` — make sure Flask is running first.

---

## 🔌 API Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `GET  /health` | GET | Check backend status |
| `POST /upload-id` | POST | Upload ID image, extract face |
| `POST /liveness-check` | POST | Analyze frames for blink detection |
| `POST /verify-face` | POST | Compare selfie vs ID face |
| `GET  /result/<session_id>` | GET | Get stored result |

### Example: Upload ID
```json
POST /upload-id
{
  "image": "<base64-encoded image>"
}

Response:
{
  "success": true,
  "session_id": "uuid-here",
  "face_detected": true,
  "face_image": "<base64 cropped face>",
  "annotated_id": "<base64 with bbox>"
}
```

### Example: Verify Face
```json
POST /verify-face
{
  "session_id": "uuid-here",
  "selfie": "<base64 selfie>"
}

Response:
{
  "success": true,
  "verified": true,
  "match_score": 87.3,
  "method": "DeepFace (VGG-Face Neural Network)",
  "verification_hash": "sha256-hash",
  "timestamp": "2024-01-15T10:30:00"
}
```

---

## 🧠 Technical Deep Dive

### Face Extraction (OpenCV)
```python
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
```
Haar Cascade is a classical ML algorithm trained to detect face-shaped patterns using contrast differences across rectangular regions.

### Liveness Detection (MediaPipe + EAR)
```
Eye Aspect Ratio = (vertical_distances) / (2 × horizontal_distance)
EAR < 0.22  →  eye is closed  →  blink detected
```
MediaPipe provides 468 3D facial landmarks. We use 6 points around each eye to compute EAR across 10 captured frames.

### Face Verification (DeepFace)
```python
result = DeepFace.verify(img1_path, img2_path, model_name='VGG-Face')
# Returns: distance, verified (bool)
```
VGG-Face is a deep CNN trained on 2.6M face images. It produces a 4096-dimensional embedding vector. We compare vectors using cosine similarity.

### Blockchain Simulation
```python
data = f"{session_id}:{match_score}:{timestamp}"
hash_value = hashlib.sha256(data.encode()).hexdigest()
```
In production: this hash would be written to Hyperledger Fabric or a private Ethereum chain as an immutable audit record.

---

## 🔒 Security Features Demonstrated

| Feature | Technique |
|---------|-----------|
| Anti-Spoofing | Eye Aspect Ratio blink detection |
| Face Matching | Deep neural network embeddings |
| Session Isolation | UUID-based sessions |
| Audit Trail | SHA-256 verification hash |
| No Data Leakage | Images stored server-side only |

---

## 🚀 Future Scope (Mention in Interview)

- **Decentralized Identity**: Store verification records on Hyperledger Fabric
- **Advanced Liveness**: 3D depth maps, IR cameras, head pose estimation
- **OCR Integration**: Extract Aadhaar/PAN number from ID automatically
- **Mobile SDK**: React Native app with native camera pipeline
- **Zero-Knowledge Proofs**: Verify identity without revealing raw data
- **Facial Age Estimation**: Ensure user matches age on ID

---

## 🎤 Interview Talking Points

### Project Introduction
> *"After researching IDS's focus on digital identity and eKYC infrastructure, I built this prototype to explore the domain hands-on. It demonstrates the core components of a real-world identity verification pipeline."*

### On Face Verification
> *"I used DeepFace with the VGG-Face model, which produces high-dimensional face embeddings and computes cosine similarity between them. The threshold I set was 60% for a demo environment — production systems typically use stricter thresholds with additional contextual checks."*

### On Liveness Detection
> *"The liveness check uses MediaPipe's 468-point face mesh to compute Eye Aspect Ratio across multiple frames. When EAR drops below 0.22, it indicates a blink — proving the subject is alive, not a static photograph. This prevents basic photo-spoofing attacks."*

### On Blockchain Simulation
> *"I simulate the audit trail using SHA-256 hashing of the session ID, match score, and timestamp. In an enterprise deployment, this hash would be written to an immutable ledger like Hyperledger Fabric, ensuring the verification record cannot be tampered with retroactively."*

### On Architecture
> *"The system is designed with separation of concerns — the Flask backend handles all AI inference and session management, while the frontend is a pure HTML/JS client that communicates via REST APIs. This makes it straightforward to swap the frontend for a React Native mobile app or integrate the backend into an existing enterprise system."*

---

## ⚠️ Known Limitations (Be Honest in Interview)

- Liveness detection is basic — production systems use multi-modal approaches
- Sessions are in-memory, not persistent (would use Redis/PostgreSQL in production)
- No encryption of stored images (would use AES-256 in production)
- Histogram fallback is not as accurate as deep learning

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend Framework | Flask (Python) |
| Face Detection | OpenCV Haar Cascade |
| Face Verification | DeepFace (VGG-Face) |
| Liveness Detection | MediaPipe Face Mesh |
| Blockchain Sim | hashlib SHA-256 |
| Frontend | HTML5 + CSS3 + Vanilla JS |
| Camera API | WebRTC getUserMedia |

---

*Built as a proof-of-concept for IDS interview — demonstrating AI, Computer Vision, Security Thinking, and Digital Identity workflows.*
