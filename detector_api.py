from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import cv2
import numpy as np
from ultralytics import YOLO
import uvicorn
import json
import os
import time
from datetime import datetime
from collections import deque

# ======================================================
# KONFIGURASI
# ======================================================

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")

IMG_SIZE = int(os.getenv("IMG_SIZE", "416"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.60"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "1"))

LOG_FILE = "detections_history.json"
MAX_LOG_ITEMS = 300

# Untuk mengurangi log duplikat
last_logged_label = None
last_logged_time = 0
LOG_INTERVAL_SECONDS = 1.5

# Buffer sederhana untuk debugging
recent_predictions = deque(maxlen=20)

# ======================================================
# LOAD MODEL
# ======================================================

try:
    print("[INFO] Sedang memuat model...")
    model = YOLO(MODEL_PATH)

    # Warm up agar request pertama tidak terlalu lambat
    dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    model.predict(
        dummy,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        max_det=MAX_DETECTIONS,
        verbose=False
    )

    print(f"[INFO] Model berhasil dimuat.")
    print(f"[INFO] Total kelas: {len(model.names)}")
    print(f"[INFO] Classes: {model.names}")

except Exception as e:
    print(f"[ERROR] Gagal memuat model: {e}")
    model = None


# ======================================================
# FASTAPI APP
# ======================================================

app = FastAPI(
    title="BISINDO Gesture Detector API",
    description="API deteksi gestur BISINDO menggunakan YOLO dan FastAPI",
    version="1.2.0"
)

# Untuk lokal boleh *, nanti pas deploy sebaiknya diganti domain frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ======================================================
# HELPER FUNCTIONS
# ======================================================

def read_image(image_bytes: bytes):
    """
    Mengubah bytes gambar dari frontend menjadi format OpenCV.
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def is_valid_bbox(bbox, frame_w, frame_h):
    """
    Filter bbox agar deteksi kecil/noise tidak ikut dianggap valid.
    """
    x1, y1, x2, y2 = bbox

    if x1 >= x2 or y1 >= y2:
        return False

    if x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h:
        return False

    bbox_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_w * frame_h

    # Minimal area bbox 1.5% dari frame
    if bbox_area < frame_area * 0.015:
        return False

    return True


def save_to_json(label, confidence, bbox, inference_ms):
    """
    Menyimpan hasil deteksi yang stabil/kuat ke JSON.
    Tidak menyimpan semua frame agar tidak berat.
    """
    global last_logged_label, last_logged_time

    now = time.time()

    # Jangan spam log label yang sama terlalu cepat
    if label == last_logged_label and (now - last_logged_time) < LOG_INTERVAL_SECONDS:
        return

    last_logged_label = label
    last_logged_time = now

    history = []

    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    entry = {
        "timestamp": datetime.now().isoformat(),
        "label": label,
        "confidence": confidence,
        "bbox": bbox,
        "inference_ms": inference_ms
    }

    history.append(entry)

    if len(history) > MAX_LOG_ITEMS:
        history = history[-MAX_LOG_ITEMS:]

    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan log JSON: {e}")


def run_inference(img):
    """
    Menjalankan YOLO inference dan mengembalikan hasil deteksi yang sudah difilter.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model tidak tersedia di server")

    frame_h, frame_w = img.shape[:2]

    start_time = time.perf_counter()

    results = model.predict(
        img,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        max_det=MAX_DETECTIONS,
        verbose=False
    )

    inference_ms = round((time.perf_counter() - start_time) * 1000, 2)

    detections = []

    if not results:
        return detections, inference_ms

    r = results[0]

    if r.boxes is None:
        return detections, inference_ms

    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf_val = round(float(box.conf[0]), 4)
        bbox = [int(x) for x in box.xyxy[0].tolist()]

        if conf_val < CONF_THRESHOLD:
            continue

        if not is_valid_bbox(bbox, frame_w, frame_h):
            continue

        detections.append({
            "label": model.names[cls_id],
            "confidence": conf_val,
            "bbox": bbox
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)

    return detections, inference_ms


# ======================================================
# ENDPOINTS
# ======================================================

@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_ready": model is not None,
        "model_path": MODEL_PATH,
        "img_size": IMG_SIZE,
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "max_detections": MAX_DETECTIONS
    }


@app.get("/ping")
async def ping():
    return {
        "ok": True,
        "message": "API reachable",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", summary="Deteksi Gestur dari Gambar")
async def predict(image: UploadFile = File(...)):
    if model is None:
        return {
            "ok": False,
            "error": "Model tidak tersedia di server",
            "detections": []
        }

    content = await image.read()
    img = read_image(content)

    if img is None:
        return {
            "ok": False,
            "error": "Format gambar tidak valid",
            "detections": []
        }

    try:
        detections, inference_ms = run_inference(img)

        if detections:
            top = detections[0]

            recent_predictions.append({
                "label": top["label"],
                "confidence": top["confidence"],
                "time": datetime.now().strftime("%H:%M:%S")
            })

            # Simpan hanya prediksi paling kuat
            save_to_json(
                label=top["label"],
                confidence=top["confidence"],
                bbox=top["bbox"],
                inference_ms=inference_ms
            )

        return {
            "ok": True,
            "count": len(detections),
            "detections": detections,
            "inference_ms": inference_ms
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "detections": []
        }


@app.get("/stats")
async def stats():
    return {
        "model_ready": model is not None,
        "recent_predictions": list(recent_predictions),
        "config": {
            "model_path": MODEL_PATH,
            "img_size": IMG_SIZE,
            "conf_threshold": CONF_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "max_detections": MAX_DETECTIONS
        }
    }


# ======================================================
# RUN LOCAL
# ======================================================

if __name__ == "__main__":
    uvicorn.run(
        "detector_api:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )