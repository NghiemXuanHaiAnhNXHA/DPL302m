import os
import cv2
import numpy as np
import base64
import time
import shutil
import sqlite3
import math
import folium
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

# Initialize FastAPI app
app = FastAPI(title="Pothole Detection AI Web Portal")

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOADS_DIR = os.path.join(STATIC_DIR, "uploads")
OUTPUTS_DIR = os.path.join(STATIC_DIR, "outputs")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

# Ensure directories exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Load YOLO model
weights_path = os.path.join(BASE_DIR, "best.pt")
if not os.path.exists(weights_path):
    # Fallback to absolute path from original load.py if needed
    weights_path = r"D:\học liệu\summer 2026\dpl\pothole\best.pt"

print(f"Loading YOLO model from {weights_path}...")
try:
    model = YOLO(weights_path)
    print("YOLO Model loaded successfully.")
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    model = None

# Mount static files and images directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

# SQLite Database Setup
DB_PATH = os.path.join(BASE_DIR, "potholes.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS potholes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL,
            longitude REAL,
            image_path TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Haversine formula to compute distance between coordinates in meters
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

# Request model for webcam frame
class FrameData(BaseModel):
    image: str  # Base64 encoded JPEG image
    lat: float = None
    lng: float = None

@app.get("/")
async def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h2>Frontend is being generated. Please refresh in a moment.</h2>")

@app.post("/api/detect-frame")
async def detect_frame(data: FrameData):
    if model is None:
        raise HTTPException(status_code=500, detail="YOLO Model is not loaded on server.")
        
    try:
        # Decode base64 image
        header, encoded = data.image.split(",", 1) if "," in data.image else ("", data.image)
        img_data = base64.b64decode(encoded)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image frame.")
            
        # Run YOLO inference
        results = model(frame, imgsz=640, conf=0.6, verbose=False)
        
        # Plot predictions on frame
        annotated_frame = results[0].plot()
        
        # Encode annotated frame back to base64
        _, buffer = cv2.imencode(".jpg", annotated_frame)
        encoded_img = base64.b64encode(buffer).decode("utf-8")
        
        # Count detected potholes
        pothole_count = 0
        if len(results) > 0 and results[0].boxes is not None:
            pothole_count = len(results[0].boxes)
            
        # If potholes detected and valid GPS coordinates provided
        if pothole_count > 0 and data.lat is not None and data.lng is not None:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT latitude, longitude FROM potholes")
            existing_potholes = cursor.fetchall()
            
            is_duplicate = False
            # Check if any existing pothole is within 20 meters
            for ex_lat, ex_lng in existing_potholes:
                dist = haversine_distance(data.lat, data.lng, ex_lat, ex_lng)
                if dist <= 100.0:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                # Save snapshot to images folder: images/lat_lng.jpg
                existing_numbers = []

                for f in os.listdir(IMAGES_DIR):
                    name, ext = os.path.splitext(f)

                    if name.isdigit():
                        existing_numbers.append(int(name))

                # Save snapshot to images folder: images/lat_lng.jpg
                filename = f"{int(time.time() * 1000)}.jpg"
                file_path = os.path.join(IMAGES_DIR, filename)

                cv2.imwrite(file_path, annotated_frame)
                            
                # Insert to SQLite
                web_path = f"/images/{filename}"
                timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute(
                    "INSERT INTO potholes (latitude, longitude, image_path, timestamp) VALUES (?, ?, ?, ?)",
                    (data.lat, data.lng, web_path, timestamp_str)
                )
                conn.commit()
                print(f"[SQLITE] Saved new pothole: ({data.lat}, {data.lng}) -> {web_path}")
                
            conn.close()
            
        return {
            "image": f"data:image/jpeg;base64,{encoded_img}",
            "count": pothole_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/map")
async def get_map():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT latitude, longitude, image_path, timestamp FROM potholes")
        rows = cursor.fetchall()
        conn.close()
        
        # Center map
        if rows:
            center_lat = sum(r[0] for r in rows) / len(rows)
            center_lon = sum(r[1] for r in rows) / len(rows)
            zoom = 16
        else:
            center_lat, center_lon = 21.0285, 105.8542  # Hanoi Center
            zoom = 12
            
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)
        
        # Add markers
        for lat, lon, img_path, timestamp in rows:
            popup_html = f"""
            <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; width: 220px; color: #1e293b; line-height: 1.4;">
                <h4 style="margin: 0 0 6px 0; color: #ef4444; font-size: 13px; font-weight: 700; display: flex; align-items: center; gap: 4px;">
                    🚨 Phát Hiện Ổ Gà
                </h4>
                <p style="margin: 0 0 4px 0; font-size: 11px; color: #475569;">
                    <b>Vị trí:</b> {lat:.6f}, {lon:.6f}
                </p>
                <p style="margin: 0 0 8px 0; font-size: 11px; color: #475569;">
                    <b>Thời gian:</b> {timestamp}
                </p>
                <div style="width: 100%; border-radius: 6px; overflow: hidden; border: 1px solid #cbd5e1; box-shadow: 0 4px 10px rgba(0,0,0,0.15);">
                    <img src="{img_path}?t={int(time.time())}" style="width: 100%; height: auto; display: block;" />
                </div>
            </div>
            """
            
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=250),
                icon=folium.Icon(color="red", icon="warning-sign")
            ).add_to(m)
            
        # Wrap Folium HTML to prevent height collapse inside the iframe
        html_wrapper = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                html, body {{
                    width: 100%;
                    height: 100%;
                    margin: 0;
                    padding: 0;
                    overflow: hidden;
                }}
            </style>
        </head>
        <body>
            {m._repr_html_()}
        </body>
        </html>
        """
        return HTMLResponse(content=html_wrapper, status_code=200)
    except Exception as e:
        return HTMLResponse(content=f"<h3>Lỗi tạo bản đồ: {str(e)}</h3>", status_code=500)

@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="YOLO Model is not loaded on server.")
        
    # Save the uploaded file
    filename = f"{int(time.time())}_{file.filename}"
    input_path = os.path.join(UPLOADS_DIR, filename)
    output_filename = f"processed_{filename}"
    output_path = os.path.join(OUTPUTS_DIR, output_filename)
    
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Open video with OpenCV
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file.")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 100:
            fps = 30.0
            
        # Set up VideoWriter with browser-compatible H.264 if possible
        # Try 'avc1' first, fall back to 'mp4v'
        fourcc_codecs = ['avc1', 'mp4v']
        writer = None
        
        for codec in fourcc_codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if writer.isOpened():
                    break
            except Exception:
                continue
                
        if writer is None or not writer.isOpened():
            # Ultimate fallback to standard AVI format or XVID
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_filename = output_filename.rsplit(".", 1)[0] + ".avi"
            output_path = os.path.join(OUTPUTS_DIR, output_filename)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        if not writer.isOpened():
            raise HTTPException(status_code=500, detail="Failed to initialize video writer.")
            
        potholes_detected_total = 0
        frames_processed = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run YOLO
            results = model(frame, imgsz=640, conf=0.6, verbose=False)
            
            # Plot predictions
            annotated_frame = results[0].plot()
            
            # Write frame to output video
            writer.write(annotated_frame)
            
            # Keep track of detections
            if len(results) > 0 and results[0].boxes is not None:
                potholes_detected_total += len(results[0].boxes)
                
            frames_processed += 1
            
        cap.release()
        writer.release()
        
        # Clean up input file to save disk space
        if os.path.exists(input_path):
            os.remove(input_path)
            
        return {
            "video_url": f"/static/outputs/{output_filename}",
            "frames_processed": frames_processed,
            "total_detections": potholes_detected_total
        }
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(input_path):
            os.remove(input_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
