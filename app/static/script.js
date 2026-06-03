// Global State
let stream = null;
let cameraActive = false;
let frameInterval = null;
let lastFrameTime = Date.now();
let fpsArray = [];
let userLatitude = null;
let userLongitude = null;
let geoWatchId = null;
let isGpsSimulated = false;

// DOM Elements
const dropZone = document.getElementById("drop-zone");
const videoInput = document.getElementById("video-input");
const processingCard = document.getElementById("processing-card");
const resultsZone = document.getElementById("results-zone");
const actualResult = document.getElementById("actual-result");
const placeholderResult = document.getElementById("placeholder-result");
const outputVideo = document.getElementById("output-video");
const downloadBtn = document.getElementById("download-btn");
const progressBar = document.getElementById("progress-bar");
const progressPercent = document.getElementById("progress-percent");
const statusTitle = document.getElementById("status-title");
const statusDesc = document.getElementById("status-desc");

const statDetections = document.getElementById("stat-detections");
const statFrames = document.getElementById("stat-frames");

const webcam = document.getElementById("webcam");
const outputCanvas = document.getElementById("output-canvas");
const canvasCtx = outputCanvas.getContext("2d");
const viewportPlaceholder = document.getElementById("viewport-placeholder");
const startCamBtn = document.getElementById("start-cam-btn");
const camStatusIndicator = document.getElementById("cam-status-indicator");
const fpsVal = document.getElementById("fps-val");
const liveCountVal = document.getElementById("live-count-val");

// Switch Tabs
function switchTab(tab) {
    document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".content-panel").forEach(panel => panel.classList.remove("active"));

    if (tab === "video") {
        document.getElementById("tab-video").classList.add("active");
        document.getElementById("panel-video").classList.add("active");
        if (cameraActive) stopCamera();
    } else if (tab === "camera") {
        document.getElementById("tab-camera").classList.add("active");
        document.getElementById("panel-camera").classList.add("active");
    } else if (tab === "map") {
        document.getElementById("tab-map").classList.add("active");
        document.getElementById("panel-map").classList.add("active");

        if (cameraActive) stopCamera();

        setTimeout(() => {
            document.getElementById("map-iframe").src =
                "/api/map?t=" + Date.now();
        }, 300);
    }
}

// DRAG & DROP FOR VIDEO
dropZone.addEventListener("click", () => videoInput.click());

videoInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        uploadVideo(e.target.files[0]);
    }
});

["dragenter", "dragover"].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    }, false);
});

["dragleave", "drop"].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
    }, false);
});

dropZone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0 && files[0].type.startsWith("video/")) {
        uploadVideo(files[0]);
    } else {
        alert("Vui lòng tải lên tệp tin video hợp lệ.");
    }
});

// Update circular progress bar svg
function updateProgress(percent) {
    const radius = 45;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (percent / 100) * circumference;

    progressBar.style.strokeDasharray = circumference;
    progressBar.style.strokeDashoffset = offset;
    progressPercent.innerText = `${Math.round(percent)}%`;
}

// Upload & Process Video
function uploadVideo(file) {
    // UI state transitions
    dropZone.classList.add("hidden");
    processingCard.classList.remove("hidden");
    placeholderResult.classList.remove("hidden");
    actualResult.classList.add("hidden");

    updateProgress(0);
    statusTitle.innerText = "Đang tải video lên...";
    statusDesc.innerText = "Chuẩn bị gửi tệp tin tới máy chủ AI để xử lý.";

    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append("file", file);

    // Track upload progress
    xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable) {
            const percent = (e.loaded / e.total) * 100;
            // Cap at 95% during upload. The remaining 5% is the backend processing.
            const displayPercent = percent * 0.9;
            updateProgress(displayPercent);

            if (percent >= 99) {
                statusTitle.innerText = "Đang xử lý video bằng AI...";
                statusDesc.innerText = "Mô hình YOLO đang phát hiện và khoanh vùng ổ gà trong từng khung hình.";
            }
        }
    });

    xhr.onload = function () {
        if (xhr.status === 200) {
            updateProgress(100);
            const response = JSON.parse(xhr.responseText);

            // Show result
            processingCard.classList.add("hidden");
            dropZone.classList.remove("hidden");
            placeholderResult.classList.add("hidden");
            actualResult.classList.remove("hidden");

            // Set source
            outputVideo.src = response.video_url;
            outputVideo.load();
            outputVideo.play();

            // Set statistics
            statDetections.innerText = response.total_detections;
            statFrames.innerText = response.frames_processed;
            downloadBtn.href = response.video_url;
        } else {
            alert(`Lỗi xử lý video: ${xhr.statusText}`);
            resetUploadState();
        }
    };

    xhr.onerror = function () {
        alert("Lỗi kết nối tới máy chủ.");
        resetUploadState();
    };

    xhr.open("POST", "/api/detect-frame-video" || "/api/upload-video", true);
    xhr.send(formData);
}

function resetUploadState() {
    processingCard.classList.add("hidden");
    dropZone.classList.remove("hidden");
    placeholderResult.classList.remove("hidden");
    actualResult.classList.add("hidden");
}

// LIVE CAMERA REAL-TIME STREAMING
async function toggleCamera() {
    if (cameraActive) {
        stopCamera();
    } else {
        await startCamera();
    }
}

// Get available cameras
async function getCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        const cameraSelect = document.getElementById("camera-select");

        // Save current selection if any
        const currentSelection = cameraSelect.value;
        cameraSelect.innerHTML = "";

        if (videoDevices.length === 0) {
            const option = document.createElement("option");
            option.text = "Không tìm thấy camera";
            cameraSelect.appendChild(option);
            return;
        }

        videoDevices.forEach((device, index) => {
            const option = document.createElement("option");
            option.value = device.deviceId;
            option.text = device.label || `Camera ${index + 1}`;
            cameraSelect.appendChild(option);
        });

        // Restore selection if it still exists
        if (currentSelection && Array.from(cameraSelect.options).some(opt => opt.value === currentSelection)) {
            cameraSelect.value = currentSelection;
        }
    } catch (err) {
        console.error("Lỗi lấy danh sách camera: ", err);
    }
}

// Switch Camera when changed in selector
async function changeCamera() {
    if (cameraActive) {
        stopCamera();
        await startCamera();
    }
}

// Fetch cameras on load
document.addEventListener("DOMContentLoaded", () => {
    getCameras();
});

async function startCamera() {
    const cameraSelect = document.getElementById("camera-select");
    const selectedDeviceId = cameraSelect.value;

    const constraints = {
        video: {
            width: { ideal: 640 },
            height: { ideal: 480 }
        },
        audio: false
    };

    // If a specific camera is chosen, use its deviceId
    if (selectedDeviceId && selectedDeviceId !== "Không tìm thấy camera" && selectedDeviceId !== "") {
        constraints.video.deviceId = { exact: selectedDeviceId };
    } else {
        constraints.video.facingMode = "environment"; // Default fallback
    }

    try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);

        // Refresh camera names now that permission is granted (to get real labels instead of empty strings)
        await getCameras();

        // Set the active track in dropdown
        const activeTrack = stream.getVideoTracks()[0];
        const settings = activeTrack.getSettings();
        if (settings && settings.deviceId) {
            cameraSelect.value = settings.deviceId;
        }

        // Request GPS location
        if ("geolocation" in navigator) {
            geoWatchId = navigator.geolocation.watchPosition(
                (position) => {
                    userLatitude = position.coords.latitude;
                    userLongitude = position.coords.longitude;
                    isGpsSimulated = false;
                    console.log(`GPS Real: ${userLatitude}, ${userLongitude}`);
                },
                (err) => {
                    console.warn("GPS access denied or error. Simulating coordinates...", err);
                    // Mock coordinates centered at Hanoi with slight random walk
                    isGpsSimulated = true;
                    userLatitude = 21.0285 + (Math.random() - 0.5) * 0.002;
                    userLongitude = 105.8542 + (Math.random() - 0.5) * 0.002;
                },
                { enableHighAccuracy: true }
            );
        } else {
            console.warn("Geolocation not supported by browser. Simulating coordinates...");
            isGpsSimulated = true;
            userLatitude = 21.0285 + (Math.random() - 0.5) * 0.002;
            userLongitude = 105.8542 + (Math.random() - 0.5) * 0.002;
        }

        webcam.srcObject = stream;
        cameraActive = true;

        // UI updates
        viewportPlaceholder.classList.add("hidden");
        outputCanvas.style.display = "block";
        startCamBtn.innerHTML = '<i class="fa-solid fa-stop"></i> Dừng Camera';
        startCamBtn.className = "btn btn-danger";

        camStatusIndicator.className = "status-indicator online";
        camStatusIndicator.innerHTML = '<span class="pulse-dot"></span> Camera Đang Bật';

        // Wait for video load metadata to set canvas size
        webcam.onloadedmetadata = () => {
            outputCanvas.width = webcam.videoWidth;
            outputCanvas.height = webcam.videoHeight;
        };

        // Start processing frames
        lastFrameTime = Date.now();
        fpsArray = [];
        processFrameLoop();

    } catch (err) {
        console.error("Lỗi camera: ", err);
        alert("Không thể truy cập camera. Vui lòng cấp quyền camera trong cài đặt trình duyệt của bạn.");
        stopCamera();
    }
}

function stopCamera() {
    cameraActive = false;

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    // Stop geolocation tracking
    if (geoWatchId !== null) {
        navigator.geolocation.clearWatch(geoWatchId);
        geoWatchId = null;
    }
    userLatitude = null;
    userLongitude = null;
    isGpsSimulated = false;

    // UI resets
    viewportPlaceholder.classList.remove("hidden");
    outputCanvas.style.display = "none";
    startCamBtn.innerHTML = '<i class="fa-solid fa-play"></i> Bắt Đầu Camera';
    startCamBtn.className = "btn btn-success";

    camStatusIndicator.className = "status-indicator offline";
    camStatusIndicator.innerHTML = '<span class="pulse-dot"></span> Camera Tắt';

    fpsVal.innerText = "0";
    liveCountVal.innerText = "0";

    // Clear canvas
    canvasCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
}

// Capture frame and send to FastAPI for detection
async function processFrameLoop() {
    if (!cameraActive) return;

    try {
        // Create an offscreen canvas to capture frame
        const offscreenCanvas = document.createElement("canvas");
        offscreenCanvas.width = outputCanvas.width || 640;
        offscreenCanvas.height = outputCanvas.height || 480;
        const ctx = offscreenCanvas.getContext("2d");

        // Draw current webcam frame onto offscreen canvas
        ctx.drawImage(webcam, 0, 0, offscreenCanvas.width, offscreenCanvas.height);

        // Convert to base64 jpeg
        const base64Image = offscreenCanvas.toDataURL("image/jpeg", 0.7);

        // Send frame to FastAPI with coordinates if available
        const payload = { image: base64Image };
        if (userLatitude !== null && userLongitude !== null) {
            payload.lat = userLatitude;
            payload.lng = userLongitude;

            // Only jitter coordinates when using simulated (mock) GPS
            if (isGpsSimulated) {
                userLatitude += (Math.random() - 0.5) * 0.0001;
                userLongitude += (Math.random() - 0.5) * 0.0001;
            }
        }

        const response = await fetch("/api/detect-frame", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (response.ok && cameraActive) {
            const data = await response.json();

            // Render the returned base64 image on our canvas
            const img = new Image();
            img.onload = function () {
                canvasCtx.drawImage(img, 0, 0, outputCanvas.width, outputCanvas.height);
            };
            img.src = data.image;

            // Update UI metrics
            liveCountVal.innerText = data.count;
            if (data.count > 0) {
                document.querySelector(".alert-tag").style.borderColor = "var(--danger)";
                document.querySelector(".alert-tag").style.color = "var(--danger)";
            } else {
                document.querySelector(".alert-tag").style.borderColor = "var(--border-color)";
                document.querySelector(".alert-tag").style.color = "var(--text-primary)";
            }

            // Calculate FPS
            const now = Date.now();
            const fps = 1000 / (now - lastFrameTime);
            lastFrameTime = now;

            fpsArray.push(fps);
            if (fpsArray.length > 10) fpsArray.shift();
            const avgFps = fpsArray.reduce((a, b) => a + b, 0) / fpsArray.length;
            fpsVal.innerText = Math.round(avgFps);
        }
    } catch (err) {
        console.error("Frame processing error:", err);
    }

    // Regulate loop rate (matches processing rate, but caps at max 30 fps)
    if (cameraActive) {
        setTimeout(processFrameLoop, 15);
    }
}
