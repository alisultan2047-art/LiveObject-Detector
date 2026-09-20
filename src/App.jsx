import React, { useRef, useState, useEffect, useCallback } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "./App.css";

// Color palette mapping to match multi-class dataset visualizations
const CLASS_COLORS = {
  person: "#E040FB",        // Magenta
  car: "#FFEB3B",           // Yellow
  truck: "#76FF03",         // Light Green
  bus: "#FF4081",           // Pink
  bicycle: "#00E5FF",       // Cyan
  motorcycle: "#FF9100",    // Deep Orange
  "traffic light": "#2979FF", // Blue
  handbag: "#00E676",       // Green
  backpack: "#00E676",      // Green
  cell_phone: "#FF1744",    // Bright Red
  default: "#00E676"        // Fallback Green
};

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [modelLoading, setModelLoading] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [facingMode, setFacingMode] = useState("user"); // Toggle between "user" and "environment"

  // 1. Initialize TensorFlow Backend and Load Model
  useEffect(() => {
    const loadModel = async () => {
      try {
        await tf.ready();
        // Load MobileNetV2 architecture for high FPS edge inference
        const loadedModel = await cocoSsd.load({ base: "lite_mobilenet_v2" });
        setModel(loadedModel);
        setModelLoading(false);
      } catch (err) {
        console.error("Failed to load neural network:", err);
      }
    };
    loadModel();
  }, []);

  const handleUserMedia = useCallback(() => {
    setIsStreaming(true);
  }, []);

  // 2. Real-Time Inference Loop
  useEffect(() => {
    if (!isStreaming || modelLoading || !model) return;

    let animationId;
    let lastFrameTime = performance.now();

    const detectFrame = async () => {
      const video = webcamRef.current?.video;
      const canvas = canvasRef.current;

      if (!video || video.readyState !== 4 || video.videoWidth === 0) {
        animationId = requestAnimationFrame(detectFrame);
        return;
      }

      // Synchronize internal canvas resolution with video stream
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      const ctx = canvas.getContext("2d");
      const startTime = performance.now();

      // Run inference directly on the HTMLVideoElement
      const predictions = await model.detect(video, 20, 0.40);

      // Clear the transparent canvas overlay
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Render bounding boxes and semantic tags
      predictions.forEach((prediction) => {
        const [x, y, width, height] = prediction.bbox;
        const className = prediction.class;
        const score = Math.round(prediction.score * 100);
        const color = CLASS_COLORS[className] || CLASS_COLORS.default;

        // Draw Bounding Box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);

        // Draw Label Tag Background
        const label = `${className} ${score}%`;
        ctx.font = "bold 14px monospace";
        const textWidth = ctx.measureText(label).width;
        const tagHeight = 20;

        ctx.fillStyle = color;
        ctx.fillRect(x, Math.max(0, y - tagHeight), textWidth + 10, tagHeight);

        // Draw Text Inside Tag
        ctx.fillStyle = "#000000";
        ctx.fillText(label, x + 5, Math.max(14, y - 5));
      });

      // Calculate performance telemetry
      const inferenceLatency = performance.now() - startTime;
      const currentFps = 1000 / (performance.now() - lastFrameTime);
      lastFrameTime = performance.now();

      // Render Telemetry Banner
      ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
      ctx.fillRect(10, 10, 240, 50);

      ctx.fillStyle = "#00FFFF";
      ctx.font = "bold 13px sans-serif";
      ctx.fillText(`Active Objects: ${predictions.length}`, 20, 30);

      ctx.fillStyle = "#FFB300";
      ctx.fillText(`Inference: ${inferenceLatency.toFixed(1)}ms | ${currentFps.toFixed(1)} FPS`, 20, 48);

      animationId = requestAnimationFrame(detectFrame);
    };

    detectFrame();

    return () => {
      cancelAnimationFrame(animationId);
    };
  }, [isStreaming, modelLoading, model]);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", backgroundColor: "#0f1117", color: "#fff", minHeight: "100vh", padding: "16px", fontFamily: "sans-serif" }}>
      <h2 style={{ margin: "10px 0" }}>COCO Deep Learning Detector</h2>
      
      <div style={{ position: "relative", width: "100%", maxWidth: "640px", borderRadius: "8px", overflow: "hidden", border: "2px solid #2d3748" }}>
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          videoConstraints={{
            width: 640,
            height: 480,
            facingMode: facingMode
          }}
          onUserMedia={handleUserMedia}
          style={{ width: "100%", display: "block" }}
        />
        <canvas
          ref={canvasRef}
          style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}
        />
      </div>

      <div style={{ marginTop: "16px", display: "flex", gap: "10px", alignItems: "center" }}>
        <button
          onClick={() => setFacingMode((prev) => (prev === "user" ? "environment" : "user"))}
          style={{ padding: "8px 16px", borderRadius: "6px", backgroundColor: "#2563eb", color: "#fff", border: "none", cursor: "pointer", fontWeight: "bold" }}
        >
          Switch to {facingMode === "user" ? "Rear Camera" : "Front Camera"}
        </button>
      </div>

      {modelLoading && <p style={{ color: "#fbbf24", marginTop: "12px" }}>Loading COCO-SSD Neural Weights...</p>}
      {!modelLoading && isStreaming && <p style={{ color: "#4ade80", marginTop: "12px" }}>● Neural Pipeline Running</p>}
    </div>
  );
}

export default App;