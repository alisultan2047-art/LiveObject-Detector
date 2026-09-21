import React, { useRef, useState, useEffect, useCallback } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "./App.css";

const CLASS_COLORS = {
  person: "#E040FB",
  car: "#FFEB3B",
  truck: "#76FF03",
  bus: "#FF4081",
  bicycle: "#00E5FF",
  motorcycle: "#FF9100",
  "traffic light": "#2979FF",
  handbag: "#00E676",
  backpack: "#00E676",
  cell_phone: "#FF1744",
  laptop: "#FF1744",
  default: "#00E676"
};

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [modelLoading, setModelLoading] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [facingMode, setFacingMode] = useState("environment"); // Default to rear camera for mobile

  useEffect(() => {
    const loadModel = async () => {
      try {
        await tf.ready();
        // Force WebGL backend for better mobile GPU acceleration
        await tf.setBackend('webgl'); 
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

      // Sync canvas dimensions to the dynamic video feed
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      const ctx = canvas.getContext("2d");
      const startTime = performance.now();

      // LOWERED THRESHOLD: Changed from 0.40 to 0.30 so the mobile camera catches objects easier
      const predictions = await model.detect(video, 20, 0.30);

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      predictions.forEach((prediction) => {
        const [x, y, width, height] = prediction.bbox;
        const className = prediction.class;
        const score = Math.round(prediction.score * 100);
        const color = CLASS_COLORS[className] || CLASS_COLORS.default;

        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);

        const label = `${className} ${score}%`;
        ctx.font = "bold 16px monospace";
        const textWidth = ctx.measureText(label).width;
        const tagHeight = 22;

        ctx.fillStyle = color;
        ctx.fillRect(x, Math.max(0, y - tagHeight), textWidth + 10, tagHeight);

        ctx.fillStyle = "#000000";
        ctx.fillText(label, x + 5, Math.max(16, y - 4));
      });

      const inferenceLatency = performance.now() - startTime;
      const currentFps = 1000 / (performance.now() - lastFrameTime);
      lastFrameTime = performance.now();

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

    return () => cancelAnimationFrame(animationId);
  }, [isStreaming, modelLoading, model]);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", backgroundColor: "#0f1117", color: "#fff", minHeight: "100vh", padding: "16px", fontFamily: "sans-serif" }}>
      <h3 style={{ margin: "5px 0" }}>LiveObject Detector</h3>
      
      {/* Container updated for dynamic mobile aspect ratios */}
      <div style={{ position: "relative", width: "100%", maxWidth: "100vw", borderRadius: "8px", overflow: "hidden", border: "2px solid #2d3748" }}>
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          // Removed hardcoded width/height to prevent stretching
          videoConstraints={{ facingMode: facingMode }} 
          onUserMedia={handleUserMedia}
          style={{ width: "100%", height: "auto", display: "block" }}
        />
        <canvas
          ref={canvasRef}
          style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}
        />
      </div>

      <div style={{ marginTop: "16px", display: "flex", gap: "10px", alignItems: "center" }}>
        <button
          onClick={() => setFacingMode((prev) => (prev === "user" ? "environment" : "user"))}
          style={{ padding: "12px 20px", borderRadius: "6px", backgroundColor: "#2563eb", color: "#fff", border: "none", cursor: "pointer", fontWeight: "bold" }}
        >
          Switch to {facingMode === "user" ? "Rear Camera" : "Front Camera"}
        </button>
      </div>

      {modelLoading && <p style={{ color: "#fbbf24", marginTop: "12px" }}>Loading Neural Weights...</p>}
      {!modelLoading && isStreaming && <p style={{ color: "#4ade80", marginTop: "12px" }}>● Pipeline Active</p>}
    </div>
  );
}

export default App;
