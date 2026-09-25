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

// The AI must see the object 3 frames in a row to verify it (filters out glitches without causing lag)
const REQUIRED_FRAMES = 3; 

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [modelLoading, setModelLoading] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [facingMode, setFacingMode] = useState("environment"); 
  
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const speechMemory = useRef({}); 
  
  // Temporal Smoothing Memory Bank (No trajectory math)
  const verificationMemory = useRef({});

  useEffect(() => {
    const loadModel = async () => {
      try {
        await tf.ready();
        await tf.setBackend('webgl'); 
        
        // Reverted to the high-speed "lite" model to fix the lag
        const loadedModel = await cocoSsd.load({ base: "lite_mobilenet_v2" });
        setModel(loadedModel);
        setModelLoading(false);
      } catch (err) {
        console.error("Failed to load neural network:", err);
      }
    };
    loadModel();
  }, []);

  const handleUserMedia = useCallback(() => setIsStreaming(true), []);

  const toggleVoice = () => {
    if (!voiceEnabled) {
      const unlockUtterance = new SpeechSynthesisUtterance("");
      window.speechSynthesis.speak(unlockUtterance);
    } else {
      window.speechSynthesis.cancel();
      speechMemory.current = {};
    }
    setVoiceEnabled(!voiceEnabled);
  };

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

      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      const ctx = canvas.getContext("2d");
      const startTime = performance.now();

      const predictions = await model.detect(video, 20, 0.40);
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const now = Date.now();
      const currentFrameClasses = predictions.map(p => p.class);

      // Clean up temporal memory for objects that disappeared
      for (const key in verificationMemory.current) {
        if (!currentFrameClasses.includes(key)) {
          verificationMemory.current[key] = 0; 
        }
      }

      let activeVerifiedCount = 0;

      predictions.forEach((prediction) => {
        const className = prediction.class;
        
        // TEMPORAL SMOOTHING: Increment frame counter for this class
        verificationMemory.current[className] = (verificationMemory.current[className] || 0) + 1;

        // Only draw boxes and trigger audio if verified across 3 consecutive frames
        if (verificationMemory.current[className] >= REQUIRED_FRAMES) {
          activeVerifiedCount++;
          
          const [x, y, width, height] = prediction.bbox;
          const score = Math.round(prediction.score * 100);
          const color = CLASS_COLORS[className] || CLASS_COLORS.default;

          // Draw Box
          ctx.strokeStyle = color;
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, width, height);

          // Draw Label
          const label = `${className} ${score}%`;
          ctx.font = "bold 16px monospace";
          const textWidth = ctx.measureText(label).width;
          ctx.fillStyle = color;
          ctx.fillRect(x, Math.max(0, y - 22), textWidth + 10, 22);
          ctx.fillStyle = "#000000";
          ctx.fillText(label, x + 5, Math.max(16, y - 4));

          // Voice Output
          if (voiceEnabled && window.speechSynthesis) {
            const lastSpokenTime = speechMemory.current[className] || 0;
            if (now - lastSpokenTime > 5000) {
              const utterance = new SpeechSynthesisUtterance(`${className.replace("_", " ")} confirmed`);
              utterance.rate = 1.1;
              window.speechSynthesis.speak(utterance);
              speechMemory.current[className] = now;
            }
          }
        }
      });

      const inferenceLatency = performance.now() - startTime;
      const currentFps = 1000 / (performance.now() - lastFrameTime);
      lastFrameTime = performance.now();

      // UI Telemetry
      ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
      ctx.fillRect(10, 10, 240, 50);
      ctx.fillStyle = "#00FFFF";
      ctx.font = "bold 13px sans-serif";
      ctx.fillText(`Verified Targets: ${activeVerifiedCount}`, 20, 30);
      ctx.fillStyle = "#FFB300";
      ctx.fillText(`Latency: ${inferenceLatency.toFixed(1)}ms | ${currentFps.toFixed(1)} FPS`, 20, 48);

      animationId = requestAnimationFrame(detectFrame);
    };

    detectFrame();
    return () => cancelAnimationFrame(animationId);
  }, [isStreaming, modelLoading, model, voiceEnabled]);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", backgroundColor: "#0f1117", color: "#fff", minHeight: "100vh", padding: "16px", fontFamily: "sans-serif" }}>
      <h3 style={{ margin: "5px 0" }}>LiveObject Detector</h3>
      
      <div style={{ position: "relative", width: "100%", maxWidth: "100vw", borderRadius: "8px", overflow: "hidden", border: "2px solid #2d3748" }}>
        <Webcam
          audio={false}
          muted={true}
          playsInline={true}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          videoConstraints={{ facingMode: facingMode }} 
          onUserMedia={handleUserMedia}
          style={{ width: "100%", height: "auto", display: "block" }}
        />
        <canvas
          ref={canvasRef}
          style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" }}
        />
      </div>

      <div style={{ marginTop: "16px", display: "flex", gap: "10px", alignItems: "center", flexWrap: "wrap", justifyContent: "center" }}>
        <button
          onClick={() => setFacingMode((prev) => (prev === "user" ? "environment" : "user"))}
          style={{ padding: "12px 20px", borderRadius: "6px", backgroundColor: "#2563eb", color: "#fff", border: "none", cursor: "pointer", fontWeight: "bold" }}
        >
          Switch to {facingMode === "user" ? "Rear Camera" : "Front Camera"}
        </button>

        <button
          onClick={toggleVoice}
          style={{ padding: "12px 20px", borderRadius: "6px", backgroundColor: voiceEnabled ? "#ef4444" : "#10b981", color: "#fff", border: "none", cursor: "pointer", fontWeight: "bold" }}
        >
          {voiceEnabled ? "🔇 Mute AI Voice" : "🔊 Enable AI Voice"}
        </button>
      </div>

      {modelLoading && <p style={{ color: "#fbbf24", marginTop: "12px" }}>Loading Fast Neural Weights...</p>}
      {!modelLoading && isStreaming && <p style={{ color: "#4ade80", marginTop: "12px" }}>● Pipeline Active</p>}
    </div>
  );
}

export default App;
