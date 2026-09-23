import React, { useRef, useState, useEffect, useCallback } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "./App.css";

// Color mapping for common classes out of the 80 total Microsoft COCO dataset categories
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
  cat: "#F59E0B",
  dog: "#F59E0B",
  tv: "#3B82F6",
  default: "#00E676"
};

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
  const verificationMemory = useRef({});
  
  // NEW: State and memory to drive the live right-side panel safely
  const [detectedItems, setDetectedItems] = useState([]);
  const lastVerifiedStr = useRef("");

  useEffect(() => {
    const loadModel = async () => {
      try {
        await tf.ready();
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

      for (const key in verificationMemory.current) {
        if (!currentFrameClasses.includes(key)) {
          verificationMemory.current[key] = 0; 
        }
      }

      let activeVerifiedCount = 0;
      const verifiedThisFrame = new Set(); // Track objects for the sidebar

      predictions.forEach((prediction) => {
        const className = prediction.class;
        verificationMemory.current[className] = (verificationMemory.current[className] || 0) + 1;

        if (verificationMemory.current[className] >= REQUIRED_FRAMES) {
          activeVerifiedCount++;
          verifiedThisFrame.add(className); // Add to active targets list
          
          const [x, y, width, height] = prediction.bbox;
          const score = Math.round(prediction.score * 100);
          const color = CLASS_COLORS[className] || CLASS_COLORS.default;

          ctx.strokeStyle = color;
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, width, height);

          const label = `${className} ${score}%`;
          ctx.font = "bold 16px monospace";
          const textWidth = ctx.measureText(label).width;
          ctx.fillStyle = color;
          ctx.fillRect(x, Math.max(0, y - 22), textWidth + 10, 22);
          ctx.fillStyle = "#000000";
          ctx.fillText(label, x + 5, Math.max(16, y - 4));

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

      // PANEL UPDATE LOGIC: Only update React state if the list of targets actually changed
      const verifiedArray = Array.from(verifiedThisFrame).sort();
      const currentStr = verifiedArray.join(",");
      if (lastVerifiedStr.current !== currentStr) {
        setDetectedItems(verifiedArray);
        lastVerifiedStr.current = currentStr;
      }

      const inferenceLatency = performance.now() - startTime;
      const currentFps = 1000 / (performance.now() - lastFrameTime);
      lastFrameTime = performance.now();

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
    <div style={{ backgroundColor: "#0f1117", color: "#fff", minHeight: "100vh", padding: "16px", fontFamily: "sans-serif" }}>
      <h3 style={{ margin: "5px 0", textAlign: "center" }}>LiveObject Detector</h3>
      
      {/* NEW: Responsive Flex Layout to separate Video and Sidebar */}
      <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: "20px", marginTop: "16px", maxWidth: "1000px", margin: "16px auto" }}>
        
        {/* LEFT COLUMN: Camera Feed */}
        <div style={{ flex: "1 1 600px", display: "flex", flexDirection: "column", alignItems: "center" }}>
          <div style={{ position: "relative", width: "100%", borderRadius: "8px", overflow: "hidden", border: "2px solid #2d3748" }}>
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

        {/* RIGHT COLUMN: Live Sidebar */}
        <div style={{ flex: "1 1 250px", backgroundColor: "#1e293b", border: "2px solid #334155", borderRadius: "8px", padding: "16px", minHeight: "300px" }}>
          <h4 style={{ borderBottom: "1px solid #475569", paddingBottom: "10px", marginTop: "0" }}>Active Targets</h4>
          
          {detectedItems.length === 0 ? (
            <p style={{ color: "#94a3b8", fontSize: "14px", fontStyle: "italic" }}>Scanning area...</p>
          ) : (
            <ul style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: "12px" }}>
              {detectedItems.map((item, index) => {
                const color = CLASS_COLORS[item] || CLASS_COLORS.default;
                return (
                  <li key={index} style={{ display: "flex", alignItems: "center", gap: "12px", backgroundColor: "#0f1117", padding: "10px", borderRadius: "6px", borderLeft: `5px solid ${color}` }}>
                    <div style={{ width: "12px", height: "12px", borderRadius: "50%", backgroundColor: color }}></div>
                    <span style={{ textTransform: "capitalize", fontWeight: "bold", fontSize: "15px" }}>{item.replace("_", " ")}</span>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
