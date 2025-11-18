/**
 * App.jsx — Crescent Moon Detection Frontend
 * Author: Julia Wen (wendigilane@gmail.com)
 * Date: 2025-11-18
 *
 * Purpose:
 * --------
 * Provides a simple React + JavaScript interface for demonstrating
 * crescent moon detection results from the FastAPI backend.
 *
 * When the user clicks “Run Detection,” this app:
 *   1. Loads each image from /public/dataset/...
 *   2. Sends it to http://127.0.0.1:8000/detect
 *   3. Displays thumbnail results with prediction and confidence (%)
 *
 * Requirements:
 * -------------
 *   - FastAPI backend running at http://127.0.0.1:8000
 *   - Public dataset images under:
 *       web/public/dataset/crescent/
 *       web/public/dataset/no_crescent/
 */

import React, { useState } from "react";

const dataset = [
  { folder: "crescent", file: "crescent10.jpg" },
  { folder: "crescent", file: "crescent11.jpg" },
  { folder: "crescent", file: "crescent12.jpg" },
  { folder: "no_crescent", file: "no_crescent12.jpg" },
  { folder: "no_crescent", file: "no_crescent13.jpg" }
];

export default function App() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const runDetection = async () => {
    setLoading(true);
    const resArr = [];

    for (const img of dataset) {
      const response = await fetch(`/dataset/${img.folder}/${img.file}`);
      const blob = await response.blob();
      const formData = new FormData();
      formData.append("file", blob, img.file);

      const res = await fetch("http://127.0.0.1:8000/detect", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      resArr.push({
        image: `/dataset/${img.folder}/${img.file}`,
        crescentDetected: data.detected,
        confidence: data.confidence
      });
    }

    setResults(resArr);
    setLoading(false);
  };

  return (
    <div style={{ padding: 20, textAlign: "center", fontFamily: "Arial, sans-serif" }}>
      <h1>🌓 Crescent Detection (JavaScript Frontend)</h1>
      <button
        onClick={runDetection}
        disabled={loading}
        style={{
          padding: 10,
          fontSize: "1rem",
          marginBottom: 20,
          borderRadius: 8,
          backgroundColor: "#1D4ED8",
          color: "white",
          border: "none",
          cursor: "pointer"
        }}
      >
        {loading ? "Running..." : "Run Detection"}
      </button>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
          gap: 16
        }}
      >
        {results.map((res, i) => (
          <div
            key={i}
            style={{
              border: "1px solid #ddd",
              borderRadius: 12,
              padding: 10,
              backgroundColor: "#f9f9f9"
            }}
          >
            <img
              src={res.image}
              alt={res.image}
              style={{
                width: "100%",
                height: 120,
                objectFit: "cover",
                borderRadius: 8,
                marginBottom: 8
              }}
            />
            <p style={{ fontWeight: "bold" }}>
              {res.crescentDetected ? "🌙 Crescent Detected" : "☀️ No Crescent"}
            </p>
            <p style={{ color: "#555" }}>
              Confidence: {(res.confidence * 100).toFixed(2)}%
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
