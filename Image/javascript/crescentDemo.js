/**
 * File: Image/javascript/crescentDemo.js
 * Description: JavaScript demo for simulated crescent detection in images.
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Simulated detection function
function detectCrescent(imageName) {
  const crescentDetected = Math.random() > 0.5;
  const confidence = parseFloat((Math.random() * 100).toFixed(2));
  return { imageName, crescentDetected, confidence };
}

// Helper: read images from folder
function getImagesFromFolder(folderPath) {
  if (!fs.existsSync(folderPath)) {
    console.error(`⚠️ Folder not found: ${folderPath}`);
    return [];
  }
  return fs.readdirSync(folderPath)
    .filter(file => [".jpg", ".jpeg", ".png"].includes(path.extname(file).toLowerCase()));
}

// Dataset folders (same as TS version)
const crescentFolder = path.join(__dirname, "..", "dataset", "crescent");
const noCrescentFolder = path.join(__dirname, "..", "dataset", "no_crescent");

const crescentImages = getImagesFromFolder(crescentFolder);
const noCrescentImages = getImagesFromFolder(noCrescentFolder);

if (crescentImages.length === 0 && noCrescentImages.length === 0) {
  console.error("❌ No images found.");
  process.exit(1);
}

const allImages = [...crescentImages, ...noCrescentImages];
const results = allImages.map(img => detectCrescent(img));

console.log("🌓 Crescent Detection (Simulated Results)");
console.table(results);

