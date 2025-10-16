// File: Image/typescript/crescentDemo.ts
import * as fs from "fs";
import * as path from "path";

interface DetectionResult {
  imageName: string;
  crescentDetected: boolean;
  confidence: number;
}

// Simulated crescent detection function
function detectCrescent(imageName: string): DetectionResult {
  const crescentDetected = Math.random() > 0.5; // simulated
  const confidence = parseFloat((Math.random() * 100).toFixed(2));
  return { imageName, crescentDetected, confidence };
}

// Helper: read all image files in a folder
function getImagesFromFolder(folderPath: string): string[] {
  if (!fs.existsSync(folderPath)) {
    console.error(`⚠️ Folder not found: ${folderPath}`);
    return [];
  }

  return fs
    .readdirSync(folderPath)
    .filter(file =>
      [".jpg", ".jpeg", ".png"].includes(path.extname(file).toLowerCase())
    );
}

// Dataset folders are one level above, under Image/dataset/
const crescentFolder = path.join(__dirname, "..", "dataset", "crescent");
const noCrescentFolder = path.join(__dirname, "..", "dataset", "no_crescent");

// Read images
const crescentImages = getImagesFromFolder(crescentFolder);
const noCrescentImages = getImagesFromFolder(noCrescentFolder);

if (crescentImages.length === 0 && noCrescentImages.length === 0) {
  console.error("❌ No images found in dataset folders.");
  process.exit(1);
}

// Combine and simulate detection
const allImages = [
  ...crescentImages.map(img => ({ folder: "crescent", img })),
  ...noCrescentImages.map(img => ({ folder: "no_crescent", img })),
];

const results: DetectionResult[] = allImages.map(({ img }) => detectCrescent(img));

// Print results
console.log("🌓 Crescent Detection (Simulated Results)");
console.table(results);

