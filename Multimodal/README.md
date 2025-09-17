# Multimodal AI/ML Toolkit

**Author:** Julia Wen (<wendigilane@gmail.com>)  
**License:** MIT

## Overview

The **Multimodal/** directory contains AI/ML tools that operate across multiple domains (Audio, Image, Video) to demonstrate:

- **Graph-based reasoning** using A* on embeddings (`graphs/`)  
- **LLM integration** for natural-language explanations of AI pipelines (`llm/`)  

These components showcase **cross-modal AI/ML concepts** as independent experiments, combining classical algorithms, embeddings, and large language models.

---

## Directory Structure

```
Multimodal/
  graphs/       # A* and knowledge graph demos
    app_images_astar.py
  llm/          # LLM integration scripts
    ai_llm_fft_demo.py
    test_ai_llm_fft_demo.py
  requirements.txt
```

---

## Setup

1. **Activate your Python virtual environment** (from the root repo if needed):
```bash
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows PowerShell
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

---

## Graph-based Demos (`graphs/`)

- **Purpose**: Build embeddings from images/audio/video and perform A* search on the graph.  
- **Run**:
```bash
cd Multimodal/graphs
streamlit run app_images_astar.py
```
- **Output**: Interactive Streamlit web app showing nodes, embeddings, and computed paths.

---

## LLM Integration (`llm/`)

- **Purpose**: Generate natural-language explanations of AI workflows, e.g., FFT analysis.  
- **Scripts**:
  - `ai_llm_fft_demo.py` — Main demo script (requires WAV file input)
  - `test_ai_llm_fft_demo.py` — Test script for validation  
- **Run demo**:
```bash
cd Multimodal/llm
python ai_llm_fft_demo.py path/to/input.wav --output_dir results --show --verbose
```
  - `path/to/input.wav` → path to your WAV file
  - `--output_dir` → optional directory to save results
  - `--show` → optional, display plots interactively
  - `--verbose` → optional, print runtime info

- **Run tests with verbose output (applies to all WAV files in `test/test_files`)**:
```bash
cd Multimodal/llm
pytest -v test_ai_llm_fft_demo.py
```

- **Requirements**: Optional OpenAI API key for LLM explanations:
```bash
export OPENAI_API_KEY="your_api_key_here"   # macOS/Linux
setx OPENAI_API_KEY "your_api_key_here"     # Windows
export LLM_MODEL="gpt-4o-mini"             # optional, default: gpt-3.5-turbo
```

---

## Notes

- Graph and LLM scripts are **independent demos** for experimentation and AI/ML exploration.  
- Each script demonstrates a specific AI/ML capability on its own.  
- No combined workflow across scripts is currently implemented.

---

## License

MIT License  
© 2025 Julia Wen (<wendigilane@gmail.com>)

