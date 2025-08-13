# Human Pose Estimation with MoveNet (MultiPose Lightning) — Images · Video · GIF · COCO Export

A production-ready, script-based Python project to run **Google MoveNet MultiPose Lightning** (via **TensorFlow Hub**) for **multi-person 2D human pose estimation** on **images**, **videos**, and **animated GIFs**. Includes an ergonomic CLI, reliable GIF I/O, and a COCO-style exporter (with optional annotated overlay output).

> **Highlights:** multi-person keypoints (17 COCO joints), robust GIF handling, clean src/ layout, TF-Hub model loading, COCO JSON export, Windows-friendly, and WSL2 GPU notes.

---

## ✨ Features

* **CLI** for quick inference on `image | video | gif`
* **Clean overlays** (keypoints + skeleton) with OpenCV
* **COCO-style JSON export** (optionally also writes the annotated media in the same run)
* **Reliable GIF support** (read/write) using `imageio`
* **Reproducible project layout** (separate modules, ready for VS Code)
* **Windows / WSL2 friendly** (GPU on WSL2 Linux; CPU on native Windows)

---

## 🔎 Keywords (for discoverability)

`MoveNet`, `MultiPose Lightning`, `TensorFlow Hub`, `human pose estimation`, `multi-person pose`, `COCO keypoints`, `2D pose`, `OpenCV`, `imageio`, `GIF`, `WSL2`, `Windows`, `Python`, `CLI`, `computer vision`

---

## 🧱 Project Structure

```
pose_project/
├─ data/                             # put your inputs/outputs here
│  └─ .gitkeep
├─ src/
│  └─ movenet_runner/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ model.py
│     ├─ draw.py
│     ├─ io_utils.py
│     ├─ infer_core.py
│     ├─ cli.py
│     └─ export_coco.py
├─ tests/
│  ├─ test_io_utils.py
│  └─ test_infer_core.py
├─ requirements.txt
├─ pyproject.toml
├─ README.md
└─ .gitignore
```

---

## 🚀 Quickstart

### 1) Create environment & install

**Windows (Command Prompt / PowerShell):**

```bat
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

**WSL2 Ubuntu (recommended for GPU with TensorFlow):**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

> Note: On **native Windows**, TensorFlow runs **CPU-only**. For **GPU**, use **WSL2 (Ubuntu)**.

### 2) Run inference

**GIF → annotated GIF**

```bat
pose-run --input data\input.gif --output data\output.gif --kind gif --size 256 --threshold 0.11 --fps 15
```

**Image → annotated PNG**

```bat
pose-run --input data\person.jpg --output data\person_out.png --kind image
```

**Video → annotated MP4**

```bat
pose-run --input data\clip.mp4 --output data\clip_out.mp4 --kind video --fps 30
```

---

## 📦 COCO Export (with optional overlay output)

**JSON only (no overlay media):**

```bat
python -m movenet_runner.export_coco ^
  --input data\input.gif ^
  --kind gif ^
  --output data\export.json
```

**JSON + annotated overlay in the same run:**

```bat
python -m movenet_runner.export_coco ^
  --input data\input.gif ^
  --kind gif ^
  --output data\export.json ^
  --overlay_out data\overlay.gif ^
  --fps 15
```

> Works similarly for `--kind image` (write PNG) and `--kind video` (write MP4).

---

## ⚙️ CLI Reference

### `pose-run` (inference & overlay)

```
--input <path>          # image/video/gif
--output <path>         # output media path
--kind [image|video|gif]
--size 256              # square model input (192/256/320)
--threshold 0.11        # keypoint visibility threshold
--fps 15                # for GIF/video writing
```

### `python -m movenet_runner.export_coco` (export COCO JSON)

```
--input <path>          # image/video/gif
--kind [image|video|gif]
--output <path.json>    # COCO JSON
--overlay_out <path>    # (optional) also write annotated media (PNG/MP4/GIF)
--size 256
--threshold 0.11
--fps 15
```

---

## 📝 Notes & Tips

* **First run downloads model** to TF-Hub cache (default under your user cache dir). You can customize with:

  ```
  TFHUB_CACHE_DIR=<custom_folder>
  ```
* **Performance:** `--size 192` is faster (slight accuracy drop). Larger sizes cost more time.
* **GIF handling:** Done via `imageio` to avoid OpenCV GIF quirks. Output FPS is controlled by `--fps`.
* **Multi-person:** MoveNet MultiPose Lightning returns up to **6 persons**, each with **17 keypoints**.
* **Coordinate mapping:** Frames are letterboxed to a square for the model, then keypoints are mapped back to original resolution for overlay and COCO export.

---

## 🧪 Testing (optional but recommended)

```bat
pytest -q
```

---

## 🐛 Troubleshooting

* **`ModuleNotFoundError: movenet_runner`**
  Ensure you ran `pip install -e .` in the activated venv, or run with `PYTHONPATH=./src`.

* **Red squiggles in VS Code (imports unresolved)**
  Select the interpreter: *Python: Select Interpreter* → choose your project `.venv`.
  Add (optional) `.env` with `PYTHONPATH=${workspaceFolder}/src`.

* **Slow on Windows**
  Expected on CPU-only TF. Use **WSL2** for GPU acceleration.

---

## 📄 License

MIT © 2025 Siddharth Varshney

---

## 🙏 Acknowledgments

* Google **MoveNet MultiPose Lightning** (TensorFlow Hub)
* OpenCV, ImageIO, NumPy

---