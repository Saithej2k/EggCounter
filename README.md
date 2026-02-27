# 🐟 Clownfish Egg Counter

A web app for counting clownfish eggs using classical computer vision.

## How it works

1. **Color filter** — HSV thresholding isolates orange egg pixels
2. **Bandpass filter** — Highlights egg-sized features in the LAB A-channel
3. **Peak detection** — Non-maximum suppression finds one peak per egg center
4. **Validation** — Saturation + color checks reject false positives

> ⚠️ This uses classical CV (not deep learning). Results are estimates and may have errors in very dense clusters or poor lighting.

## Setup

```bash
# Clone / download this folder
cd egg-counter-web

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run
python app.py
```

Open **http://localhost:5000** in your browser.

## Usage

1. Upload a photo of clownfish eggs (JPG/PNG, up to 16MB)
2. Click "Count Eggs"
3. View the count + detection overlay, bandpass filter, and orange mask

## Tech Stack

- **Backend**: Flask + OpenCV + scipy
- **Frontend**: Vanilla HTML/CSS/JS
- **Algorithm**: Bandpass (Difference of Gaussians) + local maxima with NMS + saturation filtering
