# 🖐️ Hand Sign Recognition System Pro

A high-performance, modular, and professional hand sign recognition system built with Python, TensorFlow, and OpenCV.

## ✨ Project Highlights

*   **Real-time Recognition**: Sub-millisecond inference using optimized TensorFlow/Keras models.
*   **Modular Architecture**: Clean separation of concerns with dedicated modules for detection, classification, and data collection.
*   **Modern GUI**: A sleek, dark-themed Tkinter interface for intuitive interaction.
*   **Robust Detection**: Leverages `cvzone` and MediaPipe for industry-standard hand tracking.
*   **Seamless Data Collection**: Built-in tools to expand your dataset with just a few clicks.

## 📂 Project Structure

```text
DTL.youtube/
├── main.py                 # 🚀 Main Entry Point
├── requirements.txt        # 📦 Dependencies
├── .gitignore              # 🙈 Ignored files
├── src/                    # 🛠️ Source Code
│   ├── app.py              # 🎨 GUI Application
│   ├── config.py           # ⚙️ Configuration
│   ├── hand_detector.py    # 🎯 Hand Detection Module
│   ├── hand_sign_classifier.py # 🧠 Classification Module
│   ├── data_collector.py   # 📸 Data Collection Module
│   ├── collect_data.py     # ⌨️ CLI Data Collection Tool
│   └── test_model.py       # 🧪 CLI Testing Tool
├── model/                  # 🤖 AI Models
│   ├── keras_model.h5      # Pre-trained Weights
│   └── labels.txt          # Label Mappings
└── data_img/               # 📊 Dataset Storage
```

## 🚀 Quick Start

### 1. Installation

Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
# source venv/bin/activate
```

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Launching the App

The easiest way to use the system is via the main GUI:

```bash
python main.py
```

### 3. CLI Utilities

For advanced users or specific tasks:

*   **Test Model**: `python src/test_model.py`
*   **Collect Data**: `python src/collect_data.py`

## 🛠️ Configuration

Customize the system behavior in `src/config.py`:

*   `CONFIDENCE_THRESHOLD`: Adjust sensitivity (default: 0.8).
*   `CAMERA_ID`: Change if using external webcams.
*   `THEME_*`: Customize the UI color palette.

## 📈 Training Your Own Model

1.  **Collect Data**: Use the "Collection" mode in the GUI to gather 200+ images per sign.
2.  **Export Images**: Your images are saved in `data_img/[sign_name]`.
3.  **Train**: Use [Google Teachable Machine](https://teachablemachine.withgoogle.com/) for a quick setup:
    *   Upload folders from `data_img`.
    *   Train as an Image Project.
    *   Export as **Keras** model.
4.  **Deploy**: Replace the files in the `model/` directory with your new `keras_model.h5` and `labels.txt`.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📜 License

This project is open-source and available under the MIT License.

---
Made with ❤️ for the AI Community.
