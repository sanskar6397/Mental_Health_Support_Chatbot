# 🧠 AI-Powered Mental Health Chatbot with Emotion Detection

A privacy-focused mental health support chatbot that adapts responses based on real-time emotion detection from the user's webcam, powered by a locally run **LLaMA 2** model.  
Designed to provide empathetic, CBT-style (Cognitive Behavioral Therapy) conversations without sending sensitive data to the cloud.

## 📌 Features
- **Emotion-Aware Conversations**: Detects user emotions via webcam and adjusts chatbot tone accordingly.
- **On-Device LLaMA 2 Model**: Uses Hugging Face Transformers + BitsAndBytes for efficient local inference.
- **CBT-Inspired Dialogue**: Generates supportive, empathetic responses.
- **Privacy-First**: All processing runs locally; no external API calls.
- **Modular Backend**: Easy to extend with new models or features.

## 🛠️ Technologies Used
- **Programming Language**: Python 3.10
- **Machine Learning**: PyTorch, Hugging Face Transformers, BitsAndBytes, Accelerate
- **Computer Vision**: OpenCV
- **Model**: LLaMA 2 (quantized for local use)
- **Environment**: Virtualenv

## 📂 Project Structure
```
mental_health_chatbot/
│
├── backend/
│   ├── app.py            # Main backend application
│   ├── logic.py          # Chat session logic
│   ├── emotion.py        # Emotion detection via webcam
│   └── model_loader.py   # Loads and configures LLaMA 2 model
│
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── venv310/              # Virtual environment (ignored in .gitignore)
```

## 🚀 Installation and Setup

1. **Clone this repository**
```bash
git clone https://github.com/your-username/mental-health-chatbot.git
cd mental-health-chatbot
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv310
venv310\Scripts\activate  # Windows
source venv310/bin/activate  # Mac/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the backend**
```bash
python -m backend.app
```

## 📸 Emotion Detection
The chatbot uses your webcam to capture a single frame during conversation and applies facial expression recognition to detect emotions such as happy, sad, neutral, angry, surprised, fearful, or disgusted.

## 📈 Future Improvements
- Add a web-based frontend for better user interaction
- Fine-tune LLaMA 2 for mental health dialogues
- Support for multilingual emotional responses
- Real-time emotion tracking during conversations

## ⚠️ Disclaimer
This chatbot is not a substitute for professional mental health support. If you are in emotional distress or thinking about self-harm, please reach out to a licensed mental health professional or a crisis helpline.

## 📄 License
MIT License. You are free to use, modify, and distribute this project.
