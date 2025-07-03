# AI Video Dubber

Transform your videos with AI-powered voice dubbing in multiple Indian languages! This project uses OpenAI Whisper for transcription, Google Translate for translation, and Microsoft Edge TTS for high-quality voice synthesis—all wrapped in a modern Flask web app.

## Features
- 🎥 Upload any video (MP4, AVI, MOV, etc.)
- 🗣️ Automatic speech-to-text transcription (Whisper)
- 🌐 Translate to 10+ Indian languages
- 🗣️ Neural voice dubbing (Edge TTS)
- ⚡ Modern, responsive UI (Tailwind CSS)
- 📥 Download your dubbed video

## Requirements
- Python 3.8+
- FFmpeg (must be in your system PATH)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Googletrans](https://pypi.org/project/googletrans/)
- [edge-tts](https://pypi.org/project/edge-tts/)
- [Flask](https://palletsprojects.com/p/flask/)
- [pydub](https://github.com/jiaaro/pydub)

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Setup
1. **Clone the repo:**
   ```bash
   git clone https://github.com/nikita2999/AI-Video-Dubber.git
   cd AI-Video-Dubber/video_dubber
   ```
2. **Install FFmpeg:**
   - Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to your PATH.
3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the app:**
   ```bash
   python app.py
   ```
5. **Open in your browser:**
   - Go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Usage
- Upload a video file.
- Select your target language.
- Click "Start Dubbing".
- Wait for processing (may take a few minutes for long videos).
- Download your dubbed video!

## Deployment
- For production, use [Gunicorn](https://gunicorn.org/) or [uWSGI](https://uwsgi-docs.readthedocs.io/en/latest/) with Nginx.
- Make sure FFmpeg is installed on your server.
- See the main repo or ask for Render/Heroku deployment tips.

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
MIT License

## Credits
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Google Translate](https://translate.google.com/)
- [Microsoft Edge TTS](https://github.com/rany2/edge-tts)
- [Flask](https://flask.palletsprojects.com/)
- [Tailwind CSS](https://tailwindcss.com/) 