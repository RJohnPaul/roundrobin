# 🎤 Audio Transcription & MoM Generator - Quick Start Guide

## 🚀 How to Run the Application

### 1. Start the Server
```powershell
cd "c:\Users\iamjo\OneDrive\Desktop\mom"
python main.py
```

### 2. Open in Browser
Navigate to: `http://127.0.0.1:8080`

### 3. Use the Application
1. **Upload Audio File**: Drag & drop or click to select (MP3, WAV, M4A, OGG, FLAC, AAC)
2. **Choose Model**: Select transcription accuracy (tiny=fastest, medium=balanced, large=most accurate)
3. **Start Transcription**: Click "Start Transcription" button
4. **Generate MoM**: After transcription, click "Generate Meeting Minutes" for LaTeX output

## 🔧 Technical Features

### Audio Transcription
- **Engine**: faster-whisper (offline processing)
- **File Size Limit**: 100MB
- **Supported Formats**: MP3, WAV, M4A, OGG, FLAC, AAC
- **Models Available**: 
  - `tiny` - Fastest, lower accuracy
  - `base` - Good balance
  - `small` - Better accuracy
  - `medium` - High accuracy (recommended)

### LaTeX Generation  
- **API**: Groq LLaMA integration with hardcoded key
- **Fallback**: Offline template-based generation
- **Output**: Clean, professional meeting minutes format
- **Features**: Automatic section detection, bullet points, action items

### Frontend Features
- **Framework**: Alpine.js for reactivity
- **Styling**: Tailwind CSS for modern UI
- **Real-time**: Progress tracking and notifications
- **Responsive**: Mobile-friendly design

## 📁 Key Files

| File | Purpose |
|------|---------|
| `main.py` | Flask backend with all routes |
| `templates/index.html` | Frontend UI with Alpine.js |
| `.env` | Environment variables (Groq API key) |
| `requirements.txt` | Python dependencies |
| `audio/` | Sample audio files |
| `whisperx/` | WhisperX integration modules |

## 🧹 Cleanup Commands

### Remove Unused Files
```powershell
python cleanup_unused_files.py
```

### Check Dependencies
```powershell
pip list | findstr -i "whisper groq flask"
```

## 🐛 Troubleshooting

### Common Issues
1. **Port Already in Use**: Change port in `main.py` line `app.run(host='0.0.0.0', port=8080, debug=True)`
2. **Module Not Found**: Run `pip install -r requirements.txt`
3. **Audio Upload Fails**: Check file size (<100MB) and format
4. **LaTeX Generation Fails**: Check `.env` file exists with Groq API key

### Development Mode
- Debug mode is enabled in `main.py`
- Console logs available in browser DevTools (F12)
- Backend logs appear in terminal

## 🔑 API Configuration

The application now properly uses environment variables for API keys. Make sure you have a `.env` file:
```env
GROQ_API_KEY=your_api_key_here
```

The application loads this automatically using python-dotenv.

## 📊 Performance Notes

- **Model Selection Impact**:
  - `tiny`: ~1-2 seconds for 5-minute audio
  - `medium`: ~10-15 seconds for 5-minute audio
  - Processing time scales with audio length

- **Memory Usage**:
  - Models are cached after first load
  - RAM usage depends on model size
  - Audio files are processed in chunks

## 🆘 Support

If you encounter issues:
1. Check terminal logs for backend errors
2. Check browser console (F12) for frontend errors  
3. Ensure all dependencies are installed
4. Verify audio file format and size
5. Test with sample audio file in `audio/` directory
