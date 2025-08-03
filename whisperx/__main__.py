from .whisperx_api import WhisperXTranscriber
import argparse
import os

DEFAULT_AUDIO_DIR = "audio"
SUPPORTED_EXTENSIONS = ('.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac')

def get_audio_files(audio_dir, specific_files=None):
    if specific_files:
        return specific_files
    return [
        os.path.join(audio_dir, f)
        for f in os.listdir(audio_dir)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", nargs="*", help="Specific audio file(s) to process")
    parser.add_argument("--audio_dir", default=DEFAULT_AUDIO_DIR, help="Directory to scan for audio files")
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN"), help="HuggingFace token (optional if set in env)")
    parser.add_argument("--model", default="large-v2", help="Model size (tiny, base, small, medium, large-v1, large-v2)")
    parser.add_argument("--device", default="cpu", help="device to use for PyTorch inference")
    parser.add_argument("--compute_type", default="float32", type=str, choices=["float16", "float32", "int8"], help="compute type for computation")
    parser.add_argument("--language", type=str, default="en", help="language spoken in the audio")
    
    args = parser.parse_args()
    
    audio_files = get_audio_files(args.audio_dir, args.audio)
    if not audio_files:
        raise ValueError(f"No supported audio files found in {args.audio_dir}")
    
    transcriber = WhisperXTranscriber(
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        hf_token=args.hf_token
    )
    
    for audio_path in audio_files:
        transcriber.transcribe(
            audio_path=audio_path,
            language=args.language,
            diarize=True if args.hf_token else False
        )
    
    transcriber.cleanup()

if __name__ == "__main__":
    cli()