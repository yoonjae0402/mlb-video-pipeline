import os
import sys
import wave
import struct
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.video import AssetManager, ChartGenerator, VideoAssembler

def create_dummy_audio(path: str, duration: float = 5.0):
    with wave.open(path, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(44100)
        n_frames = int(duration * 44100)
        data = struct.pack('<h', 0) * n_frames
        f.writeframes(data)

def verify_video_production():
    print("🎬 Verifying MLB Video Pipeline - Video Production 🎬")
    
    output_dir = Path("outputs/verify")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Assets
    print("\n1. Visual Asset Management...")
    asset_manager = AssetManager(cache_dir="data/verify_assets")
    # Mocking downloads to avoid network dependency in this script
    asset_manager.fetch_team_logo = MagicMock(return_value=None) 
    asset_manager.fetch_player_headshot = MagicMock(return_value=None)
    asset_manager.get_background_video = MagicMock(return_value=None)
    print("   AssetManager initialized (network calls mocked).")
    
    # 2. Charts
    print("\n2. Chart Generation...")
    chart_gen = ChartGenerator(output_dir="data/verify_charts")
    trend_path = chart_gen.generate_trend_chart(
        [0.250, 0.260, 0.300, 0.280, 0.320], 
        ["G1", "G2", "G3", "G4", "G5"],
        "Batting Trend"
    )
    print(f"   Generated Chart: {trend_path}")
    
    # 3. Video Assembly
    print("\n3. Video Assembly...")
    assembler = VideoAssembler(asset_manager, output_dir="outputs/verify")
    
    # Create required dummy audio
    audio_path = str(output_dir / "test_audio.wav")
    create_dummy_audio(audio_path, duration=5.0)
    
    # Mock Script Data
    script_data = {
        "prediction": {"prediction": "YANKEES WIN"}
    }
    
    try:
        video_path = assembler.assemble_video(
            script_data=script_data,
            audio_path=audio_path,
            charts=[trend_path] if trend_path else []
        )
        
        if video_path and os.path.exists(video_path):
            print(f"✅ Video generated successfully: {video_path}")
            # Verify file size > 0
            size = os.path.getsize(video_path)
            print(f"   File size: {size / 1024:.2f} KB")
        else:
            print("❌ Video generation failed.")
            
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_video_production()
