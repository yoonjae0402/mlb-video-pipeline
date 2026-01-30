import logging
import os
from pathlib import Path
from typing import List, Dict, Optional
from moviepy.editor import (
    VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip, 
    TextClip, ColorClip, vfx
)
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from src.video.asset_manager import AssetManager

# Monkey patch for Pillow 10+ compatibility with MoviePy 1.x
if hasattr(Image, 'Resampling'):
    if not hasattr(Image, 'ANTIALIAS'):
        Image.ANTIALIAS = Image.Resampling.LANCZOS

logger = logging.getLogger(__name__)

class VideoAssembler:
    """
    Assembles final video using MoviePy.
    """
    
    def __init__(self, asset_manager: AssetManager, output_dir: str = "outputs/videos"):
        self.asset_manager = asset_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def assemble_video(
        self,
        script_data: Dict, # Contains script segments/timing info if we had it
        audio_path: str,
        charts: List[str],
        output_filename: str = "final_video.mp4"
    ) -> Optional[str]:
        """
        Stitch together audio, background, charts, and overlays.
        """
        try:
            output_path = self.output_dir / output_filename
            logger.info(f"Assembling video to {output_path}...")
            
            # 1. Load Audio
            audio_clip = AudioFileClip(audio_path)
            duration = audio_clip.duration
            
            # 2. visual Background
            # Try to get background, fallback to ColorClip if missing
            bg_path = self.asset_manager.get_background_video("generic")
            if bg_path:
                bg_clip = VideoFileClip(bg_path)
                # Loop background to match audio duration
                bg_clip = bg_clip.loop(duration=duration)
            else:
                # Dark gray generic background
                bg_clip = ColorClip(size=(1080, 1920), color=(20, 20, 20), duration=duration)
                
            # Resize logic if needed (ensure 1080x1920)
            # bg_clip = bg_clip.resize(height=1920)
            # bg_clip = bg_clip.crop(x1=..., width=1080)
            
            # 3. Create Overlays
            clips = [bg_clip]
            
            # Header with Logos
            # Hypothetical: Fetch these IDs from game_data
            # For now, we'll assume they are passed or mocked
            away_logo = self.asset_manager.fetch_team_logo(147) # Yankees placeholder
            home_logo = self.asset_manager.fetch_team_logo(111) # Red Sox placeholder
            
            if away_logo and home_logo:
                # Place logos at top
                # Note: SVGs might need conversion to PNG for moviepy depending on version
                # AssetManager handles download, but we might need a helper to ensure PNG
                # For this MVP, assumng download works or we skip if fail
                pass 
                
            # Charts Overlay (Middle Section)
            # Show charts from 5s to 20s (Example)
            current_time = 5.0
            for chart_path in charts:
                if os.path.exists(chart_path):
                    chart_clip = (
                        ImageClip(chart_path)
                        .set_start(current_time)
                        .set_duration(5.0)
                        .set_position(("center", "center"))
                        .resize(width=900) # Fit within 1080 width
                    )
                    clips.append(chart_clip)
                    current_time += 5.0

            # Text Overlays (Burn-in)
            # Safe alternative to TextClip: Generate Image with Pillow
            title_img = self._create_text_image("MLB DAILY RECAP", font_size=80, color="white")
            title_clip = (
                ImageClip(title_img)
                .set_start(0)
                .set_duration(5)
                .set_position(("center", 200)) # Top area
            )
            clips.append(title_clip)
            
            # Prediction Text (End)
            pred_text = script_data.get("prediction", {}).get("prediction", "See Description")
            pred_img = self._create_text_image(f"PREDICTION:\n{pred_text.upper()}", font_size=70, color="yellow")
            pred_clip = (
                ImageClip(pred_img)
                .set_start(duration - 10) # Last 10 seconds
                .set_duration(10)
                .set_position(("center", "center"))
            )
            clips.append(pred_clip)

            # 4. Composite
            final_video = CompositeVideoClip(clips, size=(1080, 1920))
            final_video = final_video.set_audio(audio_clip)
            final_video = final_video.set_duration(duration)
            
            # 5. Write
            # low fps for speed in dev, use 30/60 for prod
            final_video.write_videofile(
                str(output_path), 
                fps=24, 
                codec="libx264", 
                audio_codec="aac",
                threads=4,
                logger=None # Silence verbose ffmpeg logs
            )
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error assembling video: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_text_image(self, text: str, width: int = 1080, font_size: int = 60, color: str = "white") -> np.ndarray:
        """Create a text overlay image using Pillow."""
        # Estimate height based on newlines
        lines = text.split('\n')
        height = len(lines) * (font_size + 20) + 50
        
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # We don't have font files, use default
        # For actual production, load ttf
        try:
             # Try standard font if available, else default
             font = ImageFont.truetype("Arial", font_size)
        except:
             font = ImageFont.load_default()
        
        # Center text
        y_text = 20
        for line in lines:
            # bbox = draw.textbbox((0, 0), line, font=font)
            # text_width = bbox[2] - bbox[0]
            # x_text = (width - text_width) / 2
            
            # Simple centering (anchor="mm" is middle-middle)
            draw.text((width/2, y_text), line, font=font, fill=color, anchor="mt") # mt = middle top
            y_text += font_size + 10
            
        return np.array(img)
