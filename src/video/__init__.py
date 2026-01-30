"""
MLB Video Pipeline - Video Package

Video generation using FFmpeg and MoviePy:
- generator: Create videos with stats, charts, and narration
- templates: Video layout and style definitions

Usage:
    from src.video import VideoGenerator

    generator = VideoGenerator()
    video_path = generator.create_video(
        script="...",
        audio_path=audio_path,
        stats_data=stats
    )
"""

from src.video.generator import VideoGenerator
from src.video.templates import VideoTemplate, TEMPLATES

__all__ = ["VideoGenerator", "VideoTemplate", "TEMPLATES"]
