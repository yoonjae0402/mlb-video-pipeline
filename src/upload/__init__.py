"""
MLB Video Pipeline - Upload Package

YouTube API integration for video uploads:
- youtube: Upload videos and manage channel

Usage:
    from src.upload import YouTubeUploader

    uploader = YouTubeUploader()
    video_id = uploader.upload(
        video_path=video_path,
        title="Game Recap: NYY vs BOS",
        description="...",
        tags=["MLB", "baseball"]
    )
"""

from src.upload.youtube import YouTubeUploader

__all__ = ["YouTubeUploader"]
