"""
MLB Video Pipeline - YouTube Uploader

Handles video uploads to YouTube using the YouTube Data API v3.
Supports setting titles, descriptions, tags, thumbnails, and scheduling.

Usage:
    from src.upload.youtube import YouTubeUploader

    uploader = YouTubeUploader()

    # Authenticate (first time)
    uploader.authenticate()

    # Upload video
    video_id = uploader.upload(
        video_path="output.mp4",
        title="Game Recap: NYY vs BOS",
        description="Full game analysis...",
        tags=["MLB", "baseball", "Yankees", "Red Sox"]
    )

    # Set custom thumbnail
    uploader.set_thumbnail(video_id, "thumbnail.png")
"""

from pathlib import Path
from typing import Any
from datetime import datetime
import pickle

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from config.settings import settings
from src.utils.logger import get_logger


logger = get_logger(__name__)


class YouTubeUploader:
    """
    Upload videos to YouTube.

    Handles OAuth authentication, video uploads, metadata setting,
    and thumbnail uploads.
    """

    # OAuth scopes required for uploading
    SCOPES = [
        "https://www.googleapis.com/auth/youtube.upload",
        "https://www.googleapis.com/auth/youtube",
    ]

    # Video categories
    CATEGORIES = {
        "sports": "17",
        "entertainment": "24",
        "education": "27",
        "science": "28",
    }

    def __init__(
        self,
        credentials_path: Path | None = None,
        token_path: Path | None = None,
    ):
        """
        Initialize the YouTube uploader.

        Args:
            credentials_path: Path to OAuth client secrets JSON
            token_path: Path to store/load auth token
        """
        self.credentials_path = credentials_path or settings.base_dir / "credentials.json"
        self.token_path = token_path or settings.base_dir / "token.pickle"

        self.credentials: Credentials | None = None
        self.youtube = None

        logger.info("YouTubeUploader initialized")

    # =========================================================================
    # Authentication
    # =========================================================================

    def authenticate(self, force_refresh: bool = False) -> bool:
        """
        Authenticate with YouTube API.

        On first run, this will open a browser for OAuth consent.
        Subsequent runs will use the stored token.

        Args:
            force_refresh: Force re-authentication

        Returns:
            True if authentication successful
        """
        if not settings.enable_youtube_upload:
            logger.warning("YouTube upload is disabled in settings")
            return False

        # Try to load existing credentials
        if self.token_path.exists() and not force_refresh:
            with open(self.token_path, "rb") as token_file:
                self.credentials = pickle.load(token_file)

        # Check if credentials are valid
        if self.credentials and self.credentials.valid:
            logger.info("Using cached YouTube credentials")
        elif self.credentials and self.credentials.expired and self.credentials.refresh_token:
            logger.info("Refreshing YouTube credentials")
            self.credentials.refresh(Request())
        else:
            # Need to authenticate
            if not self.credentials_path.exists():
                logger.error(f"OAuth credentials file not found: {self.credentials_path}")
                logger.info("Download from Google Cloud Console > APIs & Services > Credentials")
                return False

            logger.info("Starting OAuth flow (will open browser)")
            flow = InstalledAppFlow.from_client_secrets_file(
                str(self.credentials_path),
                self.SCOPES
            )
            self.credentials = flow.run_local_server(port=8080)

            # Save credentials for future use
            with open(self.token_path, "wb") as token_file:
                pickle.dump(self.credentials, token_file)
            logger.info("Credentials saved for future use")

        # Build YouTube API client
        self.youtube = build("youtube", "v3", credentials=self.credentials)
        logger.info("YouTube API client initialized")

        return True

    # =========================================================================
    # Video Upload
    # =========================================================================

    def upload(
        self,
        video_path: Path | str,
        title: str,
        description: str = "",
        tags: list[str] | None = None,
        category: str = "sports",
        privacy: str = "private",
        scheduled_time: datetime | None = None,
    ) -> str | None:
        """
        Upload a video to YouTube.

        Args:
            video_path: Path to video file
            title: Video title (max 100 characters)
            description: Video description (max 5000 characters)
            tags: List of tags
            category: Category name (sports, entertainment, etc.)
            privacy: Privacy status (private, unlisted, public)
            scheduled_time: When to publish (for scheduled uploads)

        Returns:
            YouTube video ID or None if failed
        """
        video_path = Path(video_path)

        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return None

        if not self.youtube:
            if not self.authenticate():
                return None

        if settings.dry_run:
            logger.info(f"DRY RUN: Would upload {video_path}")
            return "DRY_RUN_VIDEO_ID"

        # Prepare metadata
        tags = tags or []
        category_id = self.CATEGORIES.get(category, "17")  # Default to sports

        # Truncate title/description if needed
        title = title[:100]
        description = description[:5000]

        body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags,
                "categoryId": category_id,
            },
            "status": {
                "privacyStatus": privacy,
                "selfDeclaredMadeForKids": False,
            },
        }

        # Add scheduled publish time if provided
        if scheduled_time and privacy == "private":
            body["status"]["publishAt"] = scheduled_time.isoformat()
            body["status"]["privacyStatus"] = "private"

        # Create upload request
        media = MediaFileUpload(
            str(video_path),
            mimetype="video/mp4",
            resumable=True,
            chunksize=1024 * 1024,  # 1MB chunks
        )

        logger.info(f"Uploading {video_path} ({video_path.stat().st_size / 1024 / 1024:.1f} MB)")

        try:
            request = self.youtube.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media,
            )

            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    logger.info(f"Upload progress: {progress}%")

            video_id = response["id"]
            logger.info(f"Upload complete! Video ID: {video_id}")
            logger.info(f"URL: https://youtube.com/watch?v={video_id}")

            return video_id

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise

    def set_thumbnail(
        self,
        video_id: str,
        thumbnail_path: Path | str,
    ) -> bool:
        """
        Set custom thumbnail for a video.

        Args:
            video_id: YouTube video ID
            thumbnail_path: Path to thumbnail image

        Returns:
            True if successful
        """
        thumbnail_path = Path(thumbnail_path)

        if not thumbnail_path.exists():
            logger.error(f"Thumbnail file not found: {thumbnail_path}")
            return False

        if not self.youtube:
            if not self.authenticate():
                return False

        if settings.dry_run:
            logger.info(f"DRY RUN: Would set thumbnail for {video_id}")
            return True

        try:
            media = MediaFileUpload(str(thumbnail_path), mimetype="image/png")

            self.youtube.thumbnails().set(
                videoId=video_id,
                media_body=media,
            ).execute()

            logger.info(f"Thumbnail set for video {video_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to set thumbnail: {e}")
            return False

    # =========================================================================
    # Video Management
    # =========================================================================

    def update_video(
        self,
        video_id: str,
        title: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        privacy: str | None = None,
    ) -> bool:
        """
        Update video metadata.

        Args:
            video_id: YouTube video ID
            title: New title (optional)
            description: New description (optional)
            tags: New tags (optional)
            privacy: New privacy status (optional)

        Returns:
            True if successful
        """
        if not self.youtube:
            if not self.authenticate():
                return False

        try:
            # Get current video data
            response = self.youtube.videos().list(
                part="snippet,status",
                id=video_id,
            ).execute()

            if not response.get("items"):
                logger.error(f"Video not found: {video_id}")
                return False

            video = response["items"][0]

            # Update fields
            if title:
                video["snippet"]["title"] = title[:100]
            if description:
                video["snippet"]["description"] = description[:5000]
            if tags:
                video["snippet"]["tags"] = tags
            if privacy:
                video["status"]["privacyStatus"] = privacy

            # Execute update
            self.youtube.videos().update(
                part="snippet,status",
                body=video,
            ).execute()

            logger.info(f"Video {video_id} updated")
            return True

        except Exception as e:
            logger.error(f"Failed to update video: {e}")
            return False

    def get_video_status(self, video_id: str) -> dict[str, Any] | None:
        """
        Get video processing status.

        Args:
            video_id: YouTube video ID

        Returns:
            Status dictionary or None
        """
        if not self.youtube:
            if not self.authenticate():
                return None

        try:
            response = self.youtube.videos().list(
                part="status,processingDetails",
                id=video_id,
            ).execute()

            if not response.get("items"):
                return None

            video = response["items"][0]
            return {
                "privacy": video["status"]["privacyStatus"],
                "upload_status": video["status"]["uploadStatus"],
                "processing": video.get("processingDetails", {}),
            }

        except Exception as e:
            logger.error(f"Failed to get video status: {e}")
            return None

    # =========================================================================
    # Channel Information
    # =========================================================================

    def get_channel_info(self) -> dict[str, Any] | None:
        """
        Get authenticated user's channel information.

        Returns:
            Channel information dictionary
        """
        if not self.youtube:
            if not self.authenticate():
                return None

        try:
            response = self.youtube.channels().list(
                part="snippet,statistics",
                mine=True,
            ).execute()

            if not response.get("items"):
                return None

            channel = response["items"][0]
            return {
                "id": channel["id"],
                "title": channel["snippet"]["title"],
                "description": channel["snippet"]["description"],
                "subscribers": channel["statistics"].get("subscriberCount"),
                "videos": channel["statistics"].get("videoCount"),
                "views": channel["statistics"].get("viewCount"),
            }

        except Exception as e:
            logger.error(f"Failed to get channel info: {e}")
            return None

    # =========================================================================
    # Utilities
    # =========================================================================

    def generate_description(
        self,
        script: str,
        game_data: dict[str, Any],
        include_timestamps: bool = True,
    ) -> str:
        """
        Generate a YouTube description from script and game data.

        Args:
            script: Video script content
            game_data: Game information
            include_timestamps: Whether to add timestamp markers

        Returns:
            Formatted description
        """
        lines = []

        # Main description
        if script:
            # Take first paragraph as summary
            summary = script.split("\n\n")[0][:500]
            lines.append(summary)
            lines.append("")

        # Game details
        lines.append("GAME DETAILS")
        lines.append(f"Date: {game_data.get('game_date', 'N/A')}")
        lines.append(f"Teams: {game_data.get('away_team', 'Away')} @ {game_data.get('home_team', 'Home')}")
        lines.append(f"Final: {game_data.get('away_score', 0)} - {game_data.get('home_score', 0)}")
        lines.append("")

        # Timestamps placeholder
        if include_timestamps:
            lines.append("TIMESTAMPS")
            lines.append("0:00 - Intro")
            lines.append("0:15 - Game Highlights")
            lines.append("1:00 - Key Statistics")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append("Generated by MLB Video Pipeline")
        lines.append("#MLB #Baseball #GameRecap")

        return "\n".join(lines)

    def is_authenticated(self) -> bool:
        """Check if currently authenticated."""
        return self.youtube is not None
