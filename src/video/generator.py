"""
MLB Video Pipeline - Video Generator

Creates videos by compositing text, statistics, charts, and audio.
Uses MoviePy for video editing and Matplotlib/Pillow for graphics.

Usage:
    from src.video.generator import VideoGenerator

    generator = VideoGenerator()

    video_path = generator.create_video(
        audio_path=audio_path,
        scenes=[
            {"type": "intro", "title": "Game Recap", "subtitle": "NYY vs BOS"},
            {"type": "stats", "data": batting_stats},
            {"type": "chart", "chart_type": "bar", "data": run_data},
        ]
    )
"""

from pathlib import Path
from typing import Any
from datetime import datetime
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from config.settings import settings
from config.league_config import MLB_TEAMS
from src.video.templates import VideoTemplate, TEMPLATES, get_template
from src.utils.logger import get_logger
from src.utils.validators import sanitize_filename


logger = get_logger(__name__)


class VideoGenerator:
    """
    Generate videos with stats, charts, and narration.

    Creates professional-looking videos without using copyrighted
    game footage, relying instead on statistics and graphics.
    """

    def __init__(
        self,
        template: VideoTemplate | str = "modern_dark",
        output_dir: Path | None = None,
    ):
        """
        Initialize the video generator.

        Args:
            template: VideoTemplate instance or template name
            output_dir: Directory for video output
        """
        if isinstance(template, str):
            self.template = get_template(template)
        else:
            self.template = template

        self.output_dir = output_dir or settings.video_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Video settings
        self.width = self.template.layout.width
        self.height = self.template.layout.height
        self.fps = settings.video_fps

        logger.info(f"VideoGenerator initialized: {self.width}x{self.height}, template={self.template.name}")

    # =========================================================================
    # Image Generation
    # =========================================================================

    def _create_background(self) -> Image.Image:
        """Create background image based on template."""
        colors = self.template.colors

        if self.template.background_type == "gradient":
            return self._create_gradient_background(
                colors.background,
                colors.primary,
                self.template.background_gradient_direction
            )
        else:
            return Image.new("RGB", (self.width, self.height), colors.background)

    def _create_gradient_background(
        self,
        color1: str,
        color2: str,
        direction: str = "vertical"
    ) -> Image.Image:
        """Create a gradient background."""
        img = Image.new("RGB", (self.width, self.height))
        draw = ImageDraw.Draw(img)

        # Parse colors
        c1 = self._hex_to_rgb(color1)
        c2 = self._hex_to_rgb(color2)

        if direction == "vertical":
            for y in range(self.height):
                ratio = y / self.height
                r = int(c1[0] * (1 - ratio) + c2[0] * ratio)
                g = int(c1[1] * (1 - ratio) + c2[1] * ratio)
                b = int(c1[2] * (1 - ratio) + c2[2] * ratio)
                draw.line([(0, y), (self.width, y)], fill=(r, g, b))
        else:
            for x in range(self.width):
                ratio = x / self.width
                r = int(c1[0] * (1 - ratio) + c2[0] * ratio)
                g = int(c1[1] * (1 - ratio) + c2[1] * ratio)
                b = int(c1[2] * (1 - ratio) + c2[2] * ratio)
                draw.line([(x, 0), (x, self.height)], fill=(r, g, b))

        return img

    def _hex_to_rgb(self, hex_color: str) -> tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _get_font(self, font_type: str = "body", size: int | None = None) -> ImageFont.FreeTypeFont:
        """Get font for drawing text."""
        typography = self.template.typography

        font_map = {
            "title": (typography.title_font, typography.title_size),
            "subtitle": (typography.title_font, typography.subtitle_size),
            "body": (typography.body_font, typography.body_size),
            "stat": (typography.stat_font, typography.stat_size),
        }

        font_name, default_size = font_map.get(font_type, (typography.body_font, typography.body_size))
        size = size or default_size

        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            # Fallback to default font
            return ImageFont.load_default()

    # =========================================================================
    # Scene Generation
    # =========================================================================

    def create_intro_frame(
        self,
        title: str,
        subtitle: str = "",
        date: str = "",
    ) -> Image.Image:
        """
        Create an intro/title frame.

        Args:
            title: Main title text
            subtitle: Subtitle text
            date: Date text

        Returns:
            PIL Image
        """
        img = self._create_background()
        draw = ImageDraw.Draw(img)
        colors = self.template.colors

        # Title
        title_font = self._get_font("title")
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (self.width - title_width) // 2
        title_y = self.height // 3

        draw.text((title_x, title_y), title, fill=colors.text_primary, font=title_font)

        # Subtitle
        if subtitle:
            subtitle_font = self._get_font("subtitle")
            sub_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
            sub_width = sub_bbox[2] - sub_bbox[0]
            sub_x = (self.width - sub_width) // 2
            sub_y = title_y + 100

            draw.text((sub_x, sub_y), subtitle, fill=colors.text_secondary, font=subtitle_font)

        # Date
        if date:
            date_font = self._get_font("body")
            date_bbox = draw.textbbox((0, 0), date, font=date_font)
            date_width = date_bbox[2] - date_bbox[0]
            date_x = (self.width - date_width) // 2
            date_y = self.height - 200

            draw.text((date_x, date_y), date, fill=colors.text_secondary, font=date_font)

        return img

    def create_stats_frame(
        self,
        stats: list[dict[str, Any]],
        title: str = "Game Statistics",
    ) -> Image.Image:
        """
        Create a statistics display frame.

        Args:
            stats: List of {"label": str, "value": str/number}
            title: Section title

        Returns:
            PIL Image
        """
        img = self._create_background()
        draw = ImageDraw.Draw(img)
        colors = self.template.colors
        margin = self.template.layout.margin

        # Title
        title_font = self._get_font("subtitle")
        draw.text((margin, margin), title, fill=colors.accent, font=title_font)

        # Stats boxes
        stat_font = self._get_font("stat")
        label_font = self._get_font("body")

        box_width = (self.width - 3 * margin) // 2
        box_height = 150
        start_y = 200

        for i, stat in enumerate(stats[:6]):  # Max 6 stats
            row = i // 2
            col = i % 2

            x = margin + col * (box_width + margin)
            y = start_y + row * (box_height + margin)

            # Draw box
            box_color = self._hex_to_rgb(colors.secondary)
            draw.rounded_rectangle(
                [x, y, x + box_width, y + box_height],
                radius=15,
                fill=box_color,
            )

            # Draw label
            label = stat.get("label", "")
            draw.text((x + 20, y + 20), label, fill=colors.text_secondary, font=label_font)

            # Draw value
            value = str(stat.get("value", ""))
            draw.text((x + 20, y + 70), value, fill=colors.text_primary, font=stat_font)

        return img

    def create_chart_frame(
        self,
        data: dict[str, Any],
        chart_type: str = "bar",
        title: str = "",
    ) -> Image.Image:
        """
        Create a chart frame.

        Args:
            data: {"labels": [...], "values": [...]}
            chart_type: Type of chart (bar, line, pie)
            title: Chart title

        Returns:
            PIL Image
        """
        colors = self.template.colors

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(self.width/100, self.height/100), dpi=100)
        fig.patch.set_facecolor(colors.background)
        ax.set_facecolor(colors.primary)

        labels = data.get("labels", [])
        values = data.get("values", [])

        if chart_type == "bar":
            bars = ax.bar(labels, values, color=colors.accent)
            ax.bar_label(bars, fmt='%.1f', color=colors.text_primary)
        elif chart_type == "line":
            ax.plot(labels, values, color=colors.accent, linewidth=3, marker='o')
        elif chart_type == "pie":
            ax.pie(values, labels=labels, colors=[colors.accent, colors.secondary, colors.highlight])

        # Styling
        ax.tick_params(colors=colors.text_secondary)
        ax.spines['bottom'].set_color(colors.text_secondary)
        ax.spines['left'].set_color(colors.text_secondary)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if title:
            ax.set_title(title, color=colors.text_primary, fontsize=24, pad=20)

        # Convert to PIL Image
        fig.tight_layout()
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plt.close(fig)

        return Image.fromarray(img_array, 'RGBA').convert('RGB').resize((self.width, self.height))

    def create_player_frame(
        self,
        player_name: str,
        stats: dict[str, Any],
        team_id: int | None = None,
    ) -> Image.Image:
        """
        Create a player spotlight frame.

        Args:
            player_name: Player's name
            stats: Player statistics
            team_id: Team ID for colors

        Returns:
            PIL Image
        """
        img = self._create_background()
        draw = ImageDraw.Draw(img)
        colors = self.template.colors

        # Apply team colors if available
        if team_id and team_id in MLB_TEAMS:
            team = MLB_TEAMS[team_id]
            accent = team.get("colors", (colors.accent,))[0]
        else:
            accent = colors.accent

        # Player name
        name_font = self._get_font("title")
        name_bbox = draw.textbbox((0, 0), player_name, font=name_font)
        name_width = name_bbox[2] - name_bbox[0]
        name_x = (self.width - name_width) // 2

        draw.text((name_x, 150), player_name, fill=accent, font=name_font)

        # Stats
        stat_y = 350
        stat_font = self._get_font("stat")
        label_font = self._get_font("body")

        for key, value in list(stats.items())[:5]:
            display_key = key.replace("_", " ").upper()

            # Label
            draw.text((100, stat_y), display_key, fill=colors.text_secondary, font=label_font)

            # Value
            draw.text((100, stat_y + 50), str(value), fill=colors.text_primary, font=stat_font)

            stat_y += 150

        return img

    # =========================================================================
    # Video Composition
    # =========================================================================

    def create_video(
        self,
        scenes: list[dict[str, Any]],
        audio_path: Path | None = None,
        output_name: str | None = None,
        include_transitions: bool = True,
    ) -> Path:
        """
        Create a complete video from scenes.

        Args:
            scenes: List of scene configurations
            audio_path: Path to narration audio
            output_name: Output filename
            include_transitions: Whether to add transitions

        Returns:
            Path to generated video
        """
        try:
            from moviepy.editor import (
                ImageClip,
                AudioFileClip,
                concatenate_videoclips,
                CompositeVideoClip,
            )
        except ImportError:
            logger.error("MoviePy not installed. Please install moviepy.")
            raise

        logger.info(f"Creating video with {len(scenes)} scenes")

        clips = []
        temp_files = []

        for i, scene in enumerate(scenes):
            scene_type = scene.get("type", "intro")
            duration = scene.get("duration", 5.0)

            # Generate frame based on scene type
            if scene_type == "intro":
                frame = self.create_intro_frame(
                    title=scene.get("title", ""),
                    subtitle=scene.get("subtitle", ""),
                    date=scene.get("date", ""),
                )
            elif scene_type == "stats":
                frame = self.create_stats_frame(
                    stats=scene.get("data", []),
                    title=scene.get("title", "Statistics"),
                )
            elif scene_type == "chart":
                frame = self.create_chart_frame(
                    data=scene.get("data", {}),
                    chart_type=scene.get("chart_type", "bar"),
                    title=scene.get("title", ""),
                )
            elif scene_type == "player":
                frame = self.create_player_frame(
                    player_name=scene.get("player_name", ""),
                    stats=scene.get("stats", {}),
                    team_id=scene.get("team_id"),
                )
            else:
                # Default to intro
                frame = self.create_intro_frame(title=scene.get("title", ""))

            # Save frame temporarily
            temp_path = Path(tempfile.mktemp(suffix=".png"))
            frame.save(temp_path)
            temp_files.append(temp_path)

            # Create video clip
            clip = ImageClip(str(temp_path)).set_duration(duration)

            # Add fade transition if requested
            if include_transitions and i > 0:
                clip = clip.crossfadein(0.5)

            clips.append(clip)

        # Concatenate all clips
        video = concatenate_videoclips(clips, method="compose")

        # Add audio if provided
        if audio_path and audio_path.exists():
            audio = AudioFileClip(str(audio_path))
            # Adjust video duration to match audio
            if audio.duration > video.duration:
                # Extend last clip
                pass  # Keep video as is, audio will be trimmed
            video = video.set_audio(audio.set_duration(video.duration))

        # Generate output path
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"video_{timestamp}"

        output_name = sanitize_filename(output_name)
        output_path = self.output_dir / f"{output_name}.mp4"

        # Export video
        logger.info(f"Exporting video to {output_path}")
        video.write_videofile(
            str(output_path),
            fps=self.fps,
            codec="libx264",
            audio_codec="aac",
            logger=None,  # Suppress moviepy's verbose output
        )

        # Cleanup temp files
        for temp_file in temp_files:
            try:
                temp_file.unlink()
            except Exception:
                pass

        video.close()

        logger.info(f"Video created: {output_path}")
        return output_path

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def create_thumbnail(
        self,
        title: str,
        subtitle: str = "",
        output_name: str | None = None,
    ) -> Path:
        """
        Create a video thumbnail.

        Args:
            title: Thumbnail title
            subtitle: Subtitle text
            output_name: Output filename

        Returns:
            Path to thumbnail image
        """
        # Thumbnail is 16:9 for YouTube
        thumb_width = 1280
        thumb_height = 720

        # Temporarily adjust dimensions
        original_width, original_height = self.width, self.height
        self.width, self.height = thumb_width, thumb_height

        img = self.create_intro_frame(title, subtitle)

        # Restore dimensions
        self.width, self.height = original_width, original_height

        # Save thumbnail
        thumbnails_dir = settings.outputs_dir / "thumbnails"
        thumbnails_dir.mkdir(parents=True, exist_ok=True)

        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"thumb_{timestamp}"

        output_path = thumbnails_dir / f"{output_name}.png"
        img.save(output_path)

        logger.info(f"Thumbnail created: {output_path}")
        return output_path

    def set_template(self, template: VideoTemplate | str) -> None:
        """Change the video template."""
        if isinstance(template, str):
            self.template = get_template(template)
        else:
            self.template = template

        self.width = self.template.layout.width
        self.height = self.template.layout.height

        logger.info(f"Template changed to: {self.template.name}")
