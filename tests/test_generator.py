"""
Tests for video generator.

Run: pytest tests/test_generator.py -v
"""

import pytest
from pathlib import Path
import tempfile
from PIL import Image

from src.video.generator import VideoGenerator
from src.video.templates import (
    VideoTemplate,
    ColorScheme,
    Typography,
    Layout,
    TEMPLATES,
    get_template,
    create_team_template,
)


class TestVideoTemplate:
    """Tests for VideoTemplate class."""

    def test_default_template(self):
        """Test template with default values."""
        template = VideoTemplate(name="test", description="Test template")

        assert template.colors.background == "#1a1a2e"
        assert template.layout.width == 1080
        assert template.layout.height == 1920

    def test_custom_colors(self):
        """Test template with custom colors."""
        colors = ColorScheme(
            background="#000000",
            accent="#ff0000",
        )
        template = VideoTemplate(
            name="custom",
            description="Custom template",
            colors=colors,
        )

        assert template.colors.background == "#000000"
        assert template.colors.accent == "#ff0000"

    def test_to_dict(self):
        """Test template serialization."""
        template = VideoTemplate(name="test", description="Test")
        d = template.to_dict()

        assert d["name"] == "test"
        assert "colors" in d
        assert "layout" in d


class TestTemplates:
    """Tests for pre-defined templates."""

    def test_templates_exist(self):
        """Test that default templates exist."""
        assert "modern_dark" in TEMPLATES
        assert "classic_baseball" in TEMPLATES
        assert "clean_minimal" in TEMPLATES

    def test_get_template(self):
        """Test getting template by name."""
        template = get_template("modern_dark")

        assert template.name == "Modern Dark"
        assert template.colors is not None

    def test_get_invalid_template(self):
        """Test getting invalid template raises error."""
        with pytest.raises(ValueError):
            get_template("nonexistent_template")

    def test_create_team_template(self):
        """Test creating team-branded template."""
        template = create_team_template(147)  # Yankees

        assert "Yankees" in template.name or template.name == "Team Branded"


class TestVideoGenerator:
    """Tests for VideoGenerator class."""

    @pytest.fixture
    def generator(self):
        """Create generator with temp output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = VideoGenerator(
                template="modern_dark",
                output_dir=Path(tmpdir),
            )
            yield gen

    def test_init(self, generator):
        """Test generator initialization."""
        assert generator.width == 1080
        assert generator.height == 1920
        assert generator.template.name == "Modern Dark"

    def test_hex_to_rgb(self, generator):
        """Test hex color conversion."""
        rgb = generator._hex_to_rgb("#ff0000")
        assert rgb == (255, 0, 0)

        rgb = generator._hex_to_rgb("#00ff00")
        assert rgb == (0, 255, 0)

    def test_create_background(self, generator):
        """Test background creation."""
        bg = generator._create_background()

        assert isinstance(bg, Image.Image)
        assert bg.size == (generator.width, generator.height)

    def test_create_gradient_background(self, generator):
        """Test gradient background creation."""
        bg = generator._create_gradient_background(
            "#000000",
            "#ffffff",
            "vertical"
        )

        assert isinstance(bg, Image.Image)
        assert bg.size == (generator.width, generator.height)

    def test_create_intro_frame(self, generator):
        """Test intro frame creation."""
        frame = generator.create_intro_frame(
            title="Test Title",
            subtitle="Test Subtitle",
            date="2024-07-04",
        )

        assert isinstance(frame, Image.Image)
        assert frame.size == (generator.width, generator.height)

    def test_create_stats_frame(self, generator):
        """Test stats frame creation."""
        stats = [
            {"label": "Hits", "value": 10},
            {"label": "Home Runs", "value": 2},
            {"label": "RBI", "value": 5},
        ]

        frame = generator.create_stats_frame(stats, title="Game Stats")

        assert isinstance(frame, Image.Image)

    def test_create_chart_frame(self, generator):
        """Test chart frame creation."""
        data = {
            "labels": ["A", "B", "C"],
            "values": [10, 20, 15],
        }

        frame = generator.create_chart_frame(
            data,
            chart_type="bar",
            title="Test Chart",
        )

        assert isinstance(frame, Image.Image)

    def test_create_player_frame(self, generator):
        """Test player frame creation."""
        frame = generator.create_player_frame(
            player_name="Aaron Judge",
            stats={"HR": 35, "RBI": 80, "AVG": ".290"},
            team_id=147,
        )

        assert isinstance(frame, Image.Image)

    def test_set_template(self, generator):
        """Test changing template."""
        generator.set_template("clean_minimal")

        assert generator.template.name == "Clean Minimal"

    def test_create_thumbnail(self, generator):
        """Test thumbnail creation."""
        thumb_path = generator.create_thumbnail(
            title="Test Thumbnail",
            subtitle="Subtitle",
        )

        assert thumb_path.exists()
        assert thumb_path.suffix == ".png"

        # Verify dimensions (16:9 for YouTube)
        img = Image.open(thumb_path)
        assert img.size == (1280, 720)


class TestVideoGeneratorIntegration:
    """Integration tests for video generation."""

    @pytest.mark.integration
    def test_create_simple_video(self):
        """Test creating a simple video."""
        pytest.importorskip("moviepy")

        with tempfile.TemporaryDirectory() as tmpdir:
            generator = VideoGenerator(
                template="modern_dark",
                output_dir=Path(tmpdir),
            )

            scenes = [
                {
                    "type": "intro",
                    "title": "Test Video",
                    "subtitle": "Integration Test",
                    "duration": 2.0,
                },
                {
                    "type": "stats",
                    "data": [
                        {"label": "Test", "value": 100},
                    ],
                    "duration": 2.0,
                },
            ]

            video_path = generator.create_video(
                scenes=scenes,
                output_name="test_video",
            )

            assert video_path.exists()
            assert video_path.suffix == ".mp4"
