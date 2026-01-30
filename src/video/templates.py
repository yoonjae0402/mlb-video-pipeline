"""
MLB Video Pipeline - Video Templates

Defines video layouts, styles, and visual configurations.
Templates control the look and feel of generated videos.

Usage:
    from src.video.templates import TEMPLATES, VideoTemplate

    template = TEMPLATES["modern_dark"]
    generator.create_video(template=template, ...)
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ColorScheme:
    """Color configuration for video template."""
    background: str = "#1a1a2e"
    primary: str = "#16213e"
    secondary: str = "#0f3460"
    accent: str = "#e94560"
    text_primary: str = "#ffffff"
    text_secondary: str = "#a0a0a0"
    highlight: str = "#00ff88"


@dataclass
class Typography:
    """Font configuration for video template."""
    title_font: str = "Arial-Bold"
    body_font: str = "Arial"
    stat_font: str = "Courier-Bold"
    title_size: int = 72
    subtitle_size: int = 48
    body_size: int = 36
    stat_size: int = 42


@dataclass
class Layout:
    """Layout configuration for video template."""
    width: int = 1080
    height: int = 1920
    margin: int = 60
    header_height: int = 200
    footer_height: int = 150
    safe_zone: int = 100  # Safe area from edges


@dataclass
class Animation:
    """Animation settings for video template."""
    transition_duration: float = 0.5
    text_fade_in: float = 0.3
    stat_reveal_delay: float = 0.2
    chart_animation_duration: float = 1.0


@dataclass
class VideoTemplate:
    """
    Complete video template configuration.

    Defines all visual aspects of generated videos including
    colors, fonts, layout, and animations.
    """
    name: str
    description: str
    colors: ColorScheme = field(default_factory=ColorScheme)
    typography: Typography = field(default_factory=Typography)
    layout: Layout = field(default_factory=Layout)
    animation: Animation = field(default_factory=Animation)

    # Background settings
    background_type: str = "solid"  # solid, gradient, image
    background_gradient_direction: str = "vertical"

    # Overlay settings
    use_vignette: bool = True
    vignette_strength: float = 0.3

    # Stats display
    stat_box_style: str = "rounded"  # rounded, square, pill
    stat_box_opacity: float = 0.8

    # Logo/branding
    logo_position: str = "bottom_right"
    show_watermark: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert template to dictionary."""
        return {
            "name": self.name,
            "colors": {
                "background": self.colors.background,
                "primary": self.colors.primary,
                "secondary": self.colors.secondary,
                "accent": self.colors.accent,
                "text_primary": self.colors.text_primary,
                "text_secondary": self.colors.text_secondary,
            },
            "layout": {
                "width": self.layout.width,
                "height": self.layout.height,
                "margin": self.layout.margin,
            },
        }


# =============================================================================
# Pre-defined Templates
# =============================================================================

TEMPLATES = {
    "modern_dark": VideoTemplate(
        name="Modern Dark",
        description="Sleek dark theme with vibrant accents",
        colors=ColorScheme(
            background="#0d1117",
            primary="#161b22",
            secondary="#21262d",
            accent="#58a6ff",
            text_primary="#f0f6fc",
            text_secondary="#8b949e",
            highlight="#3fb950",
        ),
        typography=Typography(
            title_font="Arial-Bold",
            title_size=72,
            body_size=36,
        ),
        use_vignette=True,
    ),

    "classic_baseball": VideoTemplate(
        name="Classic Baseball",
        description="Traditional baseball aesthetic with warm colors",
        colors=ColorScheme(
            background="#2d2d2d",
            primary="#8b4513",
            secondary="#a0522d",
            accent="#ffd700",
            text_primary="#ffffff",
            text_secondary="#d4d4d4",
            highlight="#ff6347",
        ),
        typography=Typography(
            title_font="Georgia-Bold",
            body_font="Georgia",
            stat_font="Courier-Bold",
        ),
        stat_box_style="square",
    ),

    "electric_neon": VideoTemplate(
        name="Electric Neon",
        description="High-energy neon style for highlight videos",
        colors=ColorScheme(
            background="#0a0a0a",
            primary="#1a1a1a",
            secondary="#2a2a2a",
            accent="#00ffff",
            text_primary="#ffffff",
            text_secondary="#888888",
            highlight="#ff00ff",
        ),
        animation=Animation(
            transition_duration=0.3,
            text_fade_in=0.2,
        ),
        use_vignette=True,
        vignette_strength=0.5,
    ),

    "clean_minimal": VideoTemplate(
        name="Clean Minimal",
        description="Simple, professional look",
        colors=ColorScheme(
            background="#ffffff",
            primary="#f5f5f5",
            secondary="#e0e0e0",
            accent="#2196f3",
            text_primary="#212121",
            text_secondary="#757575",
            highlight="#4caf50",
        ),
        use_vignette=False,
        stat_box_style="pill",
        stat_box_opacity=0.9,
    ),

    "team_branded": VideoTemplate(
        name="Team Branded",
        description="Template for team-specific branding (customize colors)",
        colors=ColorScheme(
            background="#1a1a2e",
            primary="#16213e",
            secondary="#0f3460",
            accent="#e94560",
            text_primary="#ffffff",
            text_secondary="#cccccc",
        ),
    ),
}


# =============================================================================
# Template Helpers
# =============================================================================

def get_template(name: str) -> VideoTemplate:
    """
    Get a template by name.

    Args:
        name: Template name

    Returns:
        VideoTemplate instance

    Raises:
        ValueError: If template not found
    """
    if name not in TEMPLATES:
        raise ValueError(f"Template '{name}' not found. Available: {list(TEMPLATES.keys())}")
    return TEMPLATES[name]


def create_team_template(
    team_id: int,
    base_template: str = "team_branded"
) -> VideoTemplate:
    """
    Create a template with team colors.

    Args:
        team_id: MLB team ID
        base_template: Base template to modify

    Returns:
        VideoTemplate with team colors
    """
    from config.league_config import MLB_TEAMS

    team = MLB_TEAMS.get(team_id)
    if not team:
        return TEMPLATES[base_template]

    primary, secondary = team.get("colors", ("#000000", "#ffffff"))

    template = TEMPLATES[base_template]
    template.colors.accent = primary
    template.colors.secondary = secondary
    template.name = f"{team['name']} Theme"

    return template


def list_templates() -> list[dict[str, str]]:
    """List available templates with descriptions."""
    return [
        {"name": name, "description": template.description}
        for name, template in TEMPLATES.items()
    ]


# =============================================================================
# Scene Configurations
# =============================================================================

SCENE_CONFIGS = {
    "intro": {
        "duration": 3.0,
        "elements": ["title", "date", "teams"],
        "transition_in": "fade",
        "transition_out": "slide_left",
    },
    "stats_display": {
        "duration": 5.0,
        "elements": ["stat_boxes", "chart"],
        "transition_in": "slide_up",
        "transition_out": "fade",
    },
    "player_highlight": {
        "duration": 4.0,
        "elements": ["player_name", "stats", "photo_placeholder"],
        "transition_in": "zoom",
        "transition_out": "slide_right",
    },
    "chart_scene": {
        "duration": 6.0,
        "elements": ["chart", "caption"],
        "transition_in": "fade",
        "transition_out": "fade",
    },
    "outro": {
        "duration": 3.0,
        "elements": ["call_to_action", "branding"],
        "transition_in": "slide_up",
        "transition_out": "fade",
    },
}
