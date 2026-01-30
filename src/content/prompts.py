"""
MLB Video Pipeline - GPT Prompt Templates

Prompt templates for generating different types of video scripts.
Each prompt is designed to produce engaging, accurate baseball content.

Usage:
    from src.content.prompts import PROMPTS, format_prompt

    prompt = format_prompt("game_recap", game_data=data)
"""

from typing import Any


# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPT = """You are an expert MLB analyst and sports broadcaster with years of experience.
You create engaging, informative video scripts about baseball that:
- Use natural, conversational language suitable for voiceover
- Include specific statistics and player names
- Build narrative tension and highlight key moments
- Appeal to both casual fans and hardcore baseball enthusiasts
- Avoid speculation about injuries or personal matters
- Focus on on-field performance and statistics

Your scripts should be formatted for text-to-speech narration:
- Use natural pauses (indicated by "...")
- Avoid complex abbreviations that don't sound natural when spoken
- Spell out numbers for better TTS (e.g., "fifteen strikeouts" not "15 K's")
- Keep sentences concise for easier listening"""


# =============================================================================
# Content Type Prompts
# =============================================================================

PROMPTS = {
    "game_recap": """Create a {duration}-second video script recapping this MLB game:

**Game Details:**
- {away_team} ({away_record}) at {home_team} ({home_record})
- Final Score: {away_team} {away_score}, {home_team} {home_score}
- Date: {game_date}
- Venue: {venue}

**Key Statistics:**
{key_stats}

**Notable Performances:**
{notable_performances}

Requirements:
- Start with an attention-grabbing hook
- Cover the most impactful moments
- Highlight standout individual performances
- Include relevant statistics naturally
- End with implications for standings/playoff race if relevant
- Target word count: {word_count} words (approximately {duration} seconds at 150 WPM)

Write the script now:""",

    "player_spotlight": """Create a {duration}-second video script highlighting this player's recent performance:

**Player Information:**
- Name: {player_name}
- Team: {team_name}
- Position: {position}

**Recent Statistics ({time_period}):**
{recent_stats}

**Season Statistics:**
{season_stats}

**Context:**
{context}

Requirements:
- Open with what makes this player's current performance noteworthy
- Compare to historical performance or league averages
- Include specific game highlights or memorable moments
- Discuss what's behind their success/struggles
- Target word count: {word_count} words

Write the script now:""",

    "prediction": """Create a {duration}-second video script with predictions for this upcoming MLB game:

**Matchup:**
- {away_team} at {home_team}
- Date: {game_date}
- Starting Pitchers: {away_pitcher} vs {home_pitcher}

**Team Statistics:**
{team_stats}

**Head-to-Head History:**
{head_to_head}

**Model Predictions:**
{model_predictions}

**Factors to Consider:**
{factors}

Requirements:
- Present the prediction with confidence but appropriate hedging
- Explain the key factors driving the prediction
- Highlight players to watch
- Include a bold take or underrated storyline
- Target word count: {word_count} words

Write the script now:""",

    "weekly_roundup": """Create a {duration}-second video script summarizing the week in MLB:

**Week of {week_start} to {week_end}**

**Standings Movement:**
{standings_changes}

**Top Performances:**
{top_performances}

**Key Storylines:**
{storylines}

**Upcoming Games to Watch:**
{upcoming}

Requirements:
- Cover the most important developments across the league
- Balance coverage between AL and NL
- Highlight breakout players and struggling stars
- Mention playoff implications
- Target word count: {word_count} words

Write the script now:""",

    "stat_breakdown": """Create a {duration}-second video script breaking down this baseball statistic:

**Statistic: {stat_name}**

**Current Leaders:**
{leaders}

**Historical Context:**
{historical}

**Why It Matters:**
{importance}

**Surprising Findings:**
{surprises}

Requirements:
- Explain the statistic in accessible terms
- Use comparisons to make numbers meaningful
- Include interesting outliers or trends
- Connect stats to real game impact
- Target word count: {word_count} words

Write the script now:""",

    "matchup_preview": """Create a {duration}-second video script previewing this MLB series:

**Series Details:**
- {away_team} at {home_team}
- Dates: {series_dates}
- Games: {num_games}

**Team Form:**
{team_form}

**Key Matchups:**
{key_matchups}

**Injury Report:**
{injuries}

**What to Watch:**
{watch_points}

Requirements:
- Set up the stakes for this series
- Identify the most important battles within the matchup
- Make specific predictions for individual games
- Engage fans of both teams
- Target word count: {word_count} words

Write the script now:""",
}


# =============================================================================
# Helper Functions
# =============================================================================

def format_prompt(
    content_type: str,
    duration: int = 60,
    **kwargs: Any
) -> str:
    """
    Format a prompt template with provided data.

    Args:
        content_type: Type of content (game_recap, prediction, etc.)
        duration: Target video duration in seconds
        **kwargs: Data to fill the template

    Returns:
        Formatted prompt string

    Raises:
        ValueError: If content_type is not recognized
    """
    if content_type not in PROMPTS:
        raise ValueError(
            f"Unknown content type: {content_type}. "
            f"Available: {list(PROMPTS.keys())}"
        )

    # Calculate approximate word count
    words_per_second = 2.5  # Average speaking pace for narration
    word_count = int(duration * words_per_second)

    # Format the template
    template = PROMPTS[content_type]

    return template.format(
        duration=duration,
        word_count=word_count,
        **kwargs
    )


def get_system_prompt() -> str:
    """Get the system prompt for GPT."""
    return SYSTEM_PROMPT


def estimate_duration(text: str, wpm: int = 150) -> int:
    """
    Estimate narration duration for a script.

    Args:
        text: Script text
        wpm: Words per minute speaking rate

    Returns:
        Estimated duration in seconds
    """
    word_count = len(text.split())
    return int((word_count / wpm) * 60)


# =============================================================================
# Prompt Variations
# =============================================================================

TONE_MODIFIERS = {
    "exciting": "Use energetic, enthusiastic language with dramatic pauses.",
    "analytical": "Focus on statistics and strategic analysis with a measured tone.",
    "casual": "Keep it conversational and accessible for casual fans.",
    "dramatic": "Build tension and emphasize the stakes of each moment.",
}

AUDIENCE_MODIFIERS = {
    "general": "Explain baseball concepts briefly for general audiences.",
    "hardcore": "Assume deep baseball knowledge; use advanced stats freely.",
    "fantasy": "Include fantasy baseball relevance and ownership percentages.",
}


# =============================================================================
# Series Middle Video Prompts
# =============================================================================

MIDDLE_ANALYSIS_PROMPT = """
Generate a 30-second post-game analysis script:
- Lead with the final score and winning team
- Highlight 1-2 key moments that decided the game
- Feature the top performer with specific stats
- Add brief context (streak extended, standings impact)

Keep it concise, engaging, and suitable for voiceover narration.
Use natural pauses (...) where appropriate.
"""

MIDDLE_PREVIEW_PROMPT = """
Generate a 30-second pre-game preview script with AI prediction:
- Announce tomorrow's matchup (teams and starting pitchers)
- State our AI's prediction for the featured player
- Explain 2-3 specific reasons WHY we predict this outcome
- Use specific numbers (batting average, recent stats, matchup history)

IMPORTANT: Make the "why" very clear and compelling.

Format example:
"Our AI predicts [player] will [perform above/at/below average] tomorrow.
First, [reason with specific stat].
Second, [reason with specific stat].
And third, [reason with specific stat].
Keep an eye on this matchup."
"""


# =============================================================================
# Series End Video Prompts
# =============================================================================

END_SERIES_SUMMARY_PROMPT = """
Generate a 20-second series wrap-up script:
- State the series result (e.g., "Dodgers take the series 2-1")
- Name the series MVP with their key stat line
- One brief takeaway sentence about the series

This section is COMPRESSED - be very concise. Every word counts.
Approximately 50 words maximum.
"""

END_NEXT_GAME_PROMPT = """
Generate a 20-second next opponent preview script:
- Announce the next opponent
- Mention one key matchup (pitcher or player to watch)
- Include a brief AI prediction with 1-2 reasons

This section is COMPRESSED - essential info only.
Approximately 50 words maximum.
"""


def apply_modifiers(
    base_prompt: str,
    tone: str = "exciting",
    audience: str = "general"
) -> str:
    """
    Apply tone and audience modifiers to a prompt.

    Args:
        base_prompt: Original prompt
        tone: Tone modifier key
        audience: Audience modifier key

    Returns:
        Modified prompt
    """
    modifiers = []

    if tone in TONE_MODIFIERS:
        modifiers.append(f"Tone: {TONE_MODIFIERS[tone]}")
    if audience in AUDIENCE_MODIFIERS:
        modifiers.append(f"Audience: {AUDIENCE_MODIFIERS[audience]}")

    if modifiers:
        modifier_text = "\n".join(modifiers)
        return f"{base_prompt}\n\n{modifier_text}"

    return base_prompt
