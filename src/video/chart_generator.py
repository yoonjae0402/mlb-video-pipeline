import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Style settings for dark mode video
plt.style.use('dark_background')
sns.set_palette("bright")

class ChartGenerator:
    """
    Generates static image charts for video overlays.
    """
    
    def __init__(self, output_dir: str = "data/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_trend_chart(
        self, 
        data: list[float], 
        labels: list[str], 
        title: str, 
        filename: str = "trend.png"
    ) -> str:
        """
        Generate a line chart showing a trend (e.g., rolling batting average).
        """
        output_path = self.output_dir / filename
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot
            sns.lineplot(x=range(len(data)), y=data, ax=ax, marker='o', linewidth=3, color='#00d2be')
            
            # Style
            ax.set_title(title, fontsize=24, color='white', pad=20, fontweight='bold')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, fontsize=12)
            ax.tick_params(colors='white')
            
            # Remove borders for clean overlay look
            sns.despine(left=True, bottom=True)
            ax.grid(axis='y', alpha=0.2)
            
            # Save with transparent background
            plt.tight_layout()
            plt.savefig(output_path, transparent=True, dpi=150)
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            return ""

    def generate_matchup_graphic(
        self,
        batter_name: str,
        batter_stat: str,
        pitcher_name: str,
        pitcher_stat: str,
        filename: str = "matchup.png"
    ) -> str:
        """
        Generate a head-to-head text graphic card.
        """
        output_path = self.output_dir / filename
        
        try:
            # Create a semi-transparent dark card
            width, height = 800, 400
            img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Draw semi-transparent background box
            draw.rectangle([(0, 0), (width, height)], fill=(20, 20, 20, 220))
            
            # We don't have custom fonts loaded yet, use default for MVP
            # In production, we'd load .ttf files from assets/fonts/
            
            # Draw Text (Simple placement)
            # Title
            draw.text((width//2, 40), "KEY MATCHUP", fill="cyan", anchor="mm", font_size=40)
            
            # Batter
            draw.text((100, 150), batter_name, fill="white", font_size=36)
            draw.text((100, 200), batter_stat, fill="yellow", font_size=30)
            
            # VS
            draw.text((width//2, 200), "VS", fill="white", anchor="mm", font_size=60)
            
            # Pitcher
            draw.text((width-100, 150), pitcher_name, fill="white", anchor="ra", font_size=36)
            draw.text((width-100, 200), pitcher_stat, fill="yellow", anchor="ra", font_size=30)
            
            img.save(output_path)
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error generating graphic: {e}")
            return ""
