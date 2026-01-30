import pytest
from pathlib import Path
from src.video.chart_generator import ChartGenerator

class TestChartGenerator:
    
    @pytest.fixture
    def generator(self, tmp_path):
        return ChartGenerator(output_dir=str(tmp_path))

    def test_generate_trend_chart(self, generator):
        data = [0.2, 0.25, 0.28, 0.3, 0.29]
        labels = ["G1", "G2", "G3", "G4", "G5"]
        
        path = generator.generate_trend_chart(data, labels, "Batting Trend")
        
        assert path != ""
        assert Path(path).exists()
        assert Path(path).name == "trend.png"

    def test_generate_matchup_graphic(self, generator):
        path = generator.generate_matchup_graphic(
            "Judge", ".320 BA",
            "Cole", "2.10 ERA"
        )
        
        assert path != ""
        assert Path(path).exists()
        assert Path(path).name == "matchup.png"
