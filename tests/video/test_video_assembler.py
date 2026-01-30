import pytest
from unittest.mock import MagicMock, patch
from src.video.video_assembler import VideoAssembler
import numpy as np

class TestVideoAssembler:
    
    @pytest.fixture
    def mock_moviepy(self):
        with patch('src.video.video_assembler.VideoFileClip') as vfc, \
             patch('src.video.video_assembler.AudioFileClip') as afc, \
             patch('src.video.video_assembler.ImageClip') as ic, \
             patch('src.video.video_assembler.CompositeVideoClip') as cvc, \
             patch('src.video.video_assembler.ColorClip') as cc:
            yield vfc, afc, ic, cvc, cc

    def test_assemble_video_flow(self, mock_moviepy, tmp_path):
        vfc, afc, ic, cvc, cc = mock_moviepy
        
        # Configure fluent interface for CompositeVideoClip
        mock_video = cvc.return_value
        mock_video.set_audio.return_value = mock_video
        mock_video.set_duration.return_value = mock_video
        
        # Setup Mocks
        mock_asset_manager = MagicMock()
        mock_asset_manager.get_background_video.return_value = None # Trigger ColorClip fallback
        mock_asset_manager.fetch_team_logo.return_value = "logo.png"
        
        # Mock Audio Duration
        mock_audio = MagicMock()
        mock_audio.duration = 10.0
        afc.return_value = mock_audio
        
        assembler = VideoAssembler(mock_asset_manager, output_dir=str(tmp_path))
        
        # Execute
        output = assembler.assemble_video(
            script_data={"prediction": {"prediction": "Win"}}, 
            audio_path="audio.mp3",
            charts=["chart1.png"]
        )
        
        # Assertions
        assert output is not None
        assert "final_video.mp4" in output
        
        # Verify calls
        afc.assert_called_with("audio.mp3")
        cc.assert_called() # Fallback background
        ic.assert_called() # For charts and text
        cvc.assert_called() # Composite
        mock_video.write_videofile.assert_called()

    def test_create_text_image(self, tmp_path):
        assembler = VideoAssembler(MagicMock(), output_dir=str(tmp_path))
        img_array = assembler._create_text_image("Test\nLine 2")
        
        assert isinstance(img_array, np.ndarray)
        assert img_array.shape[1] == 1080 # Width
        assert img_array.shape[2] == 4 # RGBA channel
