import pytest
from unittest.mock import MagicMock, patch
from src.video.asset_manager import AssetManager

class TestAssetManager:
    
    @pytest.fixture
    def manager(self, tmp_path):
        return AssetManager(cache_dir=str(tmp_path))

    @patch('src.video.asset_manager.requests.get')
    def test_fetch_team_logo_download(self, mock_get, manager):
        # Mock successful download
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"<svg>Logo</svg>"
        mock_get.return_value = mock_response
        
        path = manager.fetch_team_logo(147) # Yankees ID
        
        assert path is not None
        assert "147.svg" in path
        mock_get.assert_called_once()
        
        # Verify file content
        with open(path, "rb") as f:
            assert f.read() == b"<svg>Logo</svg>"

    @patch('src.video.asset_manager.requests.get')
    def test_fetch_team_logo_cached(self, mock_get, manager):
        # Create dummy cached file
        logo_path = manager.logos_dir / "147.svg"
        logo_path.write_bytes(b"Cached")
        
        path = manager.fetch_team_logo(147)
        
        assert path == str(logo_path)
        mock_get.assert_not_called()

    @patch('src.video.asset_manager.requests.get')
    def test_fetch_player_headshot(self, mock_get, manager):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"PNGData"
        mock_get.return_value = mock_response
        
        path = manager.fetch_player_headshot(999999)
        
        assert path is not None
        assert "999999.png" in path
        
    def test_get_background_video_missing(self, manager):
        path = manager.get_background_video("non_existent")
        assert path is None
