import requests
from streamlit.testing.v1 import AppTest

from src.schema import ROOT


def test_dashboard_connects_analysis_explorer_and_export(client, monkeypatch):
    from unittest.mock import patch
    from urllib.parse import urlparse

    monkeypatch.setenv("CLOUDVISION_API", "http://testserver")
    with (
        patch("requests.get", side_effect=lambda url, **kw: client.get(urlparse(url).path)),
        patch(
            "requests.post",
            side_effect=lambda url, files, data, **kw: client.post(
                urlparse(url).path, files=files, data=data
            ),
        ),
    ):
        ui = AppTest.from_file(str(ROOT / "app/dashboard.py"), default_timeout=20).run()
        assert not ui.exception
        ui.button[0].click().run()
        assert ui.title[0].value == "Cloud analysis"
        ui.button[0].click().run()
        assert not ui.exception and any(m.label == "Cloud coverage" for m in ui.metric)
        ui.sidebar.radio[0].set_value("Segmentation explorer").run()
        assert not ui.exception and any(t.label == "Pixel inspector" for t in ui.tabs)
        ui.sidebar.radio[0].set_value("Model comparison").run()
        assert not ui.exception and len(ui.dataframe) >= 2
        ui.sidebar.radio[0].set_value("Uncertainty").run()
        assert not ui.exception and any(m.label == "Temperature" for m in ui.metric)


def test_service_outage_explained_in_ui(monkeypatch):
    from unittest.mock import patch

    monkeypatch.setenv("CLOUDVISION_API", "http://offline.test")
    from streamlit import cache_data

    cache_data.clear()
    with patch("requests.get", side_effect=requests.ConnectionError()):
        ui = AppTest.from_file(str(ROOT / "app/dashboard.py"), default_timeout=15).run()
        ui.button[0].click().run()
        assert not ui.exception and "unavailable" in ui.error[0].value
    cache_data.clear()
