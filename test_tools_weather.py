"""
Unit tests for the weather tool (Phase 2, step 4.8).

No internet is used: get_json is replaced by a fake that answers like Open-Meteo,
and one test checks how the real request URL is built (with urlopen mocked).

Usage:  python -m unittest test_tools_weather -v
"""

import io
import json
import unittest
import urllib.error
from unittest import mock

from jarvis.tools import weather
from jarvis.tools.registry import registry

FORECAST = {
    "current": {"temperature_2m": 7.4, "apparent_temperature": 4.2, "weather_code": 61, "wind_speed_10m": 15.3},
    "daily": {"weather_code": [63, 2], "temperature_2m_max": [9.2, 11.6], "temperature_2m_min": [4.1, 3.5],
              "precipitation_probability_max": [80, 10]},
}
LONDON = {"results": [{"name": "London", "country": "United Kingdom", "latitude": 51.51, "longitude": -0.13}]}


class FakeOpenMeteo:
    """Stands in for get_json: answers geocoding and forecast requests, records them."""

    def __init__(self, places=LONDON, forecast=FORECAST, error=None):
        self.places, self.forecast, self.error = places, forecast, error
        self.calls = []

    def __call__(self, url, params):
        self.calls.append((url, params))
        if self.error:
            raise self.error
        return self.places if url == weather.GEOCODING_URL else self.forecast


class TestWeather(unittest.TestCase):

    def run_tool(self, **kwargs):
        fake_args = {k: kwargs.pop(k) for k in ("places", "forecast", "error") if k in kwargs}
        self.fake = FakeOpenMeteo(**fake_args)
        with mock.patch.object(weather, "get_json", self.fake):
            return weather.weather(**kwargs)

    def test_today_at_home(self):
        result = self.run_tool()
        self.assertTrue(result.success)
        self.assertEqual(result.message,
                         "Buizingen, Belgium. Now: 7 degrees (feels like 4), light rain, wind 15 kilometres "
                         "per hour. Today: rain, 4 to 9 degrees, 80 percent chance of rain.")
        self.assertEqual(len(self.fake.calls), 1)   # home: no place lookup needed

    def test_tomorrow(self):
        message = self.run_tool(day="tomorrow").message
        self.assertEqual(message, "Buizingen, Belgium. Tomorrow: partly cloudy, 4 to 12 degrees, "
                                  "10 percent chance of rain.")

    def test_home_words_skip_the_lookup(self):
        for name in ["Buizingen", "home", " HOME "]:
            with self.subTest(name=name):
                self.run_tool(location=name)
                self.assertEqual(len(self.fake.calls), 1)

    def test_other_place(self):
        message = self.run_tool(location="London").message
        self.assertTrue(message.startswith("London, United Kingdom."))
        url, params = self.fake.calls[0]
        self.assertEqual(url, weather.GEOCODING_URL)
        self.assertEqual(params["name"], "London")
        self.assertEqual(self.fake.calls[1][1]["latitude"], 51.51)

    def test_unknown_place(self):
        result = self.run_tool(location="Atlantis", places={})
        self.assertFalse(result.success)
        self.assertIn("Could not find a place called 'Atlantis'", result.message)

    def test_no_rain_chance(self):
        forecast = json.loads(json.dumps(FORECAST))
        forecast["daily"]["precipitation_probability_max"] = [None, None]
        self.assertNotIn("chance of rain", self.run_tool(forecast=forecast).message)

    def test_offline(self):
        result = self.run_tool(error=urllib.error.URLError("no internet"))
        self.assertFalse(result.success)
        self.assertIn("Is the Mac online?", result.message)

    def test_strange_answer(self):
        result = self.run_tool(forecast={"current": {}})
        self.assertFalse(result.success)
        self.assertIn("could not read", result.message)


class TestRequest(unittest.TestCase):

    def test_url_and_timeout(self):
        answer = mock.MagicMock()
        answer.__enter__.return_value = io.BytesIO(b'{"ok": true}')
        with mock.patch("jarvis.tools.weather.urllib.request.urlopen", return_value=answer) as urlopen:
            self.assertEqual(weather.get_json(weather.FORECAST_URL, {"latitude": 50.75, "current": "a,b"}),
                             {"ok": True})
        request = urlopen.call_args[0][0]
        self.assertEqual(request.full_url, weather.FORECAST_URL + "?latitude=50.75&current=a%2Cb")
        self.assertEqual(urlopen.call_args[1]["timeout"], weather.HTTP_TIMEOUT_S)


class TestRegistration(unittest.TestCase):

    def test_safe_and_validated(self):
        self.assertEqual(registry.get("weather").risk, "safe")
        with mock.patch.object(weather, "get_json") as get_json:
            self.assertFalse(registry.execute("weather", {"day": "yesterday"}).success)
        get_json.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
