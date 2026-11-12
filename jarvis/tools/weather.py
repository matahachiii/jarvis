"""
Weather tool: current conditions and forecast from Open-Meteo (free, no API key).

Default location: Buizingen, Belgium. Another place is looked up by name with
Open-Meteo's free geocoding service. Only a place name or coordinates leave the Mac.
Uses only the standard library (urllib), so no new dependency.
"""

import json
import urllib.error
import urllib.parse
import urllib.request

from jarvis.tools.registry import ToolResult, tool

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
HOME = ("Buizingen, Belgium", 50.75, 4.26)   # approximate coordinates of sir's home
HOME_NAMES = {"buizingen", "home", "here"}
HTTP_TIMEOUT_S = 8
DAYS = ["today", "tomorrow"]

# WMO weather interpretation codes, as Open-Meteo returns them
WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "light freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "light freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "violent rain showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with heavy hail",
}


# ------------------------------------------------------------ helpers
def get_json(url: str, params: dict) -> dict:
    """GET a URL with query parameters and return the JSON answer (raises on network errors)."""
    full_url = url + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(full_url, headers={"User-Agent": "Jarvis personal assistant"})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
        return json.load(response)


def find_place(name: str) -> tuple[str, float, float] | None:
    """('London, United Kingdom', 51.5, -0.12) or None if the place is unknown."""
    data = get_json(GEOCODING_URL, {"name": name, "count": 1, "language": "en", "format": "json"})
    results = data.get("results") or []
    if not results:
        return None
    first = results[0]
    label = first["name"] + (f", {first['country']}" if first.get("country") else "")
    return label, first["latitude"], first["longitude"]


def describe_sky(code) -> str:
    return WEATHER_CODES.get(code, "unknown weather")


def build_report(place: str, data: dict, day: str) -> str:
    """Turn Open-Meteo's JSON into short, spoken-friendly sentences."""
    daily = data["daily"]
    i = 0 if day == "today" else 1
    low = round(daily["temperature_2m_min"][i])
    high = round(daily["temperature_2m_max"][i])
    rain = daily["precipitation_probability_max"][i]
    day_text = f"{day.capitalize()}: {describe_sky(daily['weather_code'][i])}, {low} to {high} degrees"
    if rain is not None:
        day_text += f", {rain} percent chance of rain"
    day_text += "."

    if day != "today":
        return f"{place}. {day_text}"
    now = data["current"]
    now_text = (f"Now: {round(now['temperature_2m'])} degrees "
                f"(feels like {round(now['apparent_temperature'])}), {describe_sky(now['weather_code'])}, "
                f"wind {round(now['wind_speed_10m'])} kilometres per hour.")
    return f"{place}. {now_text} {day_text}"


# ------------------------------------------------------------ tool
@tool("Get the current weather and the forecast for today or tomorrow. "
      "Default location is sir's home in Buizingen, Belgium.",
      {"location": ("string", "A city or town, only when sir names a place other than home"),
       "day": ("string", "Which day", DAYS)},
      optional=("location", "day"), timeout_s=20)
def weather(location=None, day="today"):
    try:
        if location and location.strip().lower() not in HOME_NAMES:
            place = find_place(location)
            if place is None:
                return ToolResult(False, f"Could not find a place called '{location}'.")
        else:
            place = HOME
        label, latitude, longitude = place
        data = get_json(FORECAST_URL, {
            "latitude": latitude, "longitude": longitude,
            "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto", "forecast_days": 2,
        })
        return ToolResult(True, build_report(label, data, day))
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return ToolResult(False, "Could not reach the weather service. Is the Mac online?")
    except (KeyError, IndexError, TypeError, ValueError):
        return ToolResult(False, "The weather service sent an answer I could not read.")
