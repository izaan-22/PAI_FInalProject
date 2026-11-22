"""Open-Meteo current conditions and latest WFP Pakistan market observations."""
import io
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests
from .utils import write_json


class MarketDataFetcher:
    def __init__(self, cfg):
        self.cfg = cfg
        self.api_url = "https://api.open-meteo.com/v1/forecast"
        self.cache_file_path = Path(cfg["cache"])

    def _read_cache(self):
        try:
            value = json.loads(self.cache_file_path.read_text(encoding="utf8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write_cache(self, cache):
        write_json(self.cache_file_path, cache)

    @staticmethod
    def _get(url, params=None):
        response = requests.get(url, params=params, timeout=(4, 12))
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError(f"Unexpected HTTP {response.status_code}")
        return response

    def _weather(self):
        data = self._get(self.api_url, {"latitude":self.cfg["latitude"], "longitude":self.cfg["longitude"],
            "current":"temperature_2m,relative_humidity_2m", "timezone":"UTC"}).json()["current"]
        temp, humidity = float(data["temperature_2m"]), float(data["relative_humidity_2m"])
        if not math.isfinite(temp) or not 0 <= humidity <= 100:
            raise ValueError("Invalid weather measurement")
        return {"observed_at_utc":data["time"], "temperature_c":temp, "humidity_pct":humidity}

    def _prices(self):
        catalog = self._get("https://data.humdata.org/api/3/action/package_show",
                            {"id":"wfp-food-prices-for-pakistan"}).json()["result"]["resources"]
        source = next((r["url"] for r in catalog if "wfp_food_prices" in (r.get("name") or "").lower()
                       and (r.get("format") or "").lower() == "csv"), None)
        if not source:
            raise ValueError("Price resource missing from HDX catalog")
        df = pd.read_csv(io.BytesIO(self._get(source).content), low_memory=False)
        for col, setting in [("market","market"), ("commodity","commodity"), ("pricetype","price_type")]:
            df = df[df[col].str.casefold() == self.cfg[setting].casefold()]
        df = df.assign(date=pd.to_datetime(df.date, errors="coerce"), price=pd.to_numeric(df.price, errors="coerce"))
        df = df.dropna(subset=["date","price","currency","unit"]).sort_values("date")
        if df.empty:
            raise ValueError("No quotes for this market, commodity and price type")
        row = df.iloc[-1]
        prior = df[(df.date < row.date) & (df.unit == row.unit) & (df.currency == row.currency)]
        return {"observation_date":row.date.date().isoformat(), "price":float(row.price),
                "currency":row.currency,"unit":row.unit,"market":row.market,"commodity":row.commodity,
                "previous_price":float(prior.iloc[-1].price) if len(prior) else None}

    def fetch_live_market_data(self, offline=False):
        cache = self._read_cache()
        keys = {"weather":f"weather|{self.cfg['latitude']}|{self.cfg['longitude']}",
                "prices":"|".join(["prices", self.cfg["market"], self.cfg["commodity"], self.cfg["price_type"]])}
        result = {}
        for name, fetch in [("weather",self._weather), ("prices",self._prices)]:
            key = keys[name]
            if not offline:
                try:
                    entry = {"fetched_at":datetime.now(timezone.utc).isoformat(), "data":fetch()}
                    cache[key] = entry
                    result[name] = entry | {"status":"live"}
                    try:
                        self._write_cache(cache)
                    except OSError as exc:
                        logging.warning("Cache write failed: %s", exc)
                    continue
                except (requests.RequestException, ValueError, KeyError, TypeError, IndexError) as exc:
                    logging.warning("%s unavailable: %s", name, exc)
            entry = cache.get(key)
            if isinstance(entry, dict) and isinstance(entry.get("data"), dict) and entry.get("fetched_at"):
                result[name] = entry | {"status":"cached"}
            else:
                result[name] = {"status":"unavailable","data":{}}
        return result
