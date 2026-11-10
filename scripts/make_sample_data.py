"""Creates the two sample datasets in sample_data/ (run: python scripts/make_sample_data.py)."""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "sample_data"
rng = np.random.default_rng(42)


def make_zomato(rows: int = 800) -> pd.DataFrame:
    cities = ["Bangalore", "Chennai", "Mumbai", "Delhi", "Hyderabad", "Kolkata"]
    city_bias = dict(zip(cities, [0.25, 0.10, 0.0, -0.05, 0.12, -0.10]))
    city_weights = [0.25, 0.18, 0.2, 0.17, 0.12, 0.08]
    cuisines = ["North Indian", "South Indian", "Chinese", "Italian", "Biryani", "Cafe", "Fast Food", "Desserts"]
    city = rng.choice(cities, rows, p=city_weights)
    price_range = rng.choice([1, 2, 3, 4], rows, p=[0.3, 0.4, 0.2, 0.1])
    rating = np.clip(rng.normal(3.8, 0.45, rows) + np.array([city_bias[c] for c in city]) + 0.08 * price_range, 1.5, 5.0)
    votes = np.clip((rating - 2) * 120 + rng.normal(0, 90, rows), 5, None).astype(int)
    df = pd.DataFrame({
        "restaurant_name": [f"Restaurant {i:04d}" for i in range(1, rows + 1)],
        "city": city, "rating": rating.round(1), "votes": votes,
        "cuisine": rng.choice(cuisines, rows), "price_range": price_range,
    })
    df.loc[rng.choice(rows, 25, replace=False), "rating"] = np.nan   # some missing ratings
    return df


def make_sales() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    regions = {"North": 1.0, "South": 1.35, "East": 0.8, "West": 1.1}
    products = {"Laptop": 900, "Phone": 600, "Tablet": 350, "Headphones": 120}
    records = []
    for date in dates:
        season = 1 + 0.25 * np.sin((date.month - 3) / 12 * 2 * np.pi) + 0.004 * date.dayofyear
        for region, factor in regions.items():
            product = rng.choice(list(products))
            units = max(1, int(rng.poisson(6 * factor * season)))
            records.append({"date": date.date().isoformat(), "region": region, "product": product,
                            "units": units, "sales": round(units * products[product] * rng.uniform(0.9, 1.1), 2)})
    return pd.DataFrame(records)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    make_zomato().to_csv(OUT / "zomato_sample.csv", index=False)
    make_sales().to_excel(OUT / "sales_sample.xlsx", index=False)
    print("Created", [p.name for p in OUT.iterdir()])
