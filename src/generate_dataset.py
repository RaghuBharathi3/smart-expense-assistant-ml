"""Generate a balanced synthetic dataset of realistic Indian expense records.

Produces at least 1,200 unique expense records across four categories
(Food, Travel, Shopping, Utilities) with realistic INR amounts, dates,
Indian merchants, and varied natural-language descriptions.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
RECORDS_PER_CATEGORY = 300
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "expenses.csv"


def _build_food_templates() -> list[tuple[str, tuple[int, int]]]:
    """Return varied Food description templates and realistic INR amount ranges."""
    return [
        ("Swiggy food delivery from {place}", (140, 850)),
        ("Ordered {item} online via Zomato", (180, 950)),
        ("Dinner with friends at {place}", (450, 2400)),
        ("Lunch {item} at {place}", (120, 650)),
        ("Morning coffee and {item} at {place}", (90, 420)),
        ("Paid online for {item} delivery from Swiggy", (160, 780)),
        ("Zomato order for {item} and beverages", (200, 900)),
        ("Evening snacks and tea near {area}", (40, 220)),
        ("Weekend biryani order from {place}", (250, 850)),
        ("Breakfast {item} at {place}", (80, 350)),
        ("Ordered pizza online from {place}", (300, 1100)),
        ("Quick {item} takeaway from {place}", (110, 520)),
        ("Monthly office canteen meal card recharge", (800, 2500)),
        ("Bakery cake and pastries for {occasion}", (350, 1400)),
        ("Late night {item} delivery via EatSure", (190, 680)),
        ("Fresh juice and {item} at {area} cafe", (120, 400)),
        ("South Indian thali lunch at {place}", (150, 480)),
        ("Dessert and ice cream order from {place}", (140, 550)),
        ("Paid bill for team lunch at {place}", (900, 3200)),
        ("Street food {item} at {area} market", (50, 260)),
        ("Ordered {item} on Amazon Fresh", (220, 950)),
        ("Monthly meal subscription renewal online", (1200, 3000)),
    ]


def _build_travel_templates() -> list[tuple[str, tuple[int, int]]]:
    """Return varied Travel description templates and realistic INR amount ranges."""
    return [
        ("Uber ride to {dest}", (110, 650)),
        ("Ola cab booking from {area} to {dest}", (130, 780)),
        ("Chennai Metro ticket to {dest}", (20, 80)),
        ("Metro smart card recharge at {area} station", (200, 1000)),
        ("Rapido bike taxi to {dest}", (45, 190)),
        ("IRCTC train ticket booking to {city}", (350, 2800)),
        ("Auto rickshaw fare from {area} to {dest}", (50, 240)),
        ("Airport cab pickup via Uber", (450, 1450)),
        ("RedBus sleeper ticket to {city}", (550, 1900)),
        ("Petrol refill for bike at {fuel_station}", (200, 1100)),
        ("Diesel fuel payment at {fuel_station}", (1000, 4500)),
        ("FASTag toll recharge for highway trip", (300, 1200)),
        ("Local suburban train pass renewal", (150, 600)),
        ("Ola Outstation trip to {city}", (1800, 5500)),
        ("Flight ticket booking to {city}", (3200, 9500)),
        ("Parking fee at {dest}", (40, 200)),
        ("Monthly metro commute pass payment", (600, 1800)),
        ("Cab ride back home from {dest}", (140, 620)),
        ("BMTC bus pass monthly renewal", (800, 1500)),
        ("Two-wheeler service and chain lube at {area}", (350, 1400)),
        ("Paid cab bill online after trip to {dest}", (180, 720)),
        ("Booked ride on Paytm Travel to {city}", (400, 2200)),
    ]


def _build_shopping_templates() -> list[tuple[str, tuple[int, int]]]:
    """Return varied Shopping description templates and realistic INR amount ranges."""
    return [
        ("Purchased {product} on Amazon", (350, 6500)),
        ("Ordered {product} online from Flipkart", (400, 5800)),
        ("Bought {apparel} from Myntra", (650, 3500)),
        ("Ajio online order for {apparel}", (550, 2900)),
        ("Purchased {product} at Croma store", (800, 12000)),
        ("Bought {home_item} from DMart", (300, 2600)),
        ("Nykaa order for {personal_care}", (350, 2200)),
        ("Decathlon purchase of {sports_item}", (450, 3800)),
        ("Paid for {product} at Reliance Digital", (900, 9500)),
        ("Ordered {home_item} via Amazon Prime", (250, 2400)),
        ("Bought {apparel} at Westside {area}", (750, 4200)),
        ("Purchased {product} from local electronics shop", (300, 3200)),
        ("Bookstore purchase of {book_item}", (250, 1400)),
        ("Bought gift {product} for {occasion}", (400, 3000)),
        ("Lenskart order for {eyewear}", (1200, 4800)),
        ("Flipkart Big Billion sale purchase of {product}", (700, 8500)),
        ("Stationery and {office_supply} order online", (180, 1200)),
        ("Bought {apparel} during weekend mall visit", (800, 4500)),
        ("Ordered {personal_care} from Amazon", (220, 1500)),
        ("Purchased {sports_item} online", (500, 3400)),
        ("Paid online for {product} order", (350, 4500)),
        ("Monthly Prime shopping delivery order of {home_item}", (299, 1999)),
    ]


def _build_utilities_templates() -> list[tuple[str, tuple[int, int]]]:
    """Return varied Utilities description templates and realistic INR amount ranges."""
    return [
        ("Monthly electricity bill payment via {pay_app}", (650, 4200)),
        ("TANGEDCO electricity bill for {month}", (700, 3900)),
        ("BESCOM power bill online payment", (800, 4500)),
        ("Airtel postpaid mobile recharge", (399, 1199)),
        ("Jio prepaid mobile plan recharge", (239, 899)),
        ("ACT Fibernet broadband monthly bill", (699, 1499)),
        ("JioFiber internet subscription renewal", (599, 1499)),
        ("Indane LPG cooking gas cylinder booking", (850, 1150)),
        ("HP Gas cylinder refill payment", (850, 1150)),
        ("Monthly water bill payment for apartment", (250, 950)),
        ("Tata Play DTH recharge online", (250, 750)),
        ("Vi mobile recharge for {month}", (249, 799)),
        ("Piped natural gas bill via {pay_app}", (400, 1600)),
        ("Apartment maintenance and utility charge", (1500, 4500)),
        ("MSEDCL home electricity bill payment", (750, 4100)),
        ("BSNL landline and broadband bill", (499, 1299)),
        ("Airtel Xstream Wi-Fi bill payment", (799, 1399)),
        ("Municipal property water tax online", (500, 2200)),
        ("Prepaid electricity meter recharge", (500, 2500)),
        ("Airtel DTH annual pack renewal", (1800, 3600)),
        ("Paid monthly utility bill via Amazon Pay", (450, 3200)),
        ("Ordered cooking gas refill online via {pay_app}", (850, 1150)),
    ]


def _fill_slots(template: str, rng: random.Random) -> str:
    """Fill placeholder tokens in a description template using controlled vocabulary."""
    slots = {
        "place": rng.choice([
            "Saravana Bhavan", "A2B", "Biryani Blues", "Dominos", "KFC",
            "Haldirams", "Chai Point", "Third Wave Coffee", "Barbeque Nation",
            "Sangeetha Veg", "Wow Momo", "McDonalds", "Subway", "Pizza Hut",
            "Udupi Upahar", "Paradise Biryani", "Cream Stone", "Belgian Waffle",
        ]),
        "item": rng.choice([
            "masala dosa", "chicken biryani", "paneer wrap", "veg thali",
            "cold coffee", "pav bhaji", "chole bhature", "idli vada",
            "margherita pizza", "fried rice", "roll", "sandwich", "momos",
            "filter coffee", "samosa", "noodles", "pasta", "shawarma",
        ]),
        "area": rng.choice([
            "Adyar", "Indiranagar", "T Nagar", "Koramangala", "Velachery",
            "Bandra", "Anna Nagar", "HSR Layout", "Whitefield", "Andheri",
            "Guindy", "Jayanagar", "Kothrud", "Gachibowli", "Salt Lake",
        ]),
        "occasion": rng.choice([
            "birthday", "anniversary", "team celebration", "weekend gathering", "family dinner",
        ]),
        "dest": rng.choice([
            "office", "airport", "railway station", "tech park", "city center",
            "Central Station", "IT Park", "bus stand", "Guindy", "MG Road",
            "Cyber Hub", "Electronic City", "Tambaram", "Kempegowda terminal",
        ]),
        "city": rng.choice([
            "Bengaluru", "Chennai", "Hyderabad", "Pune", "Mumbai",
            "Coimbatore", "Madurai", "Mysuru", "Kochi", "Jaipur",
        ]),
        "fuel_station": rng.choice([
            "IndianOil", "HPCL", "BPCL", "Shell", "Nayara Energy",
        ]),
        "product": rng.choice([
            "wireless headphones", "USB-C cable", "laptop backpack", "smartwatch",
            "bluetooth speaker", "power bank", "mechanical keyboard", "wireless mouse",
            "phone case", "desk lamp", "earbuds", "SSD drive", "water bottle",
            "trimmer", "electric kettle", "tripod stand",
        ]),
        "apparel": rng.choice([
            "cotton t-shirt", "running shoes", "denim jeans", "formal shirt",
            "kurta set", "hoodie", "sneakers", "winter jacket", "chinos", "backpack",
        ]),
        "home_item": rng.choice([
            "bedsheet set", "storage containers", "non-stick pan", "bath towels",
            "laundry basket", "kitchen organizer", "cushion covers", "door mat",
        ]),
        "personal_care": rng.choice([
            "sunscreen", "face wash", "moisturizer", "shampoo", "hair serum", "perfume",
        ]),
        "sports_item": rng.choice([
            "yoga mat", "badminton racket", "dumbbells", "cycling gloves", "football",
        ]),
        "book_item": rng.choice([
            "programming book", "fiction novel", "exam prep guide", "hardcover notebook",
        ]),
        "eyewear": rng.choice([
            "blue-light glasses", "polarized sunglasses", "prescription frames",
        ]),
        "office_supply": rng.choice([
            "printer paper", "gel pens", "sticky notes", "desk organizer",
        ]),
        "pay_app": rng.choice([
            "GPay", "PhonePe", "Paytm", "Amazon Pay", "CRED", "BHIM UPI",
        ]),
        "month": rng.choice([
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]),
    }
    return template.format(**slots)


def generate_dataset(
    records_per_category: int = RECORDS_PER_CATEGORY,
    seed: int = SEED,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Generate a balanced, deduplicated synthetic expense dataset and save to CSV."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    category_builders = {
        "Food": _build_food_templates(),
        "Travel": _build_travel_templates(),
        "Shopping": _build_shopping_templates(),
        "Utilities": _build_utilities_templates(),
    }

    start_date = date(2025, 1, 1)
    total_days = 450

    rows: list[dict[str, object]] = []
    seen_descriptions: set[str] = set()

    for category, templates in category_builders.items():
        count = 0
        attempts = 0
        max_attempts = records_per_category * 40

        while count < records_per_category and attempts < max_attempts:
            attempts += 1
            template, (low, high) = rng.choice(templates)
            description = _fill_slots(template, rng)

            # Optional natural phrasing prefixes/suffixes for diversity
            if rng.random() < 0.25:
                suffix = rng.choice([
                    "via UPI", "using credit card", "via debit card",
                    "for personal expense", "today", "this morning", "yesterday",
                ])
                if suffix not in description.lower():
                    description = f"{description} {suffix}"

            norm_key = description.strip().lower()
            if norm_key in seen_descriptions:
                continue

            seen_descriptions.add(norm_key)
            amount = round(float(np_rng.uniform(low, high)), 2)
            tx_date = start_date + timedelta(days=rng.randint(0, total_days))

            rows.append({
                "description": description,
                "amount": amount,
                "date": tx_date.isoformat(),
                "category": category,
            })
            count += 1

        if count < records_per_category:
            raise RuntimeError(
                f"Could only generate {count}/{records_per_category} unique records for {category}."
            )

    rng.shuffle(rows)

    df = pd.DataFrame(rows)
    df.insert(0, "expense_id", [f"EXP-{i:04d}" for i in range(1, len(df) + 1)])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    dataframe = generate_dataset()
    print(f"Generated {len(dataframe)} records at {OUTPUT_PATH}")
    print(dataframe["category"].value_counts())
