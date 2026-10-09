
import numpy as np
import pandas as pd

# Make the dataset reproducible
np.random.seed(42)

number_of_houses = 1200

locations = [
    "Kozhikode",
    "Kochi",
    "Thiruvananthapuram",
    "Thrissur",
    "Kannur"
]

# Approximate synthetic price rates in rupees per square foot
location_rates = {
    "Kozhikode": 4500,
    "Kochi": 6500,
    "Thiruvananthapuram": 5500,
    "Thrissur": 4000,
    "Kannur": 3500
}

data = []

for i in range(number_of_houses):
    area = np.random.randint(500, 3501)
    bedrooms = np.random.randint(1, 7)
    bathrooms = np.random.randint(1, 5)
    floors = np.random.randint(1, 4)
    parking = np.random.randint(0, 4)
    house_age = np.random.randint(0, 41)

    location = np.random.choice(locations)

    base_price = area * location_rates[location]

    # Adjust price according to house features
    bedroom_adjustment = bedrooms * 150000
    bathroom_adjustment = bathrooms * 100000
    floor_adjustment = floors * 200000
    parking_adjustment = parking * 125000
    age_adjustment = house_age * 50000

    price = (
        base_price
        + bedroom_adjustment
        + bathroom_adjustment
        + floor_adjustment
        + parking_adjustment
        - age_adjustment
    )

    # Add variation to make the data less perfect
    noise = np.random.normal(0, 500000)

    price = max(500000, price + noise)

    data.append({
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Floors": floors,
        "Parking": parking,
        "Location": location,
        "House_Age": house_age,
        "Price": round(price, 2)
    })

df = pd.DataFrame(data)

df.to_csv("house_prices.csv", index=False)

print("Dataset generated successfully!")
print("Total houses:", len(df))
print("\nFirst five records:")
print(df.head())
print("\nDataset saved as house_prices.csv")
