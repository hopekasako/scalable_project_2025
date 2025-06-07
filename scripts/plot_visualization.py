import pandas as pd
import matplotlib.pyplot as plt

# read cleaned data
df = pd.read_csv("data/netflix_cleaned.csv")

# change type of date
df["Date"] = pd.to_datetime(df["Date"])

# sort
df = df.sort_values("Date")

# Draw diagram
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Close"], label="Close Price", color="blue")
plt.title("Netflix Stock Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/netflix_close_price_plot.png")
plt.show()
