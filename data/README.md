# Data

This directory contains sample cryptocurrency market data used to
demonstrate the data-processing pipeline.

The complete dataset is not included in this repository.
# Sample Data

This directory contains approximately one year of daily BTC-USD
and ETH-USD OHLCV observations for demonstrating the data
validation, preprocessing, feature-engineering, and model-input
pipeline.

The full research dataset is not included in this repository.
Users can reproduce the dataset using the data-collection script
provided in `src/data_collection.py`.

## Columns

- `Date`: Trading date
- `Asset`: Cryptocurrency ticker
- `Open`: Opening price
- `High`: Highest daily price
- `Low`: Lowest daily price
- `Close`: Closing price
- `Volume`: Daily trading volume

## Data Source

Historical market data obtained through Yahoo Finance.
The data remains subject to the original provider's terms.
