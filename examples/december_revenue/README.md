# December Revenue Analysis Demo

This example demonstrates the AutoViz Agent analyzing a sample December revenue dataset.

## Dataset

The `december_revenue.csv` file contains:

- `date`: Daily dates for December
- `revenue`: Revenue values
- `region`: Geographic region (North, South, East, West)
- `product_category`: Product categories (Electronics, Clothing, Home, Food)

## Running the Demo

```bash
# From the repository root
autoviz run examples/december_revenue/december_revenue.csv \
  "What are the revenue trends in December?" \
  --output-dir examples/december_revenue/output
```

## Expected Output

The analysis produces:

1. **Charts** (5-10 visualizations):
   - Revenue trend over time (line chart)
   - Revenue by region (bar chart)
   - Revenue by product category (bar chart)
   - Revenue distribution (histogram)
   - Correlation heatmap

2. **Report** (`report.md`):
   - Executive summary
   - Key insights
   - Statistical findings
   - Chart references
   - Plan provenance links

3. **Provenance Artifacts**:
   - `plan_template.json`: Original template
   - `plan_adapted.json`: Adapted plan
   - `plan_diff.md`: Changes with rationale
   - `tool_calls.json`: Executed tool calls
   - `execution_log.json`: Detailed execution log

## Determinism Verification

Run the analysis twice and compare outputs:

```bash
# Run 1
autoviz run examples/december_revenue/december_revenue.csv \
  "What are the revenue trends in December?" \
  --output-dir run1 --seed 42

# Run 2
autoviz run examples/december_revenue/december_revenue.csv \
  "What are the revenue trends in December?" \
  --output-dir run2 --seed 42

# Compare (should be identical)
diff -r run1 run2
```

## Sample Data

Create the sample CSV:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Generate December dates
dates = pd.date_range(start='2024-12-01', end='2024-12-31', freq='D')

# Generate data
data = []
regions = ['North', 'South', 'East', 'West']
categories = ['Electronics', 'Clothing', 'Home', 'Food']

for date in dates:
    for region in regions:
        for category in categories:
            revenue = np.random.randint(1000, 10000)
            data.append({
                'date': date,
                'region': region,
                'product_category': category,
                'revenue': revenue
            })

df = pd.DataFrame(data)
df.to_csv('examples/december_revenue/december_revenue.csv', index=False)
print(f"Created dataset with {len(df)} rows")
```

## Analysis Questions

Try different questions:

- "What are the revenue trends in December?"
- "Which regions have the highest revenue?"
- "Compare revenue across product categories"
- "Are there any revenue anomalies?"
- "How does revenue vary over time?"

Each question will trigger different intents and plan templates.
