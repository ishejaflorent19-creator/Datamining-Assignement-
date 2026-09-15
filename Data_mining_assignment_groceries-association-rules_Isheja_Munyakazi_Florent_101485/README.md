# Groceries — Association Rule Learning

Assignment submission: association rule mining (Apriori) on the Groceries transactions dataset.

## Contents

```
.
├── README.md              this file
├── REPORT.md              full written report answering all assignment questions (Q1–Q8)
├── analysis.py             analysis code (pandas + mlxtend)
├── data/
│   └── groceries.csv       raw dataset (9,835 transactions, one basket per row)
├── figures/
│   ├── top10_items.png          bar chart of the 10 most frequent items
│   ├── top10_rules_lift.png     bar chart of the top 10 rules by lift
│   └── rules_scatter.png        support vs. confidence, colored by lift, for all rules
└── results/
    ├── all_frequent_itemsets.csv    every frequent itemset found (support ≥ 0.01)
    ├── all_rules.csv                 every association rule found (confidence ≥ 0.2)
    ├── top10_frequent_itemsets.csv   top 10 itemsets by support
    ├── top10_rules_by_lift.csv       top 10 rules by lift
    └── top_rules_summary.txt         highest-confidence / highest-lift / highest-support rule
```

## Report

👉 See [**REPORT.md**](REPORT.md) for the full write-up, with all tables and figures inline.

## Running the analysis

```bash
pip install pandas mlxtend matplotlib
python analysis.py
```

This regenerates every file under `figures/` and `results/` from `data/groceries.csv`.

## Method summary

- **Data:** 9,835 transactions, 169 distinct items, loaded from the basket-format CSV (no header, ragged rows).
- **Frequent itemsets:** mined with the Apriori algorithm (`mlxtend.frequent_patterns.apriori`), minimum support = 0.01.
- **Association rules:** generated with `mlxtend.frequent_patterns.association_rules`, minimum confidence = 0.2, ranked by lift.
