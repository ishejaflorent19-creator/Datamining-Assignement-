"""

AUCA 
Data mining 
Assigniment 1
ISheja Munyakazi Florent Assignemnt
Regn:101485 

Association Rule Learning on the Groceries Dataset
Apriori algorithm via mlxtend
"""
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------
# 1. Load data — the CSV is a "basket" file: one transaction per
#    row, items comma-separated, no header, ragged row lengths.
# ---------------------------------------------------------------
DATA_PATH = "data/groceries.csv"

transactions = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        items = [i.strip() for i in line.strip().split(",") if i.strip() != ""]
        if items:
            transactions.append(items)

n_transactions = len(transactions)
all_items = sorted(set(item for t in transactions for item in t))
n_items = len(all_items)

print(f"Number of transactions: {n_transactions}")
print(f"Number of distinct items: {n_items}")

# ---------------------------------------------------------------
# 2. Top 10 most frequent items
# ---------------------------------------------------------------
from collections import Counter
item_counts = Counter(item for t in transactions for item in t)
top10_items = pd.DataFrame(item_counts.most_common(10), columns=["item", "count"])
top10_items["support"] = top10_items["count"] / n_transactions
print("\nTop 10 items:\n", top10_items)

plt.figure(figsize=(8, 5))
plt.barh(top10_items["item"][::-1], top10_items["support"][::-1], color="#4C72B0")
plt.xlabel("Support (relative frequency)")
plt.title("Top 10 Most Frequent Items")
plt.tight_layout()
plt.savefig("figures/top10_items.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 3. One-hot encode transactions
# ---------------------------------------------------------------
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# ---------------------------------------------------------------
# 4. Frequent itemsets (Apriori), min_support chosen so we get a
#    healthy number of itemsets of size >= 2
# ---------------------------------------------------------------
min_support = 0.01
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)

itemsets_2plus = frequent_itemsets[frequent_itemsets["length"] >= 2].sort_values(
    "support", ascending=False
)
print(f"\nFrequent itemsets (size>=2) at min_support={min_support}: {len(itemsets_2plus)}")
print(itemsets_2plus.head(10))

top10_itemsets = itemsets_2plus.head(10).copy()
top10_itemsets["itemsets"] = top10_itemsets["itemsets"].apply(lambda s: ", ".join(sorted(s)))
top10_itemsets[["itemsets", "support"]].to_csv("results/top10_frequent_itemsets.csv", index=False)

# ---------------------------------------------------------------
# 5. Association rules
# ---------------------------------------------------------------
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

# keep readable columns
rules_readable = rules.copy()
rules_readable["antecedents"] = rules_readable["antecedents"].apply(lambda s: ", ".join(sorted(s)))
rules_readable["consequents"] = rules_readable["consequents"].apply(lambda s: ", ".join(sorted(s)))
cols = ["antecedents", "consequents", "support", "confidence", "lift"]
rules_readable = rules_readable[cols]

print(f"\nTotal rules generated (confidence >= 0.2): {len(rules)}")
top10_rules = rules_readable.head(10)
print(top10_rules)
top10_rules.to_csv("results/top10_rules_by_lift.csv", index=False)

# Rules sorted by confidence and support as well, for Q5
by_conf = rules_readable.sort_values("confidence", ascending=False).iloc[0]
by_lift = rules_readable.sort_values("lift", ascending=False).iloc[0]
by_supp = rules_readable.sort_values("support", ascending=False).iloc[0]

print("\nHighest confidence rule:\n", by_conf)
print("\nHighest lift rule:\n", by_lift)
print("\nHighest support rule:\n", by_supp)

with open("results/top_rules_summary.txt", "w") as f:
    f.write("HIGHEST CONFIDENCE RULE\n")
    f.write(str(by_conf) + "\n\n")
    f.write("HIGHEST LIFT RULE\n")
    f.write(str(by_lift) + "\n\n")
    f.write("HIGHEST SUPPORT RULE\n")
    f.write(str(by_supp) + "\n")

# ---------------------------------------------------------------
# 6. Scatter plot: support vs confidence, colored by lift
# ---------------------------------------------------------------
plt.figure(figsize=(7, 6))
sc = plt.scatter(rules["support"], rules["confidence"], c=rules["lift"], cmap="viridis", alpha=0.7)
plt.colorbar(sc, label="Lift")
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Association Rules: Support vs Confidence (colored by Lift)")
plt.tight_layout()
plt.savefig("figures/rules_scatter.png", dpi=150)
plt.close()

# Bar chart of top 10 rules by lift
plt.figure(figsize=(9, 5))
labels = [f"{a} -> {c}" for a, c in zip(top10_rules["antecedents"], top10_rules["consequents"])]
plt.barh(labels[::-1], top10_rules["lift"][::-1], color="#55A868")
plt.xlabel("Lift")
plt.title("Top 10 Association Rules by Lift")
plt.tight_layout()
plt.savefig("figures/top10_rules_lift.png", dpi=150)
plt.close()

frequent_itemsets.to_csv("results/all_frequent_itemsets.csv", index=False)
rules_readable.to_csv("results/all_rules.csv", index=False)

print("\nDone. Outputs written to current directory and figures/ subfolder.")
