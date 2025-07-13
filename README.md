# Association Rule Mining on Simulated Grocery Transaction Data

## Introduction

This project demonstrates the application of association rule mining to synthetic (simulated) grocery transaction data, using the Apriori algorithm. Association rule mining is a data mining technique widely used in market basket analysis to uncover patterns and relationships among items purchased together. The aim is to illustrate how retailers can use such insights to improve product placement, promotions, and inventory management.

## Objectives

- Simulate transactional data representing university student grocery purchases.
- Apply the Apriori algorithm to identify frequent itemsets.
- Generate and interpret association rules with specified support and confidence thresholds.
- Provide clear explanations of the discovered rules in the context of real-world retail scenarios.

## 1. Simulating Transaction Data

To mimic the purchasing behavior of university students, we generated 50 transactions. Each transaction (basket) contains between 2 and 5 items selected from a pool of 8 common grocery items:

- Indomie (noodles)
- Bread
- Juice
- Eggs
- Oatmeal
- Onions
- Tomatoes
- Potatoes

To create realistic patterns, certain pairs of items (e.g., Bread & Eggs, Juice & Oatmeal) are intentionally co-occurring more frequently.

**Python code for data simulation:**
```python
import random
import pandas as pd

random.seed(42)
items_pool = ['Indomie', 'Bread', 'Juice', 'Eggs', 'Oatmeal', 'Onions', 'Tomatoes', 'Potatoes']

transactions = []
for _ in range(50):
    transaction = set(random.sample(items_pool, k=random.randint(2, 5)))
    # Increase likelihood of common pairs
    if random.random() < 0.3:
        transaction.update(['Bread', 'Eggs'])
    if random.random() < 0.2:
        transaction.update(['Juice', 'Oatmeal'])
    transactions.append(list(transaction))
```
**Rationale:**  
This approach simulates realistic student shopping behavior and ensures there are patterns for the Apriori algorithm to detect.

---

## 2. Data Preprocessing

The transactions are converted into a one-hot encoded format using the TransactionEncoder from the `mlxtend` library. This step is necessary for the Apriori algorithm, which requires data in binary matrix format.

**Code:**
```python
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df.head())
```

---

## 3. Mining Frequent Itemsets with Apriori

The Apriori algorithm is applied to the one-hot encoded data to identify frequent itemsets. The minimum support threshold is set to 0.3 (i.e., itemsets must appear in at least 30% of all transactions).

**Code:**
```python
from mlxtend.frequent_patterns import apriori

frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
print(frequent_itemsets)
```
**Rationale:**  
A 30% support threshold ensures that only common and meaningful itemsets are considered frequent.

---

## 4. Generating Association Rules

Association rules are generated from the frequent itemsets using a minimum confidence threshold of 0.7 (70%). Confidence measures how often the rule has been found to be true.

**Code:**
```python
from mlxtend.frequent_patterns import association_rules

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
```
**Rationale:**  
A high confidence threshold ensures that only strong and reliable rules are reported.

---

## 5. Results: Discovered Association Rules

The following rules were identified (example output):

| Antecedents | Consequents | Support | Confidence | Lift  |
|-------------|-------------|---------|------------|-------|
| Eggs        | Bread       | 0.50    | 0.86       | 1.31  |
| Bread       | Eggs        | 0.50    | 0.76       | 1.31  |
| Tomatoes    | Bread       | 0.30    | 0.71       | 1.08  |

### Explanation of the Rules

#### Rule 1: If a customer buys **Eggs**, they are also likely to buy **Bread** (confidence: 0.86).
- **Interpretation:** Among all baskets containing Eggs, 86% also contain Bread. This suggests a strong association between these two items. For a retailer, this insight could justify placing Bread and Eggs near each other in the store or offering bundle promotions.

#### Rule 2: If a customer buys **Bread**, they are also likely to buy **Eggs** (confidence: 0.76).
- **Interpretation:** Similarly, 76% of baskets with Bread also include Eggs, reinforcing the mutual association.

#### Rule 3: If a customer buys **Tomatoes**, they are also likely to buy **Bread** (confidence: 0.71).
- **Interpretation:** 71% of baskets with Tomatoes also have Bread, suggesting a possible pattern (for example, students preparing sandwiches or toast with tomatoes).

---

## 6. Significance and Applications

The discovered associations provide actionable insights for retailers, such as:
- Optimizing store layouts by placing frequently paired items together.
- Designing targeted promotions or discounts for items commonly bought together.
- Improving inventory management by understanding demand patterns.

---

## 7. How to Run the Analysis

### Requirements

- Python 3.7+
- pandas
- mlxtend

**Install dependencies:**
```
pip install pandas mlxtend
```

**To run the analysis:**
- Save the provided code snippets into a Python script or Jupyter notebook.
- Execute the code sequentially to simulate the data, preprocess, mine frequent itemsets, and generate association rules.

---

**Author:**  
Faith Mwangi  
[GitHub: FMuthoniMwangi](https://github.com/FMuthoniMwangi)