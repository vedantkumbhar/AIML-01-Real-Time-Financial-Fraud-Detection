
from pathlib import Path

import pandas as pd
import networkx as nx

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data" / "archive (5)" / "fraud.csv"

OUTPUT_DIR = ROOT / "reports" / "fraud_network"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD FRAUD TRANSACTIONS
# ============================================================

print("=" * 60)
print("NEXVIA - FRAUD RING DETECTION ENGINE")
print("=" * 60)

print("\nLoading PaySim fraud records...")

fraud = pd.read_csv(
    DATA_FILE,
    usecols=[
        "step",
        "type",
        "nameOrig",
        "nameDest",
        "amount",
        "isFraud"
    ]
)

fraud = fraud[fraud["isFraud"] == 1].copy()

print("Known fraudulent transactions:", len(fraud))

# Accounts directly involved in known fraudulent transactions
fraud_accounts = set(fraud["nameOrig"]) | set(fraud["nameDest"])

print("Accounts involved in fraud:", len(fraud_accounts))

# ============================================================
# BUILD ACCOUNT NETWORK
# ============================================================

print("\nBuilding account transaction network...")

G = nx.Graph()

# Add accounts as nodes
for account in fraud_accounts:
    G.add_node(account)

# Add known fraudulent transactions as edges
for row in fraud.itertuples(index=False):
    G.add_edge(
        row.nameOrig,
        row.nameDest,
        amount=float(row.amount),
        transaction_type=row.type,
        step=int(row.step),
        is_fraud=True
    )

# ============================================================
# ADD RELATED TRANSACTION CONNECTIONS
# ============================================================

print("Finding additional transactions involving fraud-linked accounts...")

related_count = 0

for chunk in pd.read_csv(
    DATA_FILE,
    usecols=[
        "step",
        "type",
        "nameOrig",
        "nameDest",
        "amount",
        "isFraud"
    ],
    chunksize=250000
):

    related = chunk[
        (
            chunk["nameOrig"].isin(fraud_accounts)
            | chunk["nameDest"].isin(fraud_accounts)
        )
        & (chunk["isFraud"] == 0)
    ]

    for row in related.itertuples(index=False):

        G.add_edge(
            row.nameOrig,
            row.nameDest,
            amount=float(row.amount),
            transaction_type=row.type,
            step=int(row.step),
            is_fraud=False
        )

        related_count += 1

print("Related legitimate transactions:", related_count)

# ============================================================
# NETWORK ANALYSIS
# ============================================================

print("\nAnalyzing connected account groups...")

components = list(nx.connected_components(G))

account_fraud_degree = {}

for account in G.nodes():

    fraud_degree = sum(
        1
        for neighbor in G.neighbors(account)
        if G[account][neighbor].get("is_fraud", False)
    )

    account_fraud_degree[account] = fraud_degree

# ============================================================
# IDENTIFY SUSPICIOUS GROUPS
# ============================================================

rings = []

for index, component in enumerate(components, start=1):

    subgraph = G.subgraph(component)

    fraud_edges = [
        (u, v)
        for u, v, data in subgraph.edges(data=True)
        if data.get("is_fraud", False)
    ]

    if not fraud_edges:
        continue

    rings.append({
        "ring_id": index,
        "accounts": len(component),
        "transactions": subgraph.number_of_edges(),
        "fraud_transactions": len(fraud_edges),
        "risk_level": (
            "HIGH"
            if len(fraud_edges) >= 2
            else "REVIEW"
        ),
        "account_list": ", ".join(sorted(component))
    })

# ============================================================
# ACCOUNT RISK REPORT
# ============================================================

account_rows = []

for account in G.nodes():

    account_rows.append({
        "account_id": account,
        "total_connections": G.degree(account),
        "fraud_connections": account_fraud_degree[account],
        "risk_level": (
            "HIGH"
            if account_fraud_degree[account] >= 2
            else "REVIEW"
        )
    })

account_df = pd.DataFrame(account_rows)

account_df = account_df.sort_values(
    ["fraud_connections", "total_connections"],
    ascending=False
)

ring_df = pd.DataFrame(rings)

# ============================================================
# SAVE RESULTS
# ============================================================

account_file = OUTPUT_DIR / "suspicious_accounts.csv"
ring_file = OUTPUT_DIR / "fraud_rings.csv"
graph_file = OUTPUT_DIR / "account_network.graphml"

account_df.to_csv(account_file, index=False)
ring_df.to_csv(ring_file, index=False)

nx.write_graphml(G, graph_file)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FRAUD NETWORK ANALYSIS SUMMARY")
print("=" * 60)

print("Network accounts:", G.number_of_nodes())
print("Network connections:", G.number_of_edges())
print("Connected groups:", len(components))
print("Groups containing fraud:", len(rings))

print("\nTop 10 suspicious accounts:")

print(account_df.head(10).to_string(index=False))

print("\nDetected fraud-linked groups:")

if not ring_df.empty:
    print(ring_df.head(10).to_string(index=False))
else:
    print("No connected fraud groups found.")

print("\nReports saved:")
print(account_file)
print(ring_file)
print(graph_file)

print("\nFRAUD NETWORK ANALYSIS COMPLETED")
