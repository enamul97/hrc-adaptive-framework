
import collections
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import networkx as nx

def print_histogram(rules, top_n):
    top_rules = rules.sort_values(by='confidence', ascending=False).head(top_n)
    antecedents = [' & '.join(list(row['antecedents'])) for _, row in top_rules.iterrows()]
    consequents = [' & '.join(list(row['consequents'])) for _, row in top_rules.iterrows()]
    confidences = top_rules['confidence'].values
    colors = plt.cm.viridis(confidences / max(confidences))
    plt.figure(figsize=(12, 7))
    bars = plt.barh(range(top_n), confidences, color=colors, edgecolor='black', linewidth=1.2, alpha=0.9,
                    height=0.5)
    left_margin = 0.01
    for bar in bars:
        plt.text(bar.get_width() + left_margin, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.2f}",
                 va='center', ha='left', fontsize=10, color='black')
    plt.yticks(range(top_n), [f"{antecedents[i]} → {consequents[i]}" for i in range(top_n)], fontsize=12)
    plt.xlabel('Confidence', fontsize=12)
    plt.title('Top Association Rules by Confidence', fontsize=16, fontweight='bold', color='navy')
    plt.grid(axis='x', linestyle='--', alpha=0.7, color='grey', linewidth=0.5)
    plt.xlim(0, 1.1)
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.show()

def display_network(rules, top_n):
    top_rules = rules.sort_values(by='confidence', ascending=False).head(top_n)
    G = nx.DiGraph()
    for _, row in top_rules.iterrows():
        antecedent = ' & '.join(list(row['antecedents']))
        consequent = ' & '.join(list(row['consequents']))
        G.add_edge(antecedent, consequent, weight=row['confidence'])
    plt.figure(figsize=(14, 10))
    pos = nx.circular_layout(G)

    edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]  # Thinner edges
    edge_colors = ["#1f77b4" if G[u][v]['weight'] > 0.8 else "#ff7f0e" for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6, edge_color=edge_colors)

    nx.draw_networkx_labels(G, pos, font_size=14, font_family="sans-serif", font_weight="bold", font_color="black")
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, font_color="darkgreen", font_family="sans-serif")
    plt.title("Circular Network Graph of Top Association Rules by Confidence", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def main():
    _data = pd.read_excel('D:\\CookingTask.xlsx')
    one_hot_data = pd.get_dummies(_data.stack()).groupby(level=0).sum()
    frequent_itemsets = apriori(one_hot_data, min_support=0.2, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
    rules = rules[(rules['antecedents'].str.len() <= 4) & (rules['consequents'].str.len() == 1)]
    ingred_dic = collections.defaultdict(list)
    print_histogram(rules,10)
    display_network(rules,10)
    data = []

    for _, r in rules.iterrows():
        antecedents = frozenset(r['antecedents'])
        consequents = frozenset(r['consequents'])
        confidence = r['confidence']
        ingred_dic[antecedents].append((confidence, consequents))

    for antecedent, values in ingred_dic.items():
        for confidence, consequent in values:
            data.append({
                'Antecedent': ', '.join(antecedent),
                'Consequent': ', '.join(consequent),
                'Confidence': confidence
            })
    df = pd.DataFrame(data)
    output_file = "ingred_rules_collective_users.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Dictionary exported to {output_file}")

if __name__ == "__main__":
    main()
