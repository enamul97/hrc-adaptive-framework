import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import pandas as pd


def analyze_user_rules(data, user_id=1, min_support=0.1, min_confidence=0.3, top_n=20):

    df = pd.DataFrame(data)
    user_data = df[df['User'] == user_id]['Ingredients'].tolist()
    all_ingredients = sorted(list(set([ing for sublist in user_data for ing in sublist])))

    encoded_vals = []
    for ingredients in user_data:
        row_dict = {ingredient: (1 if ingredient in ingredients else 0)
                    for ingredient in all_ingredients}
        encoded_vals.append(row_dict)

    ohe_df = pd.DataFrame(encoded_vals).astype(bool)
    frequent_itemsets = apriori(ohe_df, min_support=min_support, use_colnames=True)

    if len(frequent_itemsets) == 0:
        print(f"No frequent itemsets found for User {user_id}.")
        return None

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    if len(rules) == 0:
        print(f"No rules found for User {user_id}.")
        return None

    rules['antecedent_len'] = rules['antecedents'].apply(len)
    rules['consequent_len'] = rules['consequents'].apply(len)

    filtered_rules = rules[(rules['antecedent_len'] == 1) &
                           (rules['consequent_len'] == 1)].copy()

    filtered_rules = filtered_rules.nlargest(top_n, 'confidence')
    filtered_rules['antecedents_str'] = filtered_rules['antecedents'].apply(lambda x: ' + '.join(sorted(list(x))))
    filtered_rules['consequents_str'] = filtered_rules['consequents'].apply(lambda x: next(iter(x)))
    filtered_rules['rule_str'] = filtered_rules.apply(lambda x: f"{x['antecedents_str']} → {x['consequents_str']}",
                                                      axis=1)
    filtered_rules = filtered_rules.sort_values('confidence', ascending=True)
    plt.figure(figsize=(14, 6))
    plt.gca().set_facecolor('#f5f5f5')
    plt.gcf().set_facecolor('#f5f5f5')

    y_positions = np.arange(len(filtered_rules)) * 1.2
    bar_height = 0.6
    colors = plt.cm.Blues(np.linspace(0.6, 0.9, len(filtered_rules)))
    bars = plt.barh(y_positions, filtered_rules['confidence'], height=bar_height,
                    color=colors,
                    alpha=0.8,
                    edgecolor='none')

    plt.yticks(y_positions, filtered_rules['rule_str'], fontsize=10)
    plt.xlabel('Confidence', fontsize=12, fontweight='bold', color='#2f2f2f')
    plt.grid(axis='x', linestyle='--', alpha=0.2, color='#2f2f2f')
    plt.title(f'Association Rules for User {user_id}',
              fontsize=14, fontweight='bold', pad=20, color='#2f2f2f')
    plt.margins(y=0.01)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, y_positions[i],
                 f'  {width:.2f}',
                 ha='left', va='center',
                 fontsize=9,
                 fontweight='bold',
                 color='#2f2f2f',
                 bbox=dict(facecolor='white',
                           alpha=0.8,
                           edgecolor='none',
                           pad=1,
                           boxstyle='round,pad=0.5'))

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_color('#2f2f2f')
    plt.gca().spines['bottom'].set_color('#2f2f2f')
    plt.tight_layout()

    return plt


def excel_to_dict(excel_file_path):
    df = pd.read_excel(excel_file_path)
    data = {
        'User': [],
        'Ingredients': []
    }

    for _, row in df.iterrows():
        user_id = int(row['User'])
        ingredients = [
            row['Ingredient 1'],
            row['Ingredient 2'],
            row['Ingredient 3'],
            row['Ingredient 4'],
            row['Ingredient 5']
        ]
        data['User'].append(user_id)
        data['Ingredients'].append(ingredients)

    return data


def main(user_id, top_n):
    excel_file_path = "cooking.xlsx"
    data = excel_to_dict(excel_file_path)
    min_support, min_confidence = 0.5, 0.3
    plt_obj = analyze_user_rules(data, user_id, min_support, min_confidence, top_n)
    if plt_obj is not None:
        plt_obj.show()


if __name__ == "__main__":
    main(6, 30)
