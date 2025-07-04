import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
data = {
    'user_id': ['U1', 'U1', 'U1', 'U2', 'U2', 'U3', 'U3', 'U4', 'U4', 'U5'],
    'product': ['Laptop', 'Headphones', 'Mouse', 'Laptop', 'Mouse',
                'Laptop', 'Keyboard', 'Mouse', 'Keyboard', 'Headphones'],
    'rating': [5, 3, 4, 5, 4, 4, 5, 3, 4, 2]
}

df = pd.DataFrame(data)
print("Sample Ratings Data:\n", df)

user_item_matrix = df.pivot_table(index='user_id', columns='product', values='rating')
user_item_matrix = user_item_matrix.fillna(0)
print("\nUser-Item Matrix:\n", user_item_matrix)
plt.figure(figsize=(8, 5))
sns.heatmap(user_item_matrix, annot=True, cmap='Blues')
plt.title("User-Item Rating Matrix")
plt.show()
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
plt.figure(figsize=(6, 4))
sns.heatmap(user_similarity_df, annot=True, cmap='YlGnBu')
plt.title("User Similarity Matrix")
plt.show()
def get_user_recommendations(user_id, top_n=2):
    if user_id not in user_item_matrix.index:
        print(f"No data available for user: {user_id}")
        return []

    print(f"\nGenerating recommendations for {user_id}...")

    sim_scores = user_similarity_df[user_id]
    sim_scores = sim_scores.drop(user_id)  # remove self
    sim_users = sim_scores.sort_values(ascending=False)

    weighted_ratings = pd.Series(dtype=np.float64)

    for other_user, sim_score in sim_users.items():
        ratings = user_item_matrix.loc[other_user]
        weighted_ratings = weighted_ratings.add(ratings * sim_score, fill_value=0)

    user_rated_items = user_item_matrix.loc[user_id]
    unrated_items = user_rated_items[user_rated_items == 0].index

    recommendations = weighted_ratings[unrated_items].sort_values(ascending=False).head(top_n)
    return recommendations
recommended_items = get_user_recommendations('U5', top_n=2)

if not recommended_items.empty:
    print("\nRecommended Products for U5:")
    print(recommended_items)

    recommended_items.plot(kind='barh', color='skyblue')
    plt.title("Top Product Recommendations for U5")
    plt.xlabel("Predicted Rating Score")
    plt.ylabel("Product")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
else:
    print("No recommendations available.")