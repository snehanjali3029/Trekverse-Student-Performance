import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Load dataset
df = pd.read_csv("data/student_mat_cleaned.csv")


# Select engagement-related features
engagement_features = [
    "studytime",
    "freetime",
    "goout",
    "absences",
    "activities"
]

cluster_data = df[engagement_features].copy()


# Convert activities from yes/no to 1/0
cluster_data["activities"] = cluster_data["activities"].map({
    "yes": 1,
    "no": 0
})


# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_data)


# Create 2 clusters
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_scaled)


# Add cluster labels to original dataset
# Add cluster labels to original dataset
df["Engagement_Cluster"] = cluster_labels

# Convert activities to numeric for cluster analysis
df["activities"] = df["activities"].map({
    "yes": 1,
    "no": 0
})


print("----- CLUSTER COUNTS -----")
print(df["Engagement_Cluster"].value_counts().sort_index())


# Display average feature values for each cluster
print("\n----- ENGAGEMENT PROFILE BY CLUSTER -----")

cluster_profile = df.groupby("Engagement_Cluster")[
    engagement_features
].mean()

print(cluster_profile)


# Save cluster results
df.to_csv(
    "reports/student_clusters.csv",
    index=False
)

print("\nClustered dataset saved to:")
print("reports/student_clusters.csv")


# Create visualization using studytime and absences
plt.figure(figsize=(8, 6))

plt.scatter(
    df["studytime"],
    df["absences"],
    c=df["Engagement_Cluster"],
    alpha=0.7
)

plt.xlabel("Study Time")
plt.ylabel("Absences")
plt.title("Student Engagement Clusters")

plt.tight_layout()

plt.savefig(
    "reports/engagement_clusters.png",
    dpi=300
)

plt.close()

print("Cluster visualization saved to:")
print("reports/engagement_clusters.png")