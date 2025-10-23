import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

st.title("Customer Segmentation")
st.write("This model shows customer segmentation into 5 different groups")

customer_data = pd.read_csv("Mall_Customers.csv")
st.write("### Dataset Preview")
st.dataframe(customer_data.head())
st.write("**Shape of data:**", customer_data.shape)
st.write("**Missing values:**")
st.write(customer_data.isnull().sum())

X = customer_data.iloc[:, [3, 4]].values

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
st.pyplot(plt)

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

st.write("### New Customer Input")
annual_income = st.number_input("Annual Income (k$)", min_value=0.0, step=1.0)
spending_score = st.number_input("Spending Score (1-100)", min_value=0.0, max_value=100.0, step=1.0)
add = st.button("Add Customer and Predict Cluster")

plt.figure(figsize=(7, 5))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=80, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=80, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=80, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=80, c='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=80, c='magenta', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='yellow', label='Centroids', edgecolor='black')

if add:
    new_point = np.array([[annual_income, spending_score]])
    cluster_label = kmeans.predict(new_point)[0]
    plt.scatter(new_point[0, 0], new_point[0, 1], s=220, c='black', marker='X', label='New Customer')
    st.write("New customer assigned to cluster:", int(cluster_label) + 1)

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
st.pyplot(plt)

st.write("### Cluster Centers")
st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=['Annual Income (k$)', 'Spending Score (1-100)']))
