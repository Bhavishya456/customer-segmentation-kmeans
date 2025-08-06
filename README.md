# customer-segmentation-kmeans
Segment mall customers using K-Means clustering based on annual income and spending score to identify distinct customer groups for targeted marketing strategies.
# 🛍 Customer Segmentation with K‑Means

## Overview
This project applies *K‑Means clustering* to group mall customers into segments based on their *Annual Income* and *Spending Score*.  
The goal is to help businesses understand their customer base and target each segment with personalized marketing strategies.

## Dataset
*Source:* Mall Customers Dataset (Kaggle)  
*Fields Used:*
- Annual Income (k$) – yearly income in thousands of dollars
- Spending Score (1-100) – customer spending behavior score

The dataset is placed in the data/ folder.

## Methodology
1. *Data Preprocessing*
   - Extract relevant features
   - Standardize features for better clustering performance
2. *Finding the Optimal Number of Clusters*
   - Elbow Method
   - Silhouette Score
3. *Model Training*
   - Apply K‑Means clustering
4. *Visualization*
   - Scatter plot of clusters with centroids
5. *Output*
   - Segmented dataset saved to outputs/

## Results
Using *k = 5*, customers are grouped into 5 distinct clusters, for example:
1. High income, high spending (VIPs)
2. Low income, high spending (loyal budget shoppers)
3. Low income, low spending (occasional buyers)
4. High income, low spending (untapped potential)
5. Middle income, middle spending (average customers)

The final file Mall_Customers_segmented.csv contains the cluster label for each customer.

## Project Structure
