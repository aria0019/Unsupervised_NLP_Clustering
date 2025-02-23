# Unsupervised_NLP_Clustering

## **Project Overview**
This project applies **unsupervised learning** techniques to cluster text data. We explore:
- **K-Means (KNN) Clustering**
- **DBSCAN Clustering**
- **Hierarchical Clustering**

The goal is to discover meaningful groups in **text data** using **TF-IDF & Sentence Embeddings (BERT-based)**. Although the dataset has labels we will not use them during training, only for validation purposes. The aim is to use unsupervised methods of NLP classification. 

---

## **Dataset Information**
The dataset consists of **text documents** categorized into different topics. We preprocess the text to remove noise and extract meaningful words.

### **Data Source**
The dataset comes from a public news dataset, called **20 Newsgroups**. It contains approximately 20000 newsgroup documents, partitioned across 20 different categories.

---

## **Clustering Methods & Results**

### **1️⃣ K-Means (KNN) Clustering**
- Used **TF-IDF vectorization** for feature extraction.  
- Found **7 optimal clusters** using the **Elbow & Silhouette Method**.  
- **Issue**: K-Means assumes clusters are spherical, but **text data is high-dimensional and non-spherical**.  
- **Result**: Heatmap shows mixed clusters that **do not align** with the true topics. Some of the clusters seem to group similar categories together but others are completely random. This indicates that TF-IDFand K-Means is not the best approach for textual data.

### **2️⃣ DBSCAN Clustering**
- Used **Sentence Embeddings (MiniLM-L6-v2, a BERT-based model)** for better text representation.
- Estimated **epsilon** using K-Nearest Neighbors.
- **Issue**: DBSCAN struggles with high-dimensional embeddings, leading to difficulty in identifying dense regions of data.
- **Result**: Most data was either classified as noise or into a single large cluster (poor separation), making DBSCAN ineffective for this task.

### **3️⃣ Hierarchical Clustering**
- Used Agglomerative Clustering with cosine similarity to capture text relationships better.
- Constructed **dendrogram** to analyze clusters.
- Computationally expensive for large datasets but works well for moderate-sized corpora.
- **Result**: Performed better than K-Means & DBSCAN, providing a more meaningful cluster structure.

---

## **How to Run the Notebook**
1. Install required packages:
   ```bash
   pip install nltk scikit-learn pandas numpy matplotlib seaborn sentence-transformers
   ```
2. Run the **Jupyter Notebook** to:
   - Preprocess text
   - Apply different clustering methods
   - Analyze results using visualizations

---

## **Key Findings & Future Work**
- **K-Means is not ideal** for non-spherical text clusters.
- **DBSCAN does not work well** with high-dimensional text embeddings.
- **Hierarchical Clustering is promising** but requires careful tuning.
- Future improvements:
  - Try **BERTopic** for **better text-based clustering**.

 **Conclusion**: Hierarchical Clustering provided the best structure, with good clustering among the topics. As future work, alternative methods like BERTopic should be explored.

---


