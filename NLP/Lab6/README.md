Week 6 – Text Vectorization with Pre-trained Embeddings, Summary Report

This practical assignment explores two pre-trained word embedding models:
(1) a custom 50-dimensional GloVe-like model trained on a Wikipedia/Gigaword corpus (wiki_giga_2024), and
(2) the 300-dimensional GoogleNews Word2Vec model.
The goal is to compare their ability to represent semantic similarity and relatedness between words.

# Methods

Twenty words from each model’s vocabulary were visualized in reduced dimensionality using:

- Random 3-dimensional projection (selecting 3 embedding dimensions),
- PCA to 3 components,
- t-SNE to 3 dimensions.

Additionally, cosine similarity was computed for three semantically related word pairs (king–queen, man–woman, cat–dog) and two unrelated pairs (king–banana, table–sky).
Finally, hierarchical clustering using Ward linkage was performed on 25 words to inspect how each model groups vocabulary items.

# Results

Cosine Similarity:
The wiki_giga_2024 model produced high similarity for related pairs (up to 0.94) but also moderately high scores for unrelated pairs, indicating that its vector space is more compact and less discriminative.
The GoogleNews Word2Vec model showed slightly lower similarity for related pairs, but extremely low similarity for unrelated words (near zero), meaning it separates unrelated concepts more effectively.

Visualisation:
PCA and t-SNE showed that the custom GloVe model forms tighter, more compact clusters, even for random words, while Word2Vec produced more diffuse spatial structures, likely due to its high dimensionality and its many named entities.

Clustering:
The dendrograms reveal that the Custom GloVe model produces sharper, semantically coherent clusters, separating conceptual categories cleanly (fruits, vehicles, royalty, people, animals).
In contrast, Word2Vec yields broader and more intermixed clusters, with unrelated categories merging earlier, reflecting a more diffuse embedding geometry — likely a result of its higher dimensionality and the noisier distribution caused by many named entities in the GoogleNews corpus.

# Conclusion

GloVe provides cleaner, more interpretable clusters and more stable semantic similarity.
Word2Vec shows more diffuse relationships and weaker separation between categories.
Thus, GloVe is better for grouping similar items, while Word2Vec may be less consistent due to corpus noise.
