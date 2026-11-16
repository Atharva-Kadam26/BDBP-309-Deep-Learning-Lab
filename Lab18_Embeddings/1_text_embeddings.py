import re
import numpy as np
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from sklearn.decomposition import PCA

sentences = [
    "deep learning is a branch of machine learning",
    "machine learning uses algorithms to learn patterns",
    "neural networks are used in deep learning",
    "natural language processing works with text data",
    "word embeddings represent words as numerical vectors",
    "deep learning models learn useful representations",
    "word2vec learns relationships between words"
]

print("Original sentences:")

for sentence in sentences:
    print(sentence)

tokenized_sentences = []

for sentence in sentences:

    words = re.findall(r"\w+", sentence.lower())

    tokenized_sentences.append(words)

print("\nTokenized sentences:")

for sentence in tokenized_sentences:
    print(sentence)


text_model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,
    window=3,
    min_count=1,
    workers=4,
    sg=1,
    epochs=100
)

print("Word2Vec model trained successfully!")

word = "learning"
embedding = text_model.wv[word]
print("Word:", word)
print("Embedding shape:", embedding.shape)
print("Embedding vector:")

print(embedding)

similar_words = text_model.wv.most_similar(
    "learning",
    topn=3
)

print("Words similar to learning:")

for word, similarity in similar_words:
    print(word, similarity)

words = list(text_model.wv.index_to_key)

word_vectors = []
for word in words:
    word_vectors.append(text_model.wv[word])
word_vectors = np.array(word_vectors)

pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(word_vectors)


plt.figure(figsize=(10, 6))

for i, word in enumerate(words):

    plt.scatter(
        reduced_vectors[i, 0],
        reduced_vectors[i, 1]
    )

    plt.text(
        reduced_vectors[i, 0],
        reduced_vectors[i, 1],
        word
    )

plt.title("Word2Vec Word Embeddings")

plt.xlabel("PCA Component 1")

plt.ylabel("PCA Component 2")

plt.show()



