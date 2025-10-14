# pip install googletrans==4.0.0-rc1, matplotlib, sentence-transformers

from googletrans import Translator, LANGUAGES
import Levenshtein
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import random

# Textual similarity using Levenshtein
def textual_similarity(a, b):
    distance = Levenshtein.distance(a, b)
    similarity = 1 - distance / max(len(a), len(b))
    return similarity

# Initialize Sentence-BERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(a, b):
    emb1 = sbert_model.encode(a)
    emb2 = sbert_model.encode(b)
    return util.cos_sim(emb1, emb2).item()

# Function with random languages
def weird_translate(text, num_steps=15):
    translator = Translator()
    original_text = text
    translations = [original_text]

    # Build random language sequence
    all_langs = list(LANGUAGES.keys())
    all_langs.remove('en')  # English for final translation
    random_langs = random.sample(all_langs, min(num_steps, len(all_langs)))
    random_langs.append('en')  # back to English

    print("Random translation sequence:", random_langs)

    current_text = original_text
    for i, lang in enumerate(random_langs):
        translated = translator.translate(current_text, dest=lang)
        current_text = translated.text
        translations.append(current_text)
        print(f"Step {i+1} ({LANGUAGES[lang]}): {current_text}")

    return translations

text = input("Enter text: ")
translations = weird_translate(text)

# Compute similarities
text_sims = [textual_similarity(text, t) for t in translations]
semantic_sims = [semantic_similarity(text, t) for t in translations]

print(f"Textual similarity after final translation: {text_sims[-1]:.4f}")
print(f"Semantic similarity after final translation: {semantic_sims[-1]:.4f}")

# Plot similarities
steps = list(range(len(translations)))
plt.figure(figsize=(12, 6))
plt.plot(steps, text_sims, marker='o', label="Textual similarity (Levenshtein)")
plt.plot(steps, semantic_sims, marker='s', label="Semantic similarity (SBERT)")
plt.xticks(steps)
plt.xlabel("Translation Step")
plt.ylabel("Similarity to Original")
plt.title("Weird Translator: Textual and Semantic Change Over Steps")
plt.legend()
plt.grid(True)
plt.show()
