import nltk
from nltk.corpus import wordnet as wn
import random

# nltk.download('wordnet')
# nltk.download('omw-1.4')

def get_synset(word, pos='n'):
    try:
        return wn.synset(f"{word}.{pos}.01")
    except:
        synsets = wn.synsets(word)
        if synsets:
            return synsets[0]
        else:
            return None

def calculate_similarity(word1, word2, pos='n'):
    syn1 = get_synset(word1, pos)
    syn2 = get_synset(word2, pos)
    if syn1 and syn2:
        similarity = syn1.wup_similarity(syn2)
        return similarity if similarity is not None else 0
    else:
        return 0

def get_feedback(score):
    if score > 0.9:
        return "Amazing!"
    elif score > 0.7:
        return "Very close!"
    elif score > 0.4:
        return "Somewhat related!"
    elif score > 0.1:
        return "Weakly related!"
    else:
        return "Not related at all!"

def get_relations(word, pos='n'):
    syn = get_synset(word, pos)
    if not syn:
        return {}

    return {
        "definition": syn.definition(),
        "examples": syn.examples(),
        "hypernyms": [h.name().split('.')[0] for h in syn.hypernyms()],
        "hyponyms": [h.name().split('.')[0] for h in syn.hyponyms()],
        "meronyms": [m.name().split('.')[0] for m in syn.part_meronyms()],
        "holonyms": [h.name().split('.')[0] for h in syn.part_holonyms()],
        "similar_tos": [s.name().split('.')[0] for s in syn.similar_tos()]
    }

def get_score(word1, word2, pos='n'):
    sim = calculate_similarity(word1, word2, pos)
    score = round(sim * 100, 2) if sim else 0
    feedback = get_feedback(sim)
    return score, feedback

def get_random_word():
    synset = random.choice(list(wn.all_synsets()))
    name = synset.name().split(".")[0]
    pos = synset.pos()
    definition = synset.definition()
    return name, pos, definition
