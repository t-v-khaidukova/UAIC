import re

# Load lexicon with normalized POS and gender
def load_lexicon(filename="lexicon.txt"):
    lexicon = {}
    gender_map = {}  # noun -> gender
    current_pos = None
    current_gender = None

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Detect section headers
            if ":" in line:
                header = line.replace(":", "").strip()
                if "Masc N" in header:
                    current_pos = "N"
                    current_gender = "M"
                elif "Fem N" in header:
                    current_pos = "N"
                    current_gender = "F"
                else:
                    current_pos = header.split()[0]  # V, DET, ADJ, PREP
                    current_gender = None
                continue

            # Parse entry, English to French
            if "->" in line:
                eng, fr = [x.strip() for x in line.split("->")]
            else:
                eng = fr = line.strip()

            key = eng.lower()
            if key not in lexicon:
                lexicon[key] = []
            lexicon[key].append({
                "translation": fr,
                "pos": current_pos,
                "gender": current_gender
            })

            # Track gender for nouns
            if current_gender in ("M", "F"):
                gender_map[key] = current_gender

    return lexicon, gender_map

# Load rules dynamically
def load_rules(filename="rules.txt"):
    rules = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "->" not in line:
                continue
            left, right = [x.strip() for x in line.split("->")]
            rules.append((left.split(), right.split()))
    return rules

# Build token features from lexicon
def get_token_features(tokens, lexicon, gender_map):
    features = []
    for tok in tokens:
        low = tok.lower()
        if low in lexicon:
            entry = lexicon[low][0]
            features.append({"pos": entry["pos"], "gender": entry["gender"]})
        else:
            features.append({"pos": None, "gender": None})
    return features

# Apply POS-based rewriting rules
def apply_rewriting_rules(tokens, token_features, rules):
    i = 0
    output_tokens = []
    output_features = []

    while i < len(tokens):
        matched = False
        for left, right in rules:
            pattern_len = len([x for x in left if x != "+"])
            if i + pattern_len > len(tokens):
                continue

            # Build POS sequence for matching
            seq_match = [token_features[i+j]["pos"] for j in range(pattern_len)]
            left_pattern = [x for x in left if x != "+"]

            if seq_match == left_pattern:
                # Reorder according to rule
                new_tokens = []
                new_features = []
                for item in right:
                    if item == "+":
                        continue
                    # Determine which index in left it refers to
                    # ADJ + N -> N + ADJ
                    if item in ["ADJ", "N", "V", "DET", "PREP", "CONJ", "PNOUN"]:
                        # Match each POS in the rule to the correct original word
                        for j in range(pattern_len):
                            if token_features[i+j]["pos"] == item:
                                new_tokens.append(tokens[i+j])
                                new_features.append(token_features[i+j])
                                break
                    else:
                        # If item is literal word, just append
                        new_tokens.append(item)
                        new_features.append({"pos": None, "gender": None})

                output_tokens.extend(new_tokens)
                output_features.extend(new_features)
                i += pattern_len
                matched = True
                break

        if not matched:
            output_tokens.append(tokens[i])
            output_features.append(token_features[i])
            i += 1

    return output_tokens, output_features

# Translate using lexicon and gender-aware determiners
def translate(tokens, token_features, lexicon, gender_map):
    result = []
    for i, tok in enumerate(tokens):
        low = tok.lower()

        # DET
        if low in lexicon and lexicon[low][0]["pos"] == "DET":
            next_gender = None
            if i + 1 < len(tokens) and token_features[i+1]["gender"]:
                next_gender = token_features[i+1]["gender"]

            if next_gender == "F":
                chosen = "La" if low == "the" else "Une"
            else:
                chosen = "Le" if low == "the" else "Un"

            result.append(chosen)
            continue

        # General translation
        if low in lexicon:
            result.append(lexicon[low][0]["translation"])
        else:
            result.append(tok)

    return result

# Clean punctuation
def clean_punctuation(translated):
    cleaned = []
    for tok in translated:
        if re.match(r"^[.,!?;:]$", tok) and cleaned:
            cleaned[-1] += tok
        else:
            cleaned.append(tok)
    return cleaned

def main():
    lexicon, gender_map = load_lexicon()
    rules = load_rules()

    print("English-to-French word-by-word translator. Type 'exit' to quit.")

    while True:
        sentence = input("Enter a sentence: ")
        if sentence.strip().lower() == "exit":
            break

        # Tokenize sentence
        tokens = re.findall(r"\w+|[^\w\s]", sentence)
        token_features = get_token_features(tokens, lexicon, gender_map)

        # Apply rewriting rules
        tokens, token_features = apply_rewriting_rules(tokens, token_features, rules)

        # Translate
        translated = translate(tokens, token_features, lexicon, gender_map)

        # Clean punctuation
        translated = clean_punctuation(translated)

        print("Translation:", " ".join(translated))

if __name__ == "__main__":
    main()
