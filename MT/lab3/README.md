https://anythingtranslate.com/translators/antonyms-translator/

<img width="1226" height="716" alt="image" src="https://github.com/user-attachments/assets/1893bf08-9f0b-4bfb-b18c-c58dca499f7d" />
PROMPT:
Original Language Name (Default is Normal Language):
Normal Language
Translated Language Name:
Antonyms
Translator Title (Optional):
Antonyms Translator
Extra Info (Optional):
Please convert all words that have antonyms to their antonyms

Annotation for Lab3:

This implementation recreates the idea of “translation drift” by sequentially translating a sentence through 
a chain of randomly selected languages before returning to the original language. Repeated machine translation 
introduces semantic noise and gradual information deformation. Each translation step slightly shifts phrasing 
due to differences in grammar systems, lexical resources, and statistical tendencies of the translation model. 
When applied iteratively across distant language families, these shifts accumulate and produce significant 
divergence from the input.

To quantify this distortion, two metrics were applied:
- Levenshtein-based textual similarity — measures character-level edit distance. This reflects how
  formally different the output becomes.
- Sentence-BERT cosine similarity — captures deeper semantic drift by comparing sentence embeddings.
  This metric shows whether meaning is preserved even when phrasing changes.

The contrast between these curves helps illustrate that linguistic form and meaning do not degrade at the same 
rate.

Additionally, the random selection of 15 languages introduces variability that mimics real-world multilingual 
processing pipelines, where translation passes through heterogeneous systems. This randomness also enables 
experimenting with which language paths create maximal semantic distortion — for example, transitions between 
typologically distant languages (such as Finnish, Chinese, Zulu - different language families) often cause more 
disruption than transitions within closely related languages (Romanian languages).
