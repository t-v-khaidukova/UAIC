This project implements a rule-based word-by-word machine translation system from English to French. 
It uses a predefined lexicon (lexicon.txt) that maps English words to their French equivalents and includes 
part-of-speech (POS) tags and grammatical gender information. A separate file (rules.txt) defines rewriting 
rules that adjust word order, enforce gender agreement, and handle structural transformations.

The translation pipeline consists of three main stages: tokenization, rule application, and lexicon-based 
translation. First, the input sentence is tokenized into words and punctuation. Each token is annotated with 
its POS and gender based on the lexicon. The system then applies rewriting rules using these features. 
For example, an ADJ + N pattern is reordered as N + ADJ to match French adjective-noun order, and determiner 
rules adjust “the” or “a” to match the gender of the following noun (e.g., Le/La or Un/Une). All rules are 
applied dynamically from the rules.txt file, making the system data-driven.

After structural adjustments, each token is translated using the lexicon. Unknown words are preserved.

The system is interactive. The user enters an English sentence, and the system returns the French version.
