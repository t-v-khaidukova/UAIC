import nltk
import spacy
from nltk import CFG
from math import sqrt
from spacy import displacy
from nltk.tree import Tree
from spacy.cli import download
from nltk.parse import ChartParser
from nltk.grammar import PCFG, Nonterminal, ProbabilisticProduction
from nltk import Nonterminal, induce_pcfg, PCFG, ProbabilisticProduction
download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

'''
pip install -U pip setuptools wheel
pip install -U numpy
pip install -U spacy thinc
'''

# 1. Using ChartParser to extract phrase structure trees

sentences = ["Flying planes can be dangerous.",
             "The parents of the bride and the groom were flying.",
             "The groom loves dangerous planes more than the bride."]

grammar = CFG.fromstring("""
S -> NP VP

# Noun Phrases
NP -> Det N
NP -> Det Adj N
NP -> Adj N
NP -> Gerund
NP -> Gerund NP
NP -> NP PP
NP -> NP CONJ_NP
NP -> NP COMP

CONJ_NP -> Conj NP

# Verb Phrases
VP -> Modal VP
VP -> Cop Adj
VP -> Cop VBG
VP -> V NP
VP -> VP COMP

# Complement and Comparative
COMP -> Adv PP
PP -> Prep NP

# Lexicon
Det -> 'the'
N -> 'parents' | 'bride' | 'groom' | 'planes'
V -> 'loves'
Adj -> 'dangerous' | 'flying'
Gerund -> 'flying'
Modal -> 'can'
Cop -> 'be' | 'were'
VBG -> 'flying'
Adv -> 'more'
Prep -> 'than' | 'of'
Conj -> 'and'
""")

parser = ChartParser(grammar)

sentences = [
    "Flying planes can be dangerous".split(),
    "The parents of the bride and the groom were flying".split(),
    "The groom loves dangerous planes more than the bride".split()
]

for sent in sentences:
    sent = [w.lower() for w in sent]
    print("\nSentence:", " ".join(sent))
    for tree in parser.parse(sent):
        print(tree)
        tree.pretty_print()

# 2. Implementing a dependency parser using spaCY and parsing the three sentences

sentences = [
    "Flying planes can be dangerous.",
    "The parents of the bride and the groom were flying.",
    "The groom loves dangerous planes more than the bride."
]

for s in sentences:
    doc = nlp(s)  # the full string, not a list of words
    print("\nSentence:", s)
    for token in doc:
        print(f"{token.text:10} -> {token.head.text:10} ({token.dep_})")

# 3. Program which transforms a Context Free Grammar
# into a Chomsky Normal Form grammar, also converting the probabilities

# Build the PCFG from the sentences
tree_strs = [
    # S1
    "(S (NP (Gerund (VBG flying) (NP (N planes)))) (VP (Modal can) (VP (Cop be) (Adj dangerous))))",
    "(S (NP (Adj flying) (N planes)) (VP (Modal can) (VP (Cop be) (Adj dangerous))))",
    # S2
    "(S (NP (NP (Det the) (N parents)) (PP (Prep of) (NP (NP (Det the) (N bride)) (CONJ_NP (Conj and) (NP (Det the) (N groom)))))) (VP (Cop were) (VBG flying)))",
    "(S (NP (NP (NP (Det the) (N parents)) (PP (Prep of) (NP (Det the) (N bride)))) (CONJ_NP (Conj and) (NP (Det the) (N groom)))) (VP (Cop were) (VBG flying)))",
    # S3
    "(S (NP (Det the) (N groom)) (VP (VP (V loves) (NP (Adj dangerous) (N planes))) (COMP (Adv more) (PP (Prep than) (NP (Det the) (N bride))))))",
    "(S (NP (Det the) (N groom)) (VP (V loves) (NP (NP (Adj dangerous) (N planes)) (COMP (Adv more) (PP (Prep than) (NP (Det the) (N bride)))))))"
]

trees = [Tree.fromstring(s) for s in tree_strs]
productions = []
for t in trees:
    productions += t.productions()

S = Nonterminal('S')
pcfg = induce_pcfg(S, productions)

print("Original PCFG")
for p in pcfg.productions():
    print(p)
print(f"\nTotal rules: {len(pcfg.productions())}\n")

def pcfg_to_cnf(pcfg):
    cnf_productions = []

    def binarize(prod):
        lhs, rhs, prob = prod.lhs(), prod.rhs(), prod.prob()
        if len(rhs) <= 2:
            return [prod]
        new_prods = []
        current_lhs = lhs
        step_prob = sqrt(prob)  # distribute prob across steps

        for i in range(len(rhs) - 2):
            new_nt = Nonterminal(f"{lhs}_{i}")
            new_prods.append(ProbabilisticProduction(current_lhs, [rhs[i], new_nt], prob=step_prob))
            current_lhs = new_nt
        new_prods.append(ProbabilisticProduction(current_lhs, list(rhs[-2:]), prob=step_prob))
        return new_prods

    for prod in pcfg.productions():
        cnf_productions.extend(binarize(prod))

    # terminal mapping
    new_prods, terminal_map, next_id = [], {}, 0
    for prod in cnf_productions:
        lhs, rhs, prob = prod.lhs(), prod.rhs(), prod.prob()
        if len(rhs) == 2:
            new_rhs = []
            for sym in rhs:
                if isinstance(sym, str):
                    if sym not in terminal_map:
                        terminal_map[sym] = Nonterminal(f"T_{next_id}")
                        next_id += 1
                        new_prods.append(ProbabilisticProduction(terminal_map[sym], [sym], prob=1.0))
                    new_rhs.append(terminal_map[sym])
                else:
                    new_rhs.append(sym)
            new_prods.append(ProbabilisticProduction(lhs, new_rhs, prob=prob))
        else:
            new_prods.append(prod)

    return PCFG(pcfg.start(), new_prods)


cnf = pcfg_to_cnf(pcfg)

print("CNF Grammar")
for p in cnf.productions():
    print(p)
print(f"\nTotal CNF rules: {len(cnf.productions())}")
