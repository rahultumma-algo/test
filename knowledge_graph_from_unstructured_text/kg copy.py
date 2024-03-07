from graphviz import Digraph
import spacy
import os
import pandas as pd
from IPython.display import Image

# Specify the Graphviz executable path
os.environ["PATH"] += os.pathsep + 'C:\\softwares\\Graphviz\\bin'

# Load the CSV file

sloka = "Ascetic Valmiki enquired of Narada, preeminent among the sages ever engaged in the practice of religious austerities or study of the Vedas and best among the eloquent."

# Load the Sanskrit language model for spaCy
nlp = spacy.load("en_core_web_sm")  # Replace "en_core_web_sm" with the appropriate Sanskrit model

# Process the sloka text with spaCy
doc = nlp(sloka)

# Function to extract concept (class), relationship, function, axiom, and instance
def extract_entities(doc):
    concepts = set()
    relationships = set()
    functions = set()
    axioms = set()
    instances = set()

    for token in doc:
        if token.dep_ == 'nsubj' or token.dep_ == 'dobj':
            concepts.add(token.text)
        elif token.dep_ == 'attr' or token.dep_ == 'ROOT':
            axioms.add(token.text)
        elif token.pos_ == 'VERB':
            functions.add(token.text)
        elif token.pos_ == 'ADP':
            relationships.add(token.text)
        elif token.pos_ == 'PROPN' or token.pos_ == 'NOUN':
            instances.add(token.text)

    # Identifying special cases and adjusting the entity sets
    if "Vedas" in instances:
        instances.remove("Vedas")
        concepts.add("Vedas")
    if "Narada" in instances:
        instances.remove("Narada")
        axioms.add("Narada")

    return concepts, relationships, functions, axioms, instances

# Extract concepts (classes), relationships, functions, axioms, and instances
concepts, relationships, functions, axioms, instances = extract_entities(doc)

# Print the extracted entities
print("Concepts (Classes):", concepts)
print("Relationships:", relationships)
print("Functions:", functions)
print("Axioms:", axioms)
print("Instances:", instances)

# Ascetic Valmiki enquired of Narada, preeminent among the sages ever engaged in the practice of religious austerities or study of the Vedas and best among the eloquent.
