from graphviz import Digraph
import spacy
import os
from IPython.display import Image

# Specify the Graphviz executable path
os.environ["PATH"] += os.pathsep + 'C:\\softwares\\Graphviz\\bin'

# Load the Sanskrit language model for spaCy
nlp = spacy.load("en_core_web_sm")  # Replace "your_sanskrit_model" with the appropriate Sanskrit model

# Sloka text
sloka = "Ascetic Valmiki enquired of Narada, preeminent among the sages ever engaged in the practice of religious austerities or study of the Vedas and best among the eloquent."

# Process the sloka text with spaCy
doc = nlp(sloka)

# Create a directed graph
graph = Digraph()

# Add nodes and edges to the graph based on dependency parsing information
for token in doc:
    graph.node(str(token.i), label=f"{token.text} - {token.pos_}")
    if token.dep_ != 'ROOT':
        graph.edge(str(token.head.i), str(token.i), label=token.dep_)

# Render and display the graph
graph.render('knowledge_graph', format='png', view=True)
Image(filename='knowledge_graph.png')
