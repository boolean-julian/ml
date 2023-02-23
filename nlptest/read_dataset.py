import pandas as pd
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

introduction_doc = nlp("Keanu Reeves is an Actor who played in the movie 'The Matrix' in 1999.")
print([token.text for token in introduction_doc])

displacy.serve(introduction_doc, style="ent")