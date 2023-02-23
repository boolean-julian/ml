import pandas as pd
import spacy
from spacy import displacy
import wikipediaapi

wiki = wikipediaapi.Wikipedia("en")
page = wiki.page("Warner Bros")

nlp = spacy.load("en_core_web_sm")
doc = nlp(page.text)
displacy.serve(doc, style="ent", port=1000)