from pathlib import Path

import spacy
from spacy import displacy
from posextractor.posextract import graph_tokens

nlp = spacy.load("en_core_web_sm")



# s = "The establishment of the Mauritius garrison has been slightly reduced during the last few years, though the actual strength was somewhat depleted at the beginning of that period, owing to the war."

# s = "I am certain he did it."
s = "Which car did you say that Marry wanted to buy ?"

# s = "Mr. Goulding begged to trouble the House for a few minutes, as he was possessed of some local knowledge on the subject of the petition from John Knight."
doc = nlp(s)

extractions = graph_tokens(doc, verbose=True)
svg = displacy.render(doc, style="dep", jupyter=False)
file_name = "displacy.svg"
output_path = Path(f'./{file_name}')
output_path.open("w", encoding="utf-8").write(svg)
