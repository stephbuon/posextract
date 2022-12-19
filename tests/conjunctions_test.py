from posextract import grammatical_triples

sent = 'I eat pizza, and salad, and smoothies.'

a = grammatical_triples.extract(sent)

for t in a:
    print(t)
    
# steph's output: 
# I eat pizza
# I eat salad
# I eat smoothies
