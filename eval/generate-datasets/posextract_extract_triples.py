d = '/home/stephbuon/Downloads/sentences-20221117T105534Z-001/sentences/hansard_sentences'
file = open(d + '/' + 'comp_sent.txt')
lines = [line.strip('\n').replace('\"', '') for line in file.readlines()] 
file.close()

with open(d + '/' + 'test.txt', 'w+') as f:
    for line in lines:
        f.write(line)
        f.write('\n')
        triples = grammatical_triples.extract(line)
        for triple in triples:
            f.write(str(triple))
            f.write('\n')
        f.write('\n')
