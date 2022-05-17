import json
import random

import requests.exceptions
import spacy
import time
import re
import requests
import bs4
import argparse

import pandas as pd

# nlp = spacy.load("en_core_web_sm")
ignored_sections = ('References', 'See_also', 'External_links', 'References_2')


def get_random_article():
    response = requests.get("https://en.wikipedia.org/api/rest_v1/page/random/html")
    data = bs4.BeautifulSoup(response.text, 'html.parser')

    p_list = []

    for section in data.find_all('section'):
        header = section.find('h2')
        if header:
            header_id = header.get('id')
            if header_id in ignored_sections:
                continue
            if 'link' in header_id:
                continue

        for p in section.find_all('p'):
            if p.get('class') == 'asbox-body':
                continue

            p_text = p.text

            # Remove | from string as its used as a delimiter in the output csv.
            p_text = p_text.replace('|', '')

            # Remove text inside square brackets
            p_text = re.sub(r"\[.+\]", "", p_text)

            # Remove text inside parenthesis
            p_text = re.sub(r"\(.+\)", "", p_text)

            # Remove duplicate spaces.
            p_text = re.sub(r" +", " ", p_text)

            # Ignore really short sentences.
            num_words = len(p_text.split(' '))
            if num_words < 5:
                continue
            p_list.append(p_text)

    full_text = ' '.join(p_list)
    title = data.find('title').text
    return title, full_text


def default_sentence_filter(sentence):
    if len(sentence.split()) < 5:
        return False

    if sentence.endswith(':'):
        return False

    if sentence.startswith(','):
        return False

    if sentence.endswith(','):
        return False

    return True


def default_sentence_postprocess(sentence):
    sentence = sentence.strip().replace("\"", "")
    sentence += '.'
    return sentence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generates random wikipedia sentences')
    parser.add_argument('input_number', metavar='input_number', type=int,
                        help='the number of sentences to generate')
    parser.add_argument('output', metavar='output', type=str,
                        help='an output file path')
    args = parser.parse_args()

    n = args.input_number
    output_path = args.output

    output = []

    print('Generating %d sentences...' % n)

    while len(output) < n:
        try:
            title, text = get_random_article()
        except requests.exceptions.ConnectionError:
            time.sleep(1)
            print('Connection error while getting sentence. Retrying...')
            continue

        sentences = re.split(r'\.|\n', text)
        sentences = [sentence.strip() for sentence in sentences if sentence and default_sentence_filter(sentence)]
        if not sentences:
            print('no sentences')
            continue  # Retry with different article.

        sentence = random.choice(sentences)
        sentence = default_sentence_postprocess(sentence)

        output.append({"id": len(output) + 1, "title": title, "text": sentence})
        print(len(output), '/', n)

    print('exporting...')
    df = pd.DataFrame(output)
    df.set_index('id', inplace=True)
    df.to_csv(output_path, sep='|')
