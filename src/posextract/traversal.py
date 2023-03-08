from spacy.tokens import Token
from typing import List, Union

from spacy.tokens import Doc
from spacy.symbols import *

from posextract.triple_extraction import TripleExtraction
from posextract.util import is_root, get_verb_neg, is_verb, get_nlp, is_object, get_object_neg, is_poa, get_poa_neg, \
    get_subject_neg
from posextract.util import get_dep_matcher, should_consider_verb_phrase, VerbPhrase
from posextract import rules
from posextract.verb_phrase import VERB_PHRASE_TABLE

rule_funcs = [
    rules.rule1,
    rules.rule2,
    rules.rule3,
    rules.rule4,
    rules.rule5,
    rules.rule6,
    rules.rule7,
    rules.rule8,
    rules.rule9,
    rules.rule10,
    rules.rule11,
    rules.rule12,
]


def visit_verb(verb: Union[Token, VerbPhrase], parent_subjects, parent_objects, verbose=False):
    if verbose:
        print('beginning triple search for verb:', verb)
        print('verb dep=', verb.dep_)
        print('\tparent_subjects=', parent_subjects)
        print('\tparent_objects=', parent_objects)

    # Search for the subject.
    if isinstance(verb, VerbPhrase):
        subjects = subject_search(verb.subject_search_root, verbose=verbose, verb_phrase=True)
    else:
        subjects = subject_search(verb, verbose=verbose)

    # Search for the objects.
    if isinstance(verb, VerbPhrase):
        objects = object_search(verb.object_search_root) + parent_objects
    else:
        objects = object_search(verb) + parent_objects

    # Remove duplicates.
    subjects = list(set(subjects))
    objects = list(set(objects))

    if verbose:
        print('\tsubjects=', subjects)
        print('\tobjects=', objects)

    if not subjects:
        if verbose: print('Could not find subjects.')

    if not objects:
        if verbose: print('Could not find objects.')

    neg_adverb, neg_adverb_part = get_verb_neg(verb)

    for subject_negdet, subject in subjects:
        for poa_neg, poa, obj_negdet, obj in objects:
            if verbose: print('\tconsidering triple:', subject, verb, poa if poa else '', obj)

            for rule in rule_funcs:
                if rule(verb, subject, obj, poa):
                    if verbose: print('\tmatched with', rule.__name__, '\n')

                    extraction = TripleExtraction(
                        subject_negdet=subject_negdet, subject=subject,
                        neg_adverb=neg_adverb, neg_adverb_part=neg_adverb_part, verb=verb,
                        poa_neg=poa_neg, poa=poa, object_negdet=obj_negdet, object=obj,
                        rule=' <%s>' % rule.__name__,
                        verb_phrase=isinstance(verb, VerbPhrase))
                    yield extraction
                    break
            else:
                if verbose: print('\tNo matching rule found.\n')

    yield from visit_token(verb, parent_subjects=subjects, verbose=verbose)


def visit_token(token, parent_subjects, verbose=False):
    for child in token.children:
        if is_verb(child):
            yield from visit_verb(child, parent_subjects=[], parent_objects=[], verbose=verbose)
        else:
            # Reset inherited subjects and objects.
            yield from visit_token(child, [], verbose=verbose)


def graph_tokens(doc: Doc, verbose=False) -> List[TripleExtraction]:
    root_verb = None

    for token in doc:
        if is_root(token):
            root_verb = token
            if verbose: print(f"Root verb is {root_verb}")
            break

    if root_verb is None:
        if verbose: print('Could not find root verb.')
        return []

    triple_extractions = list(visit_verb(root_verb, [], [], verbose=verbose))

    dep_matcher = get_dep_matcher(get_nlp())
    matches = dep_matcher(doc)

    for match_id, token_ids in matches:
        match_type = get_nlp().vocab[match_id].text
        class_ = VERB_PHRASE_TABLE[match_type]
        verb_phrase = class_(*(doc[ti] for ti in token_ids))

        if not should_consider_verb_phrase(verb_phrase):
            if verbose: print('Disregarding verb phrase: %s' % repr(verb_phrase))
            continue

        if verbose:
            print('Matched verb phrase %s: %s' % (match_type, repr(verb_phrase)))

        triple_extractions.extend(visit_verb(verb_phrase, [], [], verbose=verbose))

    return triple_extractions


def object_search(token: Token):
    objects = []

    visited = set()
    considering = [token, ]

    while considering:
        candidate = considering.pop(-1)

        if candidate in visited:
            continue

        visited.add(candidate)

        if is_object(candidate):
            obj_negdet = get_object_neg(candidate)
            # obj_adj = get_object_adj(candidate)
            poa = candidate.head if is_poa(candidate.head) else None
            poa_neg = get_poa_neg(poa) if poa is not None else None
            objects.append((poa_neg, poa, obj_negdet, candidate))

        for child in candidate.children:
            if child not in visited:
                if child.pos == VERB or child.pos == AUX:
                    continue
                considering.append(child)

    return objects


def subject_search(token: Token, verbose=False, verb_phrase=False):
    objects = []

    visited = set()
    considering = [token, ]

    if verbose:
        print('\tDoing subject search for token: ', token)
        print('\tverb.head', token.head)
        print('\tverb.children', list(token.children))

    while considering:
        candidate = considering.pop(-1)

        if candidate in visited:
            continue

        visited.add(candidate)

        if candidate.dep == nsubj or candidate.dep == nsubjpass:
            objects.append((get_subject_neg(candidate), candidate))

        for child in candidate.children:
            if child not in visited:
                if child.pos == VERB:
                    continue
                if verb_phrase and child.pos == AUX:
                    continue

                if verbose:
                    print('\t\t(verb=%s) considering child:' % token, child.text, 'with POS=', child.pos_)
                    print('\t\tdependency of %s->%s:' % (candidate, child), child.dep_)
                considering.append(child)

        parent = candidate.head
        if parent not in visited:
            if (parent.pos == VERB or parent.pos == AUX) and (candidate.dep == conj or candidate.dep == advcl):
                continue

            if verbose:
                print('\t\t(verb=%s) considering parent:' % token, parent.text, 'with POS=', parent.pos_)
                print('\t\tdependency of %s->%s:' % (parent, candidate), candidate.dep_)
            considering.append(parent)

    return objects
