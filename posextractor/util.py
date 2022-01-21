from spacy.symbols import *
from spacy.tokens import Token


VERB_DEP_TAGS = {ccomp, relcl, xcomp, acl, advcl, pcomp, csubj, csubjpass, conj}
OBJ_DEP_TAGS = {dobj, pobj, acomp} # dative?


def is_verb(token: Token):
    if token.dep_ == 'ROOT':
        return True

    return token.dep in VERB_DEP_TAGS


def is_object(token: Token):
    if token.pos == NOUN and token.dep == amod:
        return True

    return token.dep in OBJ_DEP_TAGS


def is_noun_attribute(token: Token):
    return token.pos == NOUN and token.dep == attr


def is_poa(token: Token):
    return token.dep == prep or token.dep == agent or token.dep == det


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
            if is_poa(candidate.head):
                objects.append((candidate.head, candidate))
            else:
                objects.append((None, candidate))

        for child in candidate.children:
            if child not in visited:
                considering.append(child)

    return objects


def subject_search(token: Token):
    objects = []

    visited = set()
    considering = [token, ]

    # print('Doing subject search for token: ', token)
    # print('verb.head', token.head)
    # print('verb.children', list(token.children))

    while considering:
        candidate = considering.pop(-1)

        if candidate in visited:
            continue

        visited.add(candidate)

        if candidate.dep == nsubj:
            objects.append(candidate)
        elif candidate.dep == nsubjpass:
            objects.append(candidate)

        for child in candidate.children:
            if child not in visited:
                if child.pos == VERB:
                    continue
                considering.append(child)

        parent = candidate.head
        if parent not in visited:
            considering.append(parent)

    return objects
