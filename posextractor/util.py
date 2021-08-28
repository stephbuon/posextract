from spacy.symbols import *
from spacy.tokens import Token


def is_poa(token: Token):
    return token.dep == prep or token.dep == agent or token.dep == det


collect_children_of = [VERB, AUX, pobj, dobj]


def object_search(token: Token):
    objects = []

    visited = set()
    considering = [token, ]

    while considering:
        candidate = considering.pop(-1)

        if candidate in visited:
            continue

        visited.add(candidate)

        if candidate.dep == dobj or candidate.dep == pobj:
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
            if child not in visited and child.pos != VERB:
                considering.append(child)

        parent = candidate.head
        if parent not in visited:
            considering.append(parent)

    return objects
