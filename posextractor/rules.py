from spacy.symbols import *
from spacy.tokens import Token


def rule1(verb_token: Token, subject_token: Token, object_token: Token, poa: Token):
    if verb_token.dep != pcomp:
        return False

    # Check if the verb head is a preposition.
    verb_head = verb_token.head
    if verb_head.dep != prep:
        return False

    # The preposition’s head must be the same as the subject’s head
    if subject_token.head != verb_head.head:
        return False

    return object_token.dep == dobj and object_token.head == verb_token


def rule2(verb_token: Token, subject_token: Token, object_token: Token, poa: Token):
    if verb_token.dep not in {ccomp, conj, relcl, advcl, pcomp} and verb_token.dep_ != "ROOT":
        return False

    if verb_token != subject_token.head:
        return False

    if object_token.dep == pobj:
        return verb_token == poa.head and object_token.head == poa
    elif object_token.dep == dobj:
        return verb_token == object_token.head
    else:
        return False


def rule3(verb_token: Token, subject_token: Token, object_token: Token, poa: Token):
    if verb_token.dep not in {relcl, acl}:
        return False

    if verb_token.head != subject_token:
        return False

    if object_token.dep == pobj:
        return verb_token == poa.head and object_token.head == poa.head
    elif object_token.dep == dobj:
        return verb_token.head == object_token
    else:
        return False


def rule4(verb_token: Token, subject_token: Token, object_token: Token, poa: Token):
    if verb_token.dep not in {xcomp, advcl, conj}:
        return False

    # should this be any verb? or our triple's verb
    if subject_token.head != verb_token:
        return False

    if object_token.dep == pobj:
        return verb_token == poa.head and object_token.head == poa.head
    elif object_token.dep == dobj:
        return verb_token == object_token.head
    else:
        return False


def rule_5(verb_token: Token, subject_token: Token, object_token: Token, poa: Token):
    pass
