from spacy.symbols import *
from spacy.tokens import Token

from .util import is_noun_attribute


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
        if poa.head.pos == SCONJ:
            return verb_token == poa.head.head and object_token.head == poa
        else:
            return verb_token == poa.head and object_token.head == poa
    elif object_token.dep == dobj:
        return verb_token == object_token.head
    elif object_token.dep in {acomp, amod, advmod}:
        return True
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
        # return verb_token.head == object_token
        return verb_token == object_token.head
    else:
        return False


def rule4(verb_token: Token, subject_token: Token, object_token: Token, poa: Token):
    if verb_token.dep not in {xcomp, advcl, conj}:
        return False

    if subject_token.head != verb_token and subject_token.head != verb_token.head:
        return False

    # Traverse until we reach the end or the verb is the subject's head.
    # curr_verb = verb_token
    # print(f'subject head: {subject_token.head}')
    # while subject_token.head != curr_verb:
    #     print(f'curr_verb {curr_verb} {curr_verb.pos_} {curr_verb.dep_}')
    #     i += 1
    #
    #     if curr_verb.head == curr_verb:
    #         return False  # end of traversal.
    #
    #     curr_verb = curr_verb.head
    #
    #     if curr_verb.pos != VERB:
    #         return False

    if object_token.dep == pobj:
        # we originally checked object_token.head == poa.head
        return verb_token == poa.head and object_token.head.head == poa.head
    elif object_token.dep == dobj:
        return verb_token == object_token.head
    else:
        return False


def rule5(verb_token: Token, subject_token: Token, object_token: Token, poa: Token):
    if verb_token.dep not in {ccomp, advcl, pcomp} and verb_token.dep_ != "ROOT":
        return False

    if subject_token.head != verb_token:
        return False

# pobj requires POA
# acomp and amod (optional)
    if object_token.dep == pobj:
        return poa.head == verb_token and poa.head == subject_token.head
    elif object_token.dep in {acomp, amod, advmod}:
        return True
    else:
        return False


def rule6(verb_token: Token, subject_token: Token, object_token: Token, poa: Token):
    if verb_token.dep != xcomp and verb_token.dep != advcl:
        return False

    if verb_token.head != subject_token.head:
        return False

    if object_token.dep == pobj:
        return poa.head == verb_token and poa.head == object_token.head

    if object_token.dep in {acomp, amod, advmod}:  # dative?
        return True
    elif object_token.dep_ == "dative":
        return True
    else:
        return False


def rule7(verb_token: Token, subject_token: Token, object_token: Token, poa: Token):
    if verb_token.dep != relcl:
        return False

    if verb_token.head != subject_token:
        return False

    if object_token.dep in {pobj, acomp, amod, advmod}:  # dative?
        return poa.head == verb_token and poa.head == object_token.head
    elif object_token.dep_ == "dative":
        return True
    else:
        return False


def rule8(verb_token: Token, subject_token: Token, object_token: Token, poa: Token):
    if verb_token.dep != conj:
        return False

    if verb_token.head != subject_token.head:
        return False

    if object_token.dep == pobj:
        return poa.head == verb_token and object_token.head == poa
    if object_token.dep in {acomp, amod, advmod}:
        return True
    elif object_token.dep in {dobj, acomp, amod, advmod} and object_token.head == verb_token:
        return True


def rule9(verb_token: Token, subject_token: Token, object_token: Token, poa: Token):
    if verb_token.dep != relcl:
        return False

    noun_attribute = None

    for child in object_token.children:
        if is_noun_attribute(child):
            noun_attribute = child
            break

    if not noun_attribute:
        return False

    if subject_token.head != noun_attribute.head:
        return False

    if verb_token.head != noun_attribute:
        return False

    if object_token.dep in {pobj, acomp, amod, advmod} and poa.head == verb_token and object_token.head == poa:
        return True

    if object_token.dep in {dobj, acomp, amod, advmod} and object_token.head == verb_token:
        return True

    return False


def rule10(verb_token: Token, subject_token: Token, object_token: Token, poa: Token):
    if verb_token.head != subject_token.head:
        return False

    verb_conj = None

    for conjunct in verb_token.conjuncts:
        if conjunct.head == verb_token.head:
            verb_conj = conjunct
            break

    if verb_conj is None:
        return False

    if object_token.dep == pobj and verb_conj == poa.head and poa == object_token.head:
        return True

    if object_token.dep == dobj and verb_conj == object_token.head:
        return True

    return False


def rule11(verb_token: Token, subject_token: Token, object_token: Token, poa: Token):
    if verb_token.dep != ccomp:
        return False

    if subject_token.head != verb_token:
        return False

    verb_xcomp = None

    for child in verb_token.children:
        if child.dep == xcomp:
            verb_xcomp = child
            break

    if verb_xcomp is None:
        return False

    if object_token.dep == pobj:
        return False

    if object_token.dep in {dobj, acomp, amod, advmod} and verb_token.head == object_token.head:
        return True

    return False


def rule12(verb_token: Token, subject_token: Token, object_token: Token, poa: Token):
    if verb_token.dep != conj:
        return False

    if subject_token.head != verb_token:
        return False

    if object_token.dep in {pobj, acomp, amod, advmod} and poa.head == verb_token and object_token.head == poa:
        return True

    if object_token.dep == dobj and object_token.head == verb_token:
        return True

    return False
