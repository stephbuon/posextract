import re

import lark
from lark import Lark
from lark.visitors import Transformer
from posextract.posrule.grammar import GRAMMAR
from typing import List, Union, Optional, Tuple
from dataclasses import dataclass

from posextract.triple_extraction import TripleExtraction

from enum import IntEnum

RULE_PARSER = Lark(GRAMMAR, start='start_posrule', parser='lalr', lexer='contextual', debug=False)


class VarEnum(IntEnum):
    SUBJECT = 0
    VERB = 1
    PREDICATE = 2

    @staticmethod
    def get_value(s: str):
        return getattr(VarEnum, s)


@dataclass
class EqualityRule:
    var: VarEnum
    value: Union[str, re.Pattern]

    def eval(self, triple: TripleExtraction):
        if self.var == VarEnum.SUBJECT:
            target = triple.subject
        elif self.var == VarEnum.VERB:
            target = triple.verb
        elif self.var == VarEnum.PREDICATE:
            target = triple.object
        else:
            raise ValueError('invalid variable name')

        if isinstance(self.value, re.Pattern):
            return self.value.match(target.text) is not None
        else:
            return self.value == target.text


class ExpressionEnum(IntEnum):
    OR = 0
    AND = 1
    IGNORE = 2

    @staticmethod
    def get_value(s: str):
        return getattr(ExpressionEnum, s)


@dataclass
class Expression:
    op: ExpressionEnum
    lrule: Union[EqualityRule, 'Expression']
    rrule: Optional[Union[EqualityRule, 'Expression']] = None

    def eval(self, triple: TripleExtraction):
        if self.op == ExpressionEnum.AND:
            return self.lrule.eval(triple) and self.rrule.eval(triple)
        elif self.op == ExpressionEnum.OR:
            return self.lrule.eval(triple) or self.rrule.eval(triple)
        elif self.op == ExpressionEnum.IGNORE:
            return not self.lrule.eval(triple)
        else:
            raise ValueError('unknown op')

    def pretty_print(self, depth=0):
        print('  ' * depth + str(self.op))
        if isinstance(self.lrule, Expression):
            self.lrule.pretty_print(depth=depth + 1)
        else:
            print('  ' * depth + '  ' + str(self.lrule))

        if isinstance(self.rrule, Expression):
            self.rrule.pretty_print(depth=depth + 1)
        else:
            print('  ' * depth + '  ' + str(self.rrule))


class PosRuleTransformer(Transformer):
    def start_posrule(self, tree):
        return tree

    def statement(self, tree):
        return tree[0]

    def match_statement(self, tree):
        return tree[1]

    def ignore_statement(self, tree):
        return Expression(ExpressionEnum.IGNORE, tree[1])

    def non_double_quote(self, tree):
        token: lark.Token = tree[0]
        return token.value

    def string_character(self, tree):
        return tree[0]

    def string_literal(self, characters: List[str]):
        return ''.join(characters)

    def literal(self, tree):
        return tree[0]

    def rule_variable(self, tree):
        return tree[0]

    def equality_rule(self, tree):
        return EqualityRule(VarEnum.get_value(tree[0].value), tree[1])

    def regex_literal(self, tree):
        return re.compile(''.join(tree))

    def operator(self, tree):
        return tree[0]

    def rule(self, tree):
        if len(tree) == 1:
            return tree[0]
        elif len(tree) == 3 and isinstance(tree[1], lark.Token) and tree[1].type in {'KW_AND', 'KW_OR'}:
            return Expression(ExpressionEnum.get_value(tree[1].value), tree[0], tree[2])
        elif isinstance(tree[0], lark.Token) and tree[0].type == 'KW_LPAREN':
            rparen = 0
            for i in range(1, len(tree)):
                if isinstance(tree[i], lark.Token) and tree[i].type == 'KW_RPAREN':
                    rparen = i
            subtree_left = self.rule(tree[1:rparen])
            if not isinstance(subtree_left, list):
                subtree_left = [subtree_left, ]
            subtree_right = self.rule(tree[rparen + 1:])
            if not isinstance(subtree_left, list):
                subtree_right = [subtree_right, ]
            return self.rule(subtree_left + subtree_right)
        return tree


def parse_posrule(filepath) -> Expression:
    print('Parsing: %s' % filepath)
    with open(filepath, 'r') as f:
        data = f.read()
    parse_tree = RULE_PARSER.parse(data)
    return condense_expressions(PosRuleTransformer().transform(parse_tree))


def condense_expressions(expressions: List[Expression]) -> Expression:
    if not expressions:
        raise ValueError('empty expressions')
    if len(expressions) == 1:
        return expressions[0]

    matches, ignores = split_expressions(expressions)

    if not matches:
        root_expression = ignores[0]
    elif len(matches) == 1:
        root_expression = matches[0]
    else:
        root_expression = Expression(op=ExpressionEnum.OR, lrule=matches[0], rrule=matches[1])

    for i in range(2, len(matches)):
        root_expression = Expression(op=ExpressionEnum.OR, lrule=root_expression, rrule=matches[i])

    if ignores and matches:
        root_expression = Expression(op=ExpressionEnum.AND, lrule=root_expression, rrule=ignores[0])

    for i in range(1, len(ignores)):
        root_expression = Expression(op=ExpressionEnum.AND, lrule=root_expression, rrule=ignores[i])

    # root_expression.pretty_print()
    return root_expression


def split_expressions(expressions: List[Expression]) -> Tuple[List[Expression], List[Expression]]:
    match_filters = []
    ignore_filters = []
    for exp in expressions:
        if isinstance(exp, Expression) and exp.op == ExpressionEnum.IGNORE:
            ignore_filters.append(exp)
        else:
            match_filters.append(exp)

    return match_filters, ignore_filters


__all__ = ['RULE_PARSER', 'PosRuleTransformer',
           'split_expressions', 'parse_posrule',
           'ExpressionEnum', 'Expression']



