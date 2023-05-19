GRAMMAR = r"""

start_posrule: statement (statement*)
statement: (match_statement | ignore_statement) ";"
ignore_statement: KW_IGNORE rule 
match_statement: KW_MATCH rule


rule: KW_LPAREN rule KW_RPAREN (operator rule)*
    | equality_rule (operator rule)*

equality_rule: rule_variable "=" literal
rule_variable: KW_SUBJECT | KW_VERB | KW_PREDICATE
operator: KW_AND | KW_OR
literal: regex_literal | string_literal
regex_literal: "RE<" "\"" non_double_quote* "\"" ">"
string_literal: "\"" string_character* "\""
string_character: non_double_quote | escape_sequence
escape_sequence: "\\" /[^\x03-\x1F]/
non_double_quote: /[^\x03-\x1F"\\]/


KW_LPAREN: "("
KW_RPAREN: ")"
KW_OR: "OR"
KW_AND: "AND"
KW_PREDICATE: "PREDICATE"
KW_VERB: "VERB"
KW_SUBJECT: "SUBJECT"
KW_IGNORE: "IGNORE"
KW_MATCH: "MATCH"

%ignore COMMENT
COMMENT: /\/\/[^\n]*/

%ignore WHITESPACE
WHITESPACE: WHITESPACE_INLINE | "\r" | "\n"
WHITESPACE_INLINE: " " | "\t"
"""
