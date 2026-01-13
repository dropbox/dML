// Copyright 2024-2025 Andrew Yates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// GAP 42: Grammar Parser for Constrained Decoding
// Implements an extended Backus-Naur form (BNF) parser, producing a
// context-free grammar for constrained decoding.
//
// Based on whisper.cpp/examples/grammar-parser.cpp
//
// Example grammar for arithmetic:
//   root  ::= expr
//   expr  ::= term ([-+*/] term)*
//   term  ::= num | "(" space expr ")" space
//   num   ::= [0-9]+ space
//   space ::= [ \t\n]*

#pragma once

#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace whisper {

// Grammar element types (matches whisper.cpp)
enum GrammarType {
    GRETYPE_END            = 0,  // end of rule definition
    GRETYPE_ALT            = 1,  // start of alternate definition for rule
    GRETYPE_RULE_REF       = 2,  // non-terminal element: reference to rule
    GRETYPE_CHAR           = 3,  // terminal element: character (code point)
    GRETYPE_CHAR_NOT       = 4,  // inverse char(s) ([^a], [^a-b] [^abc])
    GRETYPE_CHAR_RNG_UPPER = 5,  // range upper bound ([a-z])
    GRETYPE_CHAR_ALT       = 6,  // alternate char to match ([ab], [a-zA])
};

// Grammar element: a single element in a rule
struct GrammarElement {
    GrammarType type;
    uint32_t value;  // Unicode code point or rule ID
};

namespace grammar_parser {

// Parse state holds symbol IDs and rules
struct ParseState {
    std::map<std::string, uint32_t> symbol_ids;
    std::vector<std::vector<GrammarElement>> rules;

    // Get C-style array of rule pointers (for compatibility)
    std::vector<const GrammarElement*> c_rules() const;

    // Check if parsing succeeded (has rules)
    bool valid() const { return !rules.empty(); }
};

/**
 * Parse a BNF grammar string into rules.
 * Returns empty ParseState on error.
 *
 * Grammar syntax:
 *   rule_name ::= expression
 *   "literal"           - literal string
 *   [abc]               - character class
 *   [a-z]               - character range
 *   [^abc]              - negated character class
 *   rule_name           - rule reference
 *   (group)             - grouping
 *   expr*               - zero or more
 *   expr+               - one or more
 *   expr?               - zero or one
 *   expr1 | expr2       - alternation
 */
ParseState parse(const char* src);

/**
 * Print grammar for debugging.
 */
void print_grammar(FILE* file, const ParseState& state);

}  // namespace grammar_parser

// Partial UTF-8 state for multi-byte characters
struct PartialUtf8 {
    uint32_t value;
    int n_remain;
};

// Grammar candidate for token matching
struct GrammarCandidate {
    int id;                      // Token ID
    const uint32_t* code_points; // UTF-8 decoded code points
    PartialUtf8 partial_utf8;    // Partial UTF-8 state
};

// Grammar state for constrained decoding
struct Grammar {
    std::vector<std::vector<GrammarElement>> rules;
    std::vector<std::vector<const GrammarElement*>> stacks;
    PartialUtf8 partial_utf8;

    // Initialize grammar from parsed rules
    void init(const grammar_parser::ParseState& state, size_t start_rule = 0);

    // Check if grammar is active
    bool active() const { return !rules.empty() && !stacks.empty(); }

    // Accept a token and update stacks
    void accept_token(const std::string& token_text);

    // Get rejected candidates (tokens that violate grammar)
    std::vector<GrammarCandidate> reject_candidates(
        const std::vector<GrammarCandidate>& candidates) const;
};

}  // namespace whisper
