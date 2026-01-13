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

// GAP 42: Grammar Parser Implementation
// Ported from whisper.cpp/examples/grammar-parser.cpp

#include "grammar_parser.h"
#include <stdexcept>
#include <cstdio>
#include <cassert>

namespace whisper {
namespace grammar_parser {

// UTF-8 decoder (assumes valid UTF-8)
static std::pair<uint32_t, const char*> decode_utf8(const char* src) {
    static const int lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t first_byte = static_cast<uint8_t>(*src);
    uint8_t highbits = first_byte >> 4;
    int len = lookup[highbits];
    uint8_t mask = (1 << (8 - len)) - 1;
    uint32_t value = first_byte & mask;
    const char* end = src + len;
    const char* pos = src + 1;
    for (; pos < end && *pos; pos++) {
        value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
    }
    return std::make_pair(value, pos);
}

static uint32_t get_symbol_id(ParseState& state, const char* src, size_t len) {
    uint32_t next_id = static_cast<uint32_t>(state.symbol_ids.size());
    auto result = state.symbol_ids.insert(std::make_pair(std::string(src, len), next_id));
    return result.first->second;
}

static uint32_t generate_symbol_id(ParseState& state, const std::string& base_name) {
    uint32_t next_id = static_cast<uint32_t>(state.symbol_ids.size());
    state.symbol_ids[base_name + '_' + std::to_string(next_id)] = next_id;
    return next_id;
}

static void add_rule(ParseState& state, uint32_t rule_id, const std::vector<GrammarElement>& rule) {
    if (state.rules.size() <= rule_id) {
        state.rules.resize(rule_id + 1);
    }
    state.rules[rule_id] = rule;
}

static bool is_word_char(char c) {
    return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '-' || ('0' <= c && c <= '9');
}

static std::pair<uint32_t, const char*> parse_hex(const char* src, int size) {
    const char* pos = src;
    const char* end = src + size;
    uint32_t value = 0;
    for (; pos < end && *pos; pos++) {
        value <<= 4;
        char c = *pos;
        if ('a' <= c && c <= 'f') {
            value += c - 'a' + 10;
        } else if ('A' <= c && c <= 'F') {
            value += c - 'A' + 10;
        } else if ('0' <= c && c <= '9') {
            value += c - '0';
        } else {
            break;
        }
    }
    if (pos != end) {
        throw std::runtime_error("expecting " + std::to_string(size) + " hex chars at " + src);
    }
    return std::make_pair(value, pos);
}

static const char* parse_space(const char* src, bool newline_ok) {
    const char* pos = src;
    while (*pos == ' ' || *pos == '\t' || *pos == '#' ||
           (newline_ok && (*pos == '\r' || *pos == '\n'))) {
        if (*pos == '#') {
            while (*pos && *pos != '\r' && *pos != '\n') {
                pos++;
            }
        } else {
            pos++;
        }
    }
    return pos;
}

static const char* parse_name(const char* src) {
    const char* pos = src;
    while (is_word_char(*pos)) {
        pos++;
    }
    if (pos == src) {
        throw std::runtime_error(std::string("expecting name at ") + src);
    }
    return pos;
}

static std::pair<uint32_t, const char*> parse_char(const char* src) {
    if (*src == '\\') {
        switch (src[1]) {
            case 'x': return parse_hex(src + 2, 2);
            case 'u': return parse_hex(src + 2, 4);
            case 'U': return parse_hex(src + 2, 8);
            case 't': return std::make_pair(static_cast<uint32_t>('\t'), src + 2);
            case 'r': return std::make_pair(static_cast<uint32_t>('\r'), src + 2);
            case 'n': return std::make_pair(static_cast<uint32_t>('\n'), src + 2);
            case '\\':
            case '"':
            case '[':
            case ']':
                return std::make_pair(static_cast<uint32_t>(src[1]), src + 2);
            default:
                throw std::runtime_error(std::string("unknown escape at ") + src);
        }
    } else if (*src) {
        return decode_utf8(src);
    }
    throw std::runtime_error("unexpected end of input");
}

// Forward declaration
static const char* parse_alternates(
    ParseState& state,
    const char* src,
    const std::string& rule_name,
    uint32_t rule_id,
    bool is_nested);

static const char* parse_sequence(
    ParseState& state,
    const char* src,
    const std::string& rule_name,
    std::vector<GrammarElement>& out_elements,
    bool is_nested) {

    size_t last_sym_start = out_elements.size();
    const char* pos = src;

    while (*pos) {
        if (*pos == '"') {
            // Literal string
            pos++;
            last_sym_start = out_elements.size();
            while (*pos != '"') {
                auto char_pair = parse_char(pos);
                pos = char_pair.second;
                out_elements.push_back({GRETYPE_CHAR, char_pair.first});
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '[') {
            // Character range(s)
            pos++;
            GrammarType start_type = GRETYPE_CHAR;
            if (*pos == '^') {
                pos++;
                start_type = GRETYPE_CHAR_NOT;
            }
            last_sym_start = out_elements.size();
            while (*pos != ']') {
                auto char_pair = parse_char(pos);
                pos = char_pair.second;
                GrammarType type = last_sym_start < out_elements.size()
                    ? GRETYPE_CHAR_ALT
                    : start_type;

                out_elements.push_back({type, char_pair.first});
                if (pos[0] == '-' && pos[1] != ']') {
                    auto endchar_pair = parse_char(pos + 1);
                    pos = endchar_pair.second;
                    out_elements.push_back({GRETYPE_CHAR_RNG_UPPER, endchar_pair.first});
                }
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (is_word_char(*pos)) {
            // Rule reference
            const char* name_end = parse_name(pos);
            uint32_t ref_rule_id = get_symbol_id(state, pos, name_end - pos);
            pos = parse_space(name_end, is_nested);
            last_sym_start = out_elements.size();
            out_elements.push_back({GRETYPE_RULE_REF, ref_rule_id});
        } else if (*pos == '(') {
            // Grouping - parse nested alternates into synthesized rule
            pos = parse_space(pos + 1, true);
            uint32_t sub_rule_id = generate_symbol_id(state, rule_name);
            pos = parse_alternates(state, pos, rule_name, sub_rule_id, true);
            last_sym_start = out_elements.size();
            // Output reference to synthesized rule
            out_elements.push_back({GRETYPE_RULE_REF, sub_rule_id});
            if (*pos != ')') {
                throw std::runtime_error(std::string("expecting ')' at ") + pos);
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '*' || *pos == '+' || *pos == '?') {
            // Repetition operator
            if (last_sym_start == out_elements.size()) {
                throw std::runtime_error(std::string("expecting preceding item to */+/? at ") + pos);
            }

            // Apply transformation to previous symbol according to rewrite rules:
            // S* --> S' ::= S S' |
            // S+ --> S' ::= S S' | S
            // S? --> S' ::= S |
            uint32_t sub_rule_id = generate_symbol_id(state, rule_name);
            std::vector<GrammarElement> sub_rule;

            // Add preceding symbol to generated rule
            sub_rule.insert(sub_rule.end(),
                out_elements.begin() + last_sym_start,
                out_elements.end());

            if (*pos == '*' || *pos == '+') {
                // Cause generated rule to recurse
                sub_rule.push_back({GRETYPE_RULE_REF, sub_rule_id});
            }
            // Mark start of alternate def
            sub_rule.push_back({GRETYPE_ALT, 0});

            if (*pos == '+') {
                // Add preceding symbol as alternate only for '+' (otherwise empty)
                sub_rule.insert(sub_rule.end(),
                    out_elements.begin() + last_sym_start,
                    out_elements.end());
            }
            sub_rule.push_back({GRETYPE_END, 0});
            add_rule(state, sub_rule_id, sub_rule);

            // In original rule, replace previous symbol with reference to generated rule
            out_elements.resize(last_sym_start);
            out_elements.push_back({GRETYPE_RULE_REF, sub_rule_id});

            pos = parse_space(pos + 1, is_nested);
        } else {
            break;
        }
    }
    return pos;
}

static const char* parse_alternates(
    ParseState& state,
    const char* src,
    const std::string& rule_name,
    uint32_t rule_id,
    bool is_nested) {

    std::vector<GrammarElement> rule;
    const char* pos = parse_sequence(state, src, rule_name, rule, is_nested);

    while (*pos == '|') {
        rule.push_back({GRETYPE_ALT, 0});
        pos = parse_space(pos + 1, true);
        pos = parse_sequence(state, pos, rule_name, rule, is_nested);
    }
    rule.push_back({GRETYPE_END, 0});
    add_rule(state, rule_id, rule);
    return pos;
}

static const char* parse_rule(ParseState& state, const char* src) {
    const char* name_end = parse_name(src);
    const char* pos = parse_space(name_end, false);
    size_t name_len = name_end - src;
    uint32_t rule_id = get_symbol_id(state, src, name_len);
    const std::string name(src, name_len);

    if (!(pos[0] == ':' && pos[1] == ':' && pos[2] == '=')) {
        throw std::runtime_error(std::string("expecting ::= at ") + pos);
    }
    pos = parse_space(pos + 3, true);

    pos = parse_alternates(state, pos, name, rule_id, false);

    if (*pos == '\r') {
        pos += pos[1] == '\n' ? 2 : 1;
    } else if (*pos == '\n') {
        pos++;
    } else if (*pos) {
        throw std::runtime_error(std::string("expecting newline or end at ") + pos);
    }
    return parse_space(pos, true);
}

ParseState parse(const char* src) {
    try {
        ParseState state;
        const char* pos = parse_space(src, true);
        while (*pos) {
            pos = parse_rule(state, pos);
        }
        return state;
    } catch (const std::exception& err) {
        fprintf(stderr, "%s: error parsing grammar: %s\n", __func__, err.what());
        return ParseState();
    }
}

std::vector<const GrammarElement*> ParseState::c_rules() const {
    std::vector<const GrammarElement*> result;
    result.reserve(rules.size());
    for (const auto& rule : rules) {
        result.push_back(rule.data());
    }
    return result;
}

static void print_grammar_char(FILE* file, uint32_t c) {
    if (0x20 <= c && c <= 0x7f) {
        fprintf(file, "%c", static_cast<char>(c));
    } else {
        fprintf(file, "<U+%04X>", c);
    }
}

static bool is_char_element(GrammarElement elem) {
    switch (elem.type) {
        case GRETYPE_CHAR:
        case GRETYPE_CHAR_NOT:
        case GRETYPE_CHAR_ALT:
        case GRETYPE_CHAR_RNG_UPPER:
            return true;
        default:
            return false;
    }
}

void print_grammar(FILE* file, const ParseState& state) {
    // Build reverse map from rule ID to name
    std::map<uint32_t, std::string> rule_names;
    for (const auto& kv : state.symbol_ids) {
        rule_names[kv.second] = kv.first;
    }

    for (size_t i = 0; i < state.rules.size(); i++) {
        const auto& rule = state.rules[i];
        fprintf(file, "%s ::= ", rule_names[i].c_str());

        for (size_t j = 0; j < rule.size(); j++) {
            const auto& elem = rule[j];
            switch (elem.type) {
                case GRETYPE_END:
                    break;
                case GRETYPE_ALT:
                    fprintf(file, "| ");
                    break;
                case GRETYPE_RULE_REF:
                    fprintf(file, "%s ", rule_names[elem.value].c_str());
                    break;
                case GRETYPE_CHAR:
                    fprintf(file, "[");
                    print_grammar_char(file, elem.value);
                    break;
                case GRETYPE_CHAR_NOT:
                    fprintf(file, "[^");
                    print_grammar_char(file, elem.value);
                    break;
                case GRETYPE_CHAR_RNG_UPPER:
                    fprintf(file, "-");
                    print_grammar_char(file, elem.value);
                    fprintf(file, "] ");
                    break;
                case GRETYPE_CHAR_ALT:
                    print_grammar_char(file, elem.value);
                    if (j + 1 < rule.size() && !is_char_element(rule[j + 1])) {
                        fprintf(file, "] ");
                    }
                    break;
            }
            // Close character class if last char element
            if (is_char_element(elem) && elem.type != GRETYPE_CHAR_RNG_UPPER) {
                if (j + 1 < rule.size() && !is_char_element(rule[j + 1])) {
                    fprintf(file, "] ");
                } else if (j + 1 == rule.size()) {
                    fprintf(file, "] ");
                }
            }
        }
        fprintf(file, "\n");
    }
}

}  // namespace grammar_parser

// ===========================================================================
// Grammar runtime implementation
// ===========================================================================

static bool is_end_of_sequence(const GrammarElement* pos) {
    switch (pos->type) {
        case GRETYPE_END:
        case GRETYPE_ALT:
            return true;
        default:
            return false;
    }
}

static std::pair<bool, const GrammarElement*> grammar_match_char(
    const GrammarElement* pos,
    uint32_t chr) {

    bool found = false;
    bool is_positive_char = pos->type == GRETYPE_CHAR;

    assert(is_positive_char || pos->type == GRETYPE_CHAR_NOT);

    do {
        if (pos[1].type == GRETYPE_CHAR_RNG_UPPER) {
            // Inclusive range, e.g. [a-z]
            found = found || (pos->value <= chr && chr <= pos[1].value);
            pos += 2;
        } else {
            // Single char
            found = found || (pos->value == chr);
            pos += 1;
        }
    } while (pos->type == GRETYPE_CHAR_ALT);

    return std::make_pair(found == is_positive_char, pos);
}

static void advance_stack(
    const std::vector<std::vector<GrammarElement>>& rules,
    const std::vector<const GrammarElement*>& stack,
    std::vector<std::vector<const GrammarElement*>>& new_stacks) {

    if (stack.empty()) {
        new_stacks.emplace_back();
        return;
    }

    const GrammarElement* pos = stack.back();

    switch (pos->type) {
        case GRETYPE_RULE_REF: {
            const size_t rule_id = static_cast<size_t>(pos->value);
            const GrammarElement* subpos = rules[rule_id].data();
            do {
                // Init new stack without the top (pos)
                std::vector<const GrammarElement*> new_stack(stack.begin(), stack.end() - 1);
                if (!is_end_of_sequence(pos + 1)) {
                    // If this rule ref is followed by another element, add that to stack
                    new_stack.push_back(pos + 1);
                }
                if (!is_end_of_sequence(subpos)) {
                    // If alternate is nonempty, add to stack
                    new_stack.push_back(subpos);
                }
                advance_stack(rules, new_stack, new_stacks);
                // Scan to next alternate
                while (!is_end_of_sequence(subpos)) {
                    subpos++;
                }
                if (subpos->type == GRETYPE_ALT) {
                    subpos++;
                } else {
                    break;
                }
            } while (true);
            break;
        }
        case GRETYPE_CHAR:
        case GRETYPE_CHAR_NOT:
            new_stacks.push_back(stack);
            break;
        default:
            // Unexpected element type
            assert(false);
    }
}

static std::vector<std::vector<const GrammarElement*>> grammar_accept(
    const std::vector<std::vector<GrammarElement>>& rules,
    const std::vector<std::vector<const GrammarElement*>>& stacks,
    uint32_t chr) {

    std::vector<std::vector<const GrammarElement*>> new_stacks;

    for (const auto& stack : stacks) {
        if (stack.empty()) {
            continue;
        }

        auto match = grammar_match_char(stack.back(), chr);
        if (match.first) {
            const GrammarElement* pos = match.second;
            std::vector<const GrammarElement*> new_stack(stack.begin(), stack.end() - 1);
            if (!is_end_of_sequence(pos)) {
                new_stack.push_back(pos);
            }
            advance_stack(rules, new_stack, new_stacks);
        }
    }

    return new_stacks;
}

// UTF-8 decoder for token strings
static std::pair<std::vector<uint32_t>, PartialUtf8> decode_utf8_string(
    const char* src, PartialUtf8 partial) {

    std::vector<uint32_t> code_points;
    const char* pos = src;

    // Continue partial sequence if any
    if (partial.n_remain > 0) {
        while (*pos && partial.n_remain > 0) {
            if ((static_cast<uint8_t>(*pos) & 0xC0) != 0x80) {
                // Invalid continuation byte - reset
                partial.value = 0;
                partial.n_remain = 0;
                break;
            }
            partial.value = (partial.value << 6) | (static_cast<uint8_t>(*pos) & 0x3F);
            partial.n_remain--;
            pos++;
        }
        if (partial.n_remain == 0 && partial.value != 0) {
            code_points.push_back(partial.value);
            partial.value = 0;
        }
    }

    // Decode remaining characters
    while (*pos) {
        uint8_t first_byte = static_cast<uint8_t>(*pos);
        if ((first_byte & 0x80) == 0) {
            // ASCII
            code_points.push_back(first_byte);
            pos++;
        } else if ((first_byte & 0xE0) == 0xC0) {
            // 2-byte sequence
            if (pos[1] && (static_cast<uint8_t>(pos[1]) & 0xC0) == 0x80) {
                uint32_t value = ((first_byte & 0x1F) << 6) |
                                (static_cast<uint8_t>(pos[1]) & 0x3F);
                code_points.push_back(value);
                pos += 2;
            } else {
                partial.value = first_byte & 0x1F;
                partial.n_remain = 1;
                pos++;
                break;
            }
        } else if ((first_byte & 0xF0) == 0xE0) {
            // 3-byte sequence
            if (pos[1] && pos[2] &&
                (static_cast<uint8_t>(pos[1]) & 0xC0) == 0x80 &&
                (static_cast<uint8_t>(pos[2]) & 0xC0) == 0x80) {
                uint32_t value = ((first_byte & 0x0F) << 12) |
                                ((static_cast<uint8_t>(pos[1]) & 0x3F) << 6) |
                                (static_cast<uint8_t>(pos[2]) & 0x3F);
                code_points.push_back(value);
                pos += 3;
            } else {
                partial.value = first_byte & 0x0F;
                partial.n_remain = pos[1] ? 1 : 2;
                pos += pos[1] ? 2 : 1;
                if (pos[-1]) {
                    partial.value = (partial.value << 6) |
                                   (static_cast<uint8_t>(pos[-1]) & 0x3F);
                }
                break;
            }
        } else if ((first_byte & 0xF8) == 0xF0) {
            // 4-byte sequence
            if (pos[1] && pos[2] && pos[3] &&
                (static_cast<uint8_t>(pos[1]) & 0xC0) == 0x80 &&
                (static_cast<uint8_t>(pos[2]) & 0xC0) == 0x80 &&
                (static_cast<uint8_t>(pos[3]) & 0xC0) == 0x80) {
                uint32_t value = ((first_byte & 0x07) << 18) |
                                ((static_cast<uint8_t>(pos[1]) & 0x3F) << 12) |
                                ((static_cast<uint8_t>(pos[2]) & 0x3F) << 6) |
                                (static_cast<uint8_t>(pos[3]) & 0x3F);
                code_points.push_back(value);
                pos += 4;
            } else {
                // Partial 4-byte sequence
                partial.value = first_byte & 0x07;
                partial.n_remain = 3;
                pos++;
                while (*pos && partial.n_remain > 0 &&
                       (static_cast<uint8_t>(*pos) & 0xC0) == 0x80) {
                    partial.value = (partial.value << 6) |
                                   (static_cast<uint8_t>(*pos) & 0x3F);
                    partial.n_remain--;
                    pos++;
                }
                break;
            }
        } else {
            // Invalid byte - skip
            pos++;
        }
    }

    code_points.push_back(0);  // Null terminator
    return {code_points, partial};
}

void Grammar::init(const grammar_parser::ParseState& state, size_t start_rule) {
    if (!state.valid()) {
        rules.clear();
        stacks.clear();
        return;
    }

    rules = state.rules;
    partial_utf8 = {0, 0};

    // Initialize stacks with the start rule
    stacks.clear();
    if (start_rule < rules.size()) {
        const GrammarElement* pos = rules[start_rule].data();
        do {
            if (!is_end_of_sequence(pos)) {
                std::vector<const GrammarElement*> init_stack;
                init_stack.push_back(pos);
                advance_stack(rules, init_stack, stacks);
            }
            // Scan to next alternate
            while (!is_end_of_sequence(pos)) {
                pos++;
            }
            if (pos->type == GRETYPE_ALT) {
                pos++;
            } else {
                break;
            }
        } while (true);
    }
}

void Grammar::accept_token(const std::string& token_text) {
    if (!active()) {
        return;
    }

    // Skip special tokens (start with [_)
    if (token_text.rfind("[_", 0) == 0) {
        return;
    }

    // Decode UTF-8
    auto decoded = decode_utf8_string(token_text.c_str(), partial_utf8);
    const auto& code_points = decoded.first;

    // Accept each code point
    for (auto it = code_points.begin(); it != code_points.end() - 1; ++it) {
        stacks = grammar_accept(rules, stacks, *it);
    }
    partial_utf8 = decoded.second;
}

// Forward declaration for recursive rejection
static std::vector<GrammarCandidate> reject_candidates_impl(
    const std::vector<std::vector<GrammarElement>>& rules,
    const std::vector<std::vector<const GrammarElement*>>& stacks,
    const std::vector<GrammarCandidate>& candidates);

static std::vector<GrammarCandidate> reject_candidates_for_stack(
    const std::vector<std::vector<GrammarElement>>& rules,
    const std::vector<const GrammarElement*>& stack,
    const std::vector<GrammarCandidate>& candidates) {

    std::vector<GrammarCandidate> rejects;

    if (stack.empty()) {
        // Empty stack - reject non-empty tokens
        for (const auto& tok : candidates) {
            if (*tok.code_points != 0 || tok.partial_utf8.n_remain != 0) {
                rejects.push_back(tok);
            }
        }
        return rejects;
    }

    const GrammarElement* stack_pos = stack.back();
    std::vector<GrammarCandidate> next_candidates;

    for (const auto& tok : candidates) {
        if (*tok.code_points == 0) {
            // End of token - check if partial UTF-8 would match
            if (tok.partial_utf8.n_remain != 0) {
                // TODO: Check partial match - simplified to reject
                rejects.push_back(tok);
            }
            continue;
        }

        auto match = grammar_match_char(stack_pos, *tok.code_points);
        if (match.first) {
            next_candidates.push_back({tok.id, tok.code_points + 1, tok.partial_utf8});
        } else {
            rejects.push_back(tok);
        }
    }

    if (next_candidates.empty()) {
        return rejects;
    }

    // Build next stacks for matched tokens
    std::vector<const GrammarElement*> stack_after(stack.begin(), stack.end() - 1);
    auto pos_after = grammar_match_char(stack_pos, 'a').second;  // Get position after char match
    if (!is_end_of_sequence(pos_after)) {
        stack_after.push_back(pos_after);
    }

    std::vector<std::vector<const GrammarElement*>> next_stacks;
    advance_stack(rules, stack_after, next_stacks);

    auto next_rejects = reject_candidates_impl(rules, next_stacks, next_candidates);
    for (const auto& tok : next_rejects) {
        rejects.push_back({tok.id, tok.code_points - 1, tok.partial_utf8});
    }

    return rejects;
}

static std::vector<GrammarCandidate> reject_candidates_impl(
    const std::vector<std::vector<GrammarElement>>& rules,
    const std::vector<std::vector<const GrammarElement*>>& stacks,
    const std::vector<GrammarCandidate>& candidates) {

    if (candidates.empty() || stacks.empty()) {
        return {};
    }

    auto rejects = reject_candidates_for_stack(rules, stacks.front(), candidates);

    for (size_t i = 1; i < stacks.size(); ++i) {
        rejects = reject_candidates_for_stack(rules, stacks[i], rejects);
    }

    return rejects;
}

std::vector<GrammarCandidate> Grammar::reject_candidates(
    const std::vector<GrammarCandidate>& candidates) const {

    if (!active()) {
        return {};
    }

    return reject_candidates_impl(rules, stacks, candidates);
}

}  // namespace whisper
