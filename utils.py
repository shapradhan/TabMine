import re

def _get_parent_node(fk_rel):
    start_str = 'REFERENCES'
    end_str = '('
    result = _get_word_between_strings(fk_rel[2], start_str, end_str).strip()
    return result

def get_edges(fk_rel):
    child_node = fk_rel[0]
    parent_node = _get_parent_node(fk_rel, child_node)
    return (parent_node, child_node)

def _get_word_between_strings(text, start_str, end_str):
    pattern = re.compile(rf"{re.escape(start_str)}(.*?){re.escape(end_str)}")
    match = pattern.search(text)
    if match:
        return match.group(1)
    return None
