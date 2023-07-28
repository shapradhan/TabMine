import re

def _get_parent_node(fk_rel):
    start_str = 'REFERENCES'
    end_str = '('
    result = _get_word_between_strings(fk_rel[2], start_str, end_str).strip()
    return result
def _get_word_between_strings(text, start_str, end_str):
    pattern = re.compile(rf"{re.escape(start_str)}(.*?){re.escape(end_str)}")
    match = pattern.search(text)
    if match:
        return match.group(1)
    return None
