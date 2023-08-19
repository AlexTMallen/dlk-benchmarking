import re
from collections import defaultdict

# find the annotations in each conversation
pattern = r"\[\[[^[\]]*\]\]"  # this might fail if, for example, there's a [[0]] written as a piece of code.

# split the text into conversations, stripping each of right whitespace and starting at "USER:"
def split_convs(text):
    convs = text.split("\n\nMESSAGE")
    convs = [c[c.index("\n\nPROMPTER:"):].strip() for c in convs]
    return convs

def get_assistant_texts(text):
    convs = split_convs(text)
    assistant_texts = []
    for conv in convs:
        assistant_texts.append(conv[conv.index("\n\nASSISTANT:") + len("\n\nASSISTANT:"):].strip())
    return assistant_texts

def remove_tags(text):
    if type(text) == list:
        return [remove_tags(t) for t in text]
    matches = re.finditer(pattern, text)
    conv_text = text
    for match in matches:
        conv_text = conv_text.replace(match.group(), "")
    return conv_text

def replace_tags(text, to_replace=("LE", "LH", "APT", "NORM", "IMP"), with_tag="[[APT]]"):
    """ Replaces tags with a single tag, by default [[APT]]
    if a tag is not in to_replace, it is removed
    to_replace: list of tags to replace
    with_tag: tag to replace with"""
    if type(text) == list:
        return [replace_tags(t) for t in text]
    matches = re.finditer(pattern, text)
    conv_text = text
    for match in matches:
        match_text = match.group()
        if any([tag in match_text for tag in to_replace]):
            conv_text = conv_text.replace(match_text, with_tag)
        else:
            conv_text = conv_text.replace(match_text, "")
    return conv_text

def get_tags(text):
    """ Returns a dict of tag: [list of string indices into clean text where tag occurs] """
    if type(text) == list:
        return [get_tags(t) for t in text]
    matches = re.finditer(pattern, text)
    tags = defaultdict(list)
    cumulative_offset = 0
    for match in matches:
        match_text = match.group()
        tgs = match_text[2:-2].split(", ")
        for tag in tgs:
            tags[tag].append(match.start() - cumulative_offset)
        cumulative_offset += len(match_text)  # so that results are indices into the clean text
    return tags

def get_tag_masks(text):
    """ Returns a dict of tag: [list of 1s and 0s of length len(text), where 1 indicates a tag] """
    if type(text) == list:
        return [get_tag_masks(t) for t in text]
    len_text = len(remove_tags(text))
    tags = get_tags(text)
    return {tag: [1 if i in tags[tag] else 0 for i in range(1, len_text + 1)] for tag in tags}