import re
from collections import defaultdict

# split the text into conversations, stripping each of right whitespace and starting at "USER:"
def split_convs(text):
    convs = text.split("\n\nMESSAGE ")
    convs = [c[c.index("\n\nPROMPTER:"):].strip() for c in convs]
    return convs

def get_assistant_texts(text):
    convs = split_convs(text)
    assistant_texts = []
    for conv in convs:
        t = conv[conv.index("\n\nASSISTANT:") + len("\n\nASSISTANT:"):].strip()
        if t.endswith("[NOT YET ANNOTATED]"):
            t = t[:-len("[NOT YET ANNOTATED]")].strip()
        assistant_texts.append(t)
    return assistant_texts

def get_message_ids(text):
    # e.g. MESSAGE 60cee540-2198-4ebd-8758-c4fd36a6d9e1
    convs = text.split("\n\nMESSAGE ")
    message_ids = []
    for conv in convs:
        prompter_idx = conv.index("\n\nPROMPTER:")
        message_ids.append(conv[:prompter_idx].strip())
    return message_ids

def remove_tags(text):
    if type(text) == list:
        return [remove_tags(t) for t in text]
    pattern = r"(\S\[\[[^[\]]*\]\])|(\s\[\[[^[\]]*\]\])"
    matches = re.finditer(pattern, text)
    conv_text = text
    for match in matches:
        match_text = match.group()[1:] if not match.group()[0].isspace() else match.group()
        # if match.start() > 0 and match.end() < len(text) and text[match.start() - 1] == " " and text[match.end()] == " ":
        #     match_text = match_text + " "
        # if match.end() == len(text) and text[match.start() - 1] == " ":
        #     match_text = " " + match_text
        conv_text = conv_text.replace(match_text, "")
    return conv_text

def replace_tags(text, to_replace=("LE", "LH", "APT", "NORM", "IMP"), with_tag="[[APT]]"):
    """ Replaces tags with a single tag, by default [[APT]]
    if a tag is not in to_replace, it is removed
    to_replace: list of tags to replace
    with_tag: tag to replace with"""
    if type(text) == list:
        return [replace_tags(t, to_replace=to_replace, with_tag=with_tag) for t in text]
    pattern = r"(\S\[\[[^[\]]*\]\])|(\s\[\[[^[\]]*\]\])"
    matches = re.finditer(pattern, text)
    conv_text = text
    for match in matches:
        match_text = match.group()[1:] if not match.group()[0].isspace() else match.group()
        if any([tag in match_text for tag in to_replace]):
            conv_text = conv_text.replace(match_text, with_tag)
        else:
            conv_text = conv_text.replace(match_text, "")
    return conv_text

def get_tags(text):
    """ Returns a dict of tag: [list of string indices into clean text where tag occurs] """
    if type(text) == list:
        return [get_tags(t) for t in text]
    pattern = r"(\S\[\[[^[\]]*\]\])|(\s\[\[[^[\]]*\]\])"  # matches tags with spaces around them too
    matches = re.finditer(pattern, text)
    tags = defaultdict(list)
    cumulative_offset = 0
    for match in matches:
        match_text = match.group()[1:] if not match.group()[0].isspace() else match.group()
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
    return {tag: [1 if i in tags[tag] else 0 for i in range(0, len_text)] for tag in tags}

