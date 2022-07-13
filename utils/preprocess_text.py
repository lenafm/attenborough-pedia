def token_filter(t, pos_filter={}):
    """
    Returns a boolean whether a token should be kept or not.

    Parameters:
    t (spacy.Doc.Token): A spaCy token from the document being preprocessed.

    Returns:
    bool: A boolean to indicate whether the token should be kept or not.

    """
    has_digit = lambda s: any(i.isdigit() for i in s)
    return (not t.is_punct
            and not has_digit(t.text)
            and len(t.text) > 2
            and not t.is_stop
            and not t.pos_ in pos_filter)


def preprocess(doc, pos_filter={}):
    """
    Splits documents into tokens, filters out unwanted tokens and lemmatizes the
    text.

    Parameters:
    doc (spacy.document)
    """
    out = list()
    for subtitle in doc:
        s = []
        for token in subtitle:
            if token_filter(token, pos_filter):
                s.append(token.text.lower())
        out.append(s)
    return out