"""
Reader for caseInfo.txt, the plain-text case description used by the
post-processing scripts (OpenFOAM itself does not read it).

Format: one entry per line, "name  value(s)"; text after // is a comment.
    name  0.0035                      -> number
    name  medium                      -> text
    name  23.1964  -3816.44  -46.13   -> list of numbers
    name  advancing 95  receding 75   -> named values (dictionary)
"""
from pathlib import Path

FILE_NAME = "caseInfo.txt"


def _number(text):
    try:
        return int(text)
    except ValueError:
        return float(text)


def _is_number(text):
    try:
        _number(text)
        return True
    except ValueError:
        return False


def read_case_info(path):
    """Read caseInfo.txt; path is the file or the case directory."""
    path = Path(path)
    if path.is_dir():
        path = path / FILE_NAME
    info = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.split("//", 1)[0].strip()
        if not line:
            continue
        name, *vals = line.split()
        if not vals:
            raise ValueError("%s line %d: '%s' has no value" % (path, n, name))
        if len(vals) == 1:
            info[name] = _number(vals[0]) if _is_number(vals[0]) else vals[0]
        elif all(_is_number(v) for v in vals):
            info[name] = [_number(v) for v in vals]
        elif len(vals) % 2 == 0 and not any(_is_number(v) for v in vals[0::2]) \
                and all(_is_number(v) for v in vals[1::2]):
            info[name] = {k: _number(v) for k, v in zip(vals[0::2], vals[1::2])}
        else:
            raise ValueError("%s line %d: cannot read the value of '%s'" % (path, n, name))
    return info
