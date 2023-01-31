from jamo import h2j

from .symbols import symbols

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text):
    sequence = [_symbol_to_id[s] for s in h2j(text) if s in _symbol_to_id]
    return sequence


def sequence_to_text(sequence):
    result = "".join([_id_to_symbol[s] for s in sequence if s in _id_to_symbol])
    return result
