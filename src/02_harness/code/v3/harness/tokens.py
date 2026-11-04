"""Compact token records of the sampled turns (harness v3, research/log/2026-10-31-0150.md).

A turn's completion token ids are stored as base64 of little-endian uint32 and its per-token logprobs as base64 of
little-endian float32, about 11 characters a token instead of about 30 as JSON numbers. unpack_ids and
unpack_logprobs read them back.
"""
import base64
import sys
from array import array

_U32 = "I" if array("I").itemsize == 4 else "L"


def _pack(code, values):
    a = array(code, values)
    if sys.byteorder != "little":
        a.byteswap()
    return base64.b64encode(a.tobytes()).decode("ascii")


def _unpack(code, text):
    a = array(code)
    a.frombytes(base64.b64decode(text))
    if sys.byteorder != "little":
        a.byteswap()
    return a.tolist()


def pack_ids(ids):
    return _pack(_U32, ids)


def unpack_ids(text):
    return _unpack(_U32, text)


def pack_logprobs(logprobs):
    return _pack("f", logprobs)


def unpack_logprobs(text):
    return _unpack("f", text)


def from_response(r):
    """(prompt token ids, completion token ids, completion logprobs) of a chat-completions response, each None when
    the server did not return it. vLLM returns the ids with "return_token_ids" and the logprobs with "logprobs"; with
    "return_tokens_as_token_ids" each logprob entry names its token as "token_id:<id>", which is checked against the
    ids (an entry that disagrees makes the logprobs None)."""
    ch = r.choices[0]
    prompt_ids = getattr(r, "prompt_token_ids", None)
    ids = getattr(ch, "token_ids", None)
    lp = getattr(ch, "logprobs", None)
    content = getattr(lp, "content", None) if lp is not None else None
    logprobs = None
    if content:
        logprobs = [x.logprob for x in content]
        named = [x.token[9:] if isinstance(x.token, str) and x.token.startswith("token_id:") else None for x in content]
        if ids is None and all(t is not None for t in named):
            ids = [int(t) for t in named]
        if ids is not None and (len(logprobs) != len(ids) or
                                any(t is not None and int(t) != i for t, i in zip(named, ids))):
            logprobs = None
    return (list(prompt_ids) if prompt_ids is not None else None, list(ids) if ids is not None else None, logprobs)
