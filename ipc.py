"""Tiny message passing between the duck, eye and brain programs.

Each message is a dict of numpy arrays sent as one length-prefixed .npz blob over a
pipe. Workers call `protocol_streams()` first: it moves the real stdout to a private
pipe for messages and points everything else that prints (including C libraries) at
stderr, so log output can never corrupt a message.
"""
import io
import os
import struct
import sys

import numpy as np


def send(stream, **arrays):
    buf = io.BytesIO()
    np.savez(buf, **{k: np.asarray(v) for k, v in arrays.items()})
    data = buf.getvalue()
    stream.write(struct.pack("<Q", len(data)))
    stream.write(data)
    stream.flush()


def _read_exact(stream, n):
    chunks, got = [], 0
    while got < n:
        c = stream.read(n - got)
        if not c:
            raise EOFError("pipe closed")
        chunks.append(c)
        got += len(c)
    return b"".join(chunks)


def recv(stream):
    n = struct.unpack("<Q", _read_exact(stream, 8))[0]
    with np.load(io.BytesIO(_read_exact(stream, n)), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def protocol_streams():
    """Return (in_stream, out_stream) for messages; redirect fd 1 to stderr."""
    out_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return sys.stdin.buffer, os.fdopen(out_fd, "wb")
