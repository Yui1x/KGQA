"""
Compatibility shim for a brotli/httpx2 signature mismatch.

`httpx2._decoders` calls the C brotli decompressor as

    self._decompress(data, output_buffer_limit=N)

which only works if the installed `brotli` build accepts that keyword. The build
here is the PyPI `brotli` module, whose `Decompressor.process` is a C builtin
that takes a positional argument only:

    TypeError: Decompressor.process() takes no keyword arguments

The result is that every HTTPS download through huggingface_hub fails during
gzip/brotli decoding, which is why the sentence-transformers model could not be
fetched. The bug is in the interaction, not in either library's contract, so the
fix belongs here rather than in the site-packages.

Import this module before calling into huggingface_hub or sentence_transformers:

    import tools.brotli_compat  # noqa: F401

The patch is idempotent and a no-op when the installed brotli already accepts
the keyword.
"""
from __future__ import annotations

_patched = False


def apply() -> bool:
    """Wrap brotli.Decompressor.process to ignore unsupported keyword arguments.

    Returns True if the shim was installed, False if brotli already behaves.
    """
    global _patched
    if _patched:
        return True
    try:
        import brotli
    except ImportError:  # noqa: BLE001 - brotli is optional
        return False

    decompressor_cls = getattr(brotli, "Decompressor", None)
    if decompressor_cls is None:
        return False

    original_cls = decompressor_cls

    # Probe once: if the installed build already tolerates the keyword there is
    # nothing to do, and wrapping would only add overhead.
    try:
        probe = original_cls()
        probe.process(b"", output_buffer_limit=1024)
        return False
    except TypeError:
        pass
    except Exception:  # noqa: BLE001 - any other error means it is callable
        return False

    # brotli.Decompressor is an immutable C type: its methods cannot be patched
    # and it cannot be subclassed. Replace the module attribute with a plain
    # Python wrapper that owns an instance and swallows the unsupported keyword.
    # Callers only construct it and call process()/is_finished()/finish(), which
    # is the whole of the surface httpx2 uses.
    class _CompatDecompressor:
        __slots__ = ("_inner",)

        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            self._inner = original_cls(*args, **kwargs)

        def process(self, *args, **kwargs):  # noqa: ANN002, ANN003
            kwargs.pop("output_buffer_limit", None)
            return self._inner.process(*args)

        def is_finished(self) -> bool:
            return self._inner.is_finished()

        def decompress(self, data: bytes, **kwargs) -> bytes:  # noqa: ANN003
            kwargs.pop("output_buffer_limit", None)
            return self._inner.process(data)

        def __getattr__(self, name):  # noqa: ANN001
            # Forward anything the underlying C object exposes that this wrapper
            # does not model explicitly (finish, flush, __enter__, ...). Keeps the
            # shim transparent to callers without enumerating an unstable API.
            return getattr(self._inner, name)

    _CompatDecompressor.__name__ = "Decompressor"
    brotli.Decompressor = _CompatDecompressor
    _patched = True
    return True


apply()
