"""Recursive character text splitter.

Splits long text into overlapping chunks, preferring natural boundaries
(paragraphs > newlines > sentences > words). The character-based size is
configured to approximate a target token count — for BGE-large-en-v1.5,
~2048 characters ≈ 512 tokens of English prose.

Why character-based rather than tokenizer-driven:
    A precise tokenizer requires loading either ``tokenizers`` or the
    full model. For chunking purposes the difference is negligible and
    the character-based form keeps this module free of heavy
    dependencies. The chunk sizes you get are within ~10% of the
    tokenized counterpart for typical clinical English.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")


@dataclass
class RecursiveCharacterChunker:
    """Split text into ``~chunk_size`` character chunks with ``chunk_overlap``.

    Attributes:
        chunk_size: Target character length per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.
        separators: Hierarchy of separators tried in order. Earlier
            separators are preferred; later ones are fallbacks for
            documents that lack the earlier structure.
    """

    chunk_size: int = 2048
    chunk_overlap: int = 256
    separators: tuple[str, ...] = DEFAULT_SEPARATORS

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if not self.separators:
            raise ValueError("at least one separator required")

    def split(self, text: str) -> list[str]:
        """Return non-empty chunks covering ``text``."""
        if not text or not text.strip():
            return []
        # Recursively split on the highest-priority separator that yields
        # multiple non-trivial pieces, then merge pieces back into chunks
        # of the target size.
        pieces = self._split_recursive(text, list(self.separators))
        return self._merge(pieces)

    # ---- internal ------------------------------------------------------

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        # Find the first separator that actually appears in the text. The
        # final "" separator always splits, into individual characters.
        sep = separators[-1]
        remaining: list[str] = []
        for i, s in enumerate(separators):
            if s == "":
                sep = s
                remaining = separators[i + 1 :]
                break
            if s in text:
                sep = s
                remaining = separators[i + 1 :]
                break

        # Split, then recurse into any piece that is still too long.
        if sep == "":
            # Final fallback: break into character-sized chunks.
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        splits = text.split(sep)
        out: list[str] = []
        for piece in splits:
            # Re-attach the separator to keep the text faithful when
            # possible; for separators ending in whitespace we already
            # have a clean break.
            piece_with_sep = piece + sep if sep.strip() else piece
            if len(piece_with_sep) > self.chunk_size:
                out.extend(self._split_recursive(piece_with_sep, remaining or [""]))
            else:
                out.append(piece_with_sep)
        return [p for p in out if p]

    def _merge(self, pieces: list[str]) -> list[str]:
        """Greedily pack pieces into chunks of ~``chunk_size`` with overlap."""
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for piece in pieces:
            piece_len = len(piece)
            if current_len + piece_len <= self.chunk_size:
                current.append(piece)
                current_len += piece_len
                continue

            if current:
                chunks.append("".join(current).strip())
                # Start the next chunk with overlap drawn from the end
                # of the previous one.
                if self.chunk_overlap > 0:
                    tail = "".join(current)[-self.chunk_overlap :]
                    current = [tail, piece]
                    current_len = len(tail) + piece_len
                else:
                    current = [piece]
                    current_len = piece_len
            else:
                # A single piece exceeds chunk_size even after recursion
                # — emit it as-is rather than dropping content.
                chunks.append(piece.strip())
                current = []
                current_len = 0

        if current:
            chunks.append("".join(current).strip())

        return [c for c in chunks if c]
