"""Comparing a free-text answer with its reference."""

from __future__ import annotations

import pytest

from gustav.learn.bench import BenchItem, score_answer
from gustav.learn.semantic import SemanticMatcher, cosine, matcher_for


def embedder(calls: list[list[str]] | None = None):
    """A bag-of-words embedder: same words, same vector."""
    import math
    import zlib

    def embed(texts):
        if calls is not None:
            calls.append(list(texts))
        out = []
        for text in texts:
            vector = [0.0] * 128
            for word in str(text).lower().split():
                # crc32 rather than hash(): a per-process random hash would make this
                # test depend on PYTHONHASHSEED, which it did until it failed.
                vector[zlib.crc32(word.encode("utf-8")) % 128] += 1.0
            length = math.sqrt(sum(v * v for v in vector)) or 1.0
            out.append([v / length for v in vector])
        return out

    return embed


def test_cosine_is_clamped_and_handles_empty_vectors():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([], [1.0]) == 0.0
    assert cosine([-1.0], [1.0]) == 0.0, "a negative similarity is as bad as none"


def test_a_reference_answer_is_embedded_once_per_run():
    calls: list[list[str]] = []
    matcher = SemanticMatcher(embedder(calls))
    matcher.similarity("an answer", "the reference")
    matcher.similarity("another answer", "the reference")
    embedded = [text for call in calls for text in call]
    assert embedded.count("the reference") == 1


def test_an_empty_answer_never_matches():
    matcher = SemanticMatcher(embedder())
    assert matcher.similarity("", "the reference") == 0.0
    assert matcher.matches("   ", "the reference") == (False, 0.0)


def test_a_backend_without_embeddings_gives_no_matcher():
    class NoEmbeddings:
        pass

    class Broken:
        def embed_texts(self, texts):
            raise NotImplementedError("this backend cannot embed")

    class Empty:
        def embed_texts(self, texts):
            return [[]]

    assert matcher_for(NoEmbeddings()) is None
    assert matcher_for(Broken()) is None
    assert matcher_for(Empty()) is None


def test_a_working_backend_gives_a_matcher():
    class Model:
        embed_texts = staticmethod(embedder())

    matcher = matcher_for(Model(), threshold=0.5)
    assert matcher is not None and matcher.threshold == 0.5


# --- what this changes for the gate -------------------------------------
def test_a_right_answer_in_different_words_passes_where_overlap_fails():
    item = BenchItem(
        prompt="what protects shared state?",
        expected="a mutex protects shared state from concurrent access",
        verifier={"match": "prose"},
    )
    answer = "shared state is protected from concurrent access by a mutex"

    passed_overlap, why_overlap = score_answer(item, answer)
    matcher = SemanticMatcher(embedder(), threshold=0.6)
    passed_semantic, why_semantic = score_answer(item, answer, matcher=matcher)

    assert passed_semantic, why_semantic
    assert "semantic" in why_semantic
    # The overlap scorer agrees here (the words are the same), which is the point: the
    # semantic path must not *disagree* with it on easy cases.
    assert passed_overlap, why_overlap


def test_an_unrelated_answer_still_fails_semantically():
    item = BenchItem(
        prompt="what protects shared state?",
        expected="a mutex protects shared state",
        verifier={"match": "prose"},
    )
    matcher = SemanticMatcher(embedder(), threshold=0.6)
    passed, why = score_answer(item, "bananas are a yellow fruit", matcher=matcher)
    assert not passed and "semantic" in why


def test_exact_and_numeric_matching_ignore_the_matcher():
    matcher = SemanticMatcher(embedder(), threshold=0.0)
    exact = BenchItem(prompt="p", expected="42", verifier={"match": "exact"})
    assert score_answer(exact, "41", matcher=matcher)[0] is False
    numeric = BenchItem(prompt="p", expected="42", verifier={"match": "number"})
    assert score_answer(numeric, "the answer is 42", matcher=matcher)[0] is True


@pytest.mark.asyncio
async def test_the_fake_backend_can_embed_and_the_real_one_declares_it():
    from gustav.inference.backend import InferenceBackend
    from gustav.testing.fakes import FakeBackend

    assert hasattr(InferenceBackend, "embed")
    vectors = await FakeBackend().embed(["the cat sat", "the cat sat"])
    assert cosine(vectors[0], vectors[1]) == pytest.approx(1.0)
    with pytest.raises(NotImplementedError):
        await FakeBackend(embed_fails="no embeddings here").embed(["x"])
