from espnet2.bin.align_inference import ctc_words, refine, words_and_pauses


def test_words_follow_the_transcript_not_the_boundary_unit():
    # the path skipped the boundary unit between "ab" and "c"
    spans = [
        ("<sil>", 0, 3),
        ("a", 3, 5),
        ("b", 5, 8),
        ("c", 8, 10),
        ("<sil>", 10, 12),
    ]
    words, pauses = words_and_pauses(spans, [2, 1], "<space>", ["<sil>"])
    assert words == [("ab", 3, 8), ("c", 8, 10)]
    assert pauses == [("<sil>", 0, 3), ("<sil>", 10, 12)]


def test_boundary_and_pause_units_are_not_word_frames():
    spans = [("a", 0, 2), ("<space>", 2, 3), ("<sil>", 3, 6), ("b", 6, 9)]
    words, _ = words_and_pauses(spans, [1, 1], "<space>", ["<sil>"])
    assert words == [("a", 0, 2), ("b", 6, 9)]


def test_word_tokens_are_one_word_each():
    # --token_type word: no <space> token; every token is a word
    spans = [("HELLO", 0, 30), ("WORLD", 30, 60)]
    words, _ = words_and_pauses(spans, [1, 1], "<space>", ["<sil>"])
    assert words == [("HELLO", 0, 30), ("WORLD", 30, 60)]


def test_ctc_words_and_refine():
    tokens = ["a", "b", "<space>", "c"]
    positions = [[0, 1], [3]]
    fwd = ctc_words(tokens, [(4, 5), (7, 8), (9, 9), (10, 11)], positions)
    bwd = ctc_words(tokens, [(2, 3), (5, 6), (8, 8), (8, 9)], positions)
    assert fwd == [("ab", 4, 8), ("c", 10, 11)]
    assert bwd == [("ab", 2, 6), ("c", 8, 9)]
    hmm = [("ab", 3, 8), ("c", 8, 10)]
    # start: mean of the HMM start and the reversed-CTC word start;
    # end: mean of the HMM end and the forward-CTC word end (rounded)
    assert refine(hmm, fwd, bwd) == [("ab", 2, 8), ("c", 8, 10)]
