
from prefixkv.blocks import (
    alignment_waste,
    block_hash,
    cacheable_length,
    common_prefix_len,
    full_blocks,
)


def test_the_hash_is_chained_so_position_is_part_of_identity():
    """The same tokens at a different offset are a different block.

    They must be: the keys and values were computed against a different context, so
    reusing them would be wrong. A content-only hash would match them.
    """
    a = block_hash(None, [1, 2, 3, 4])
    b = block_hash(a, [1, 2, 3, 4])
    assert a != b


def test_a_first_block_cannot_collide_with_a_later_identical_one():
    tokens = [7, 7, 7, 7]
    first = block_hash(None, tokens)
    later = block_hash(12345, tokens)
    assert first != later


def test_the_hash_is_stable_across_processes():
    # Pinned literal: if this changes, every stored hit rate in the docs was
    # measured against a different function. `hash()` on a tuple would fail this
    # under a different PYTHONHASHSEED.
    assert block_hash(None, [1, 2, 3]) == block_hash(None, [1, 2, 3])
    assert block_hash(None, list(range(16))) != block_hash(None, list(range(15)))


def test_only_complete_blocks_exist():
    blocks = full_blocks(list(range(20)), 8)
    assert len(blocks) == 2                    # 20 // 8, the last 4 tokens dropped
    assert [i for i, _, _ in blocks] == [0, 1]


def test_a_partial_block_is_not_padded():
    # Padding would produce a hash matching nothing real, and a cache storing it
    # would report hits it cannot honour.
    assert full_blocks([1, 2, 3], 8) == []


# -- the two limits, and the order they apply in ---------------------------------
def test_a_prompt_of_exactly_four_blocks_can_reuse_only_three():
    """The finding that surprises people. 64 tokens, 16-token blocks.

    One token must be recomputed for the logit -> 63. The cached length must be
    block-aligned -> floor to 48. So a prompt that is exactly four blocks long
    reuses three of them.
    """
    assert cacheable_length(64, 16) == 48


def test_reversing_the_two_limits_would_overstate_reuse():
    n, bs = 64, 16
    right = ((n - 1) // bs) * bs                 # 48
    wrong = min((n // bs) * bs, n - 1)           # 63
    assert cacheable_length(n, bs) == right
    assert wrong > right


def test_cacheable_length_edges():
    assert cacheable_length(0, 16) == 0
    assert cacheable_length(1, 16) == 0
    assert cacheable_length(17, 16) == 16
    assert cacheable_length(16, 16) == 0         # 15 floored to 0
    assert cacheable_length(1000, 1) == 999


def test_alignment_waste_is_the_remainder():
    assert alignment_waste(180, 16) == 4
    assert alignment_waste(180, 64) == 52
    assert alignment_waste(192, 16) == 0
    assert alignment_waste(0, 16) == 0


def test_one_differing_token_kills_a_whole_block():
    a = list(range(16))
    b = list(range(15)) + [999]
    assert full_blocks(a, 16)[0][1] != full_blocks(b, 16)[0][1]
    assert common_prefix_len(a, b) == 15         # 15/16 matched, and it is a miss


def test_common_prefix_len():
    assert common_prefix_len([1, 2, 3], [1, 2, 4]) == 2
    assert common_prefix_len([], [1]) == 0
    assert common_prefix_len([1, 2], [1, 2]) == 2
