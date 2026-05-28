import chess


_PIECE_TYPE_ORDER = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING
]

_PT_TO_IDX = {pt: i for i, pt in enumerate(_PIECE_TYPE_ORDER)}


def _mirror_square(sq: int) -> int:
    return sq ^ 56


def encode(board: chess.Board, perspective: bool) -> list[int]:
    indices: list[int] = []
    mirror = perspective == chess.BLACK

    for square, piece in board.piece_map().items():
        is_mine = piece.color == perspective
        pt_idx = _PT_TO_IDX[piece.piece_type] + (0 if is_mine else 6)
        sq = _mirror_square(square) if mirror else square
        indices.append(pt_idx * 64 + sq)


    return indices

def encode_both(board: chess.Board) -> tuple[list[int], list[int]]:
    stm = encode(board, board.turn)
    nstm = encode(board, not board.turn)
    return stm, nstm


def _test() -> None:
    b = chess.Board()
    stm, nstm = encode_both(b)
    assert len(stm) == 32, f"start STM count: expected 32, got {len(stm)}"
    assert len(nstm) == 32, f"start NSTM count: expected 32, got {len(nstm)}"

    assert all(0 <= i < 768 for i in stm), "index out of [0, 768)"
    
    b = chess.Board("4k3/8/8/8/8/8/8/4k2Q w - - 0 1")
    stm, nstm = encode_both(b)
    assert len(stm) == 3, f"KQK STM: expected 3, got {len(stm)}"
    assert len(nstm) == 3, f"KQK NSTM: expected 3, got {len(nstm)}"

    b.turn = chess.BLACK
    stm_b, nstm_b = encode_both(b)
    assert sorted(stm_b) == sorted(nstm), "black-to-move STM should match white-to-move NSTM"
    assert sorted(nstm_b) == sorted(stm), "black-to-move NSTM should match white-to-move STM"

    b1 = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    b2 = chess.Board("4k3/8/8/8/8/8/8/4K3 b - - 0 1")

    expected_white_view_my_king = 5 * 64 + 4
    expected_black_view_my_king = 5 * 64 + 4

    assert expected_white_view_my_king in encode(b1, chess.WHITE)
    assert expected_black_view_my_king in encode(b2, chess.BLACK)

    print(f"feature_encoder: all {6} self-tests passed.")


if __name__ == "__main__":
    _test()
    