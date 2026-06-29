import argparse
import io
import random
import struct
import sys
import time
from collections import Counter
from pathlib import Path
from typing import IO, Iterator

import chess
import chess.pgn
from tqdm import tqdm
import zstandard as zstd

from feature_encoder import encode_both


# Constants for binary record packing
RECORD_SIZE = 138
PAD_INDEX = 65535
RECORD_STRUCT = struct.Struct(f"<32H32HBBBe5x")

def pack_record(stm: list[int], nstm: list[int], side_to_move: int, label: float) -> bytes:
    stm_padded = stm + [PAD_INDEX] * (32 - len(stm))
    nstm_padded = nstm + [PAD_INDEX] * (32 - len(nstm))
    return RECORD_STRUCT.pack(*stm_padded, *nstm_padded, len(stm), len(nstm), side_to_move, label)


# Constants for filtering positions
MIN_RATING = 2000
MIN_PLY = 16
MAX_MATERIAL_BALANCE = 500
POSITION_SAMPLE_RATE = 1.0 # CHANGE LATER WHEN ACTUALLY GETTING THE DATASET
RNG = random.Random(0xC4E55)

# Piece values for material balance calculation (in centipawns)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

def passes_game_filter(headers: chess.pgn.Headers) -> bool:
    """Filter games based on headers. Returns True if the game passes the filter, 
    False otherwise."""
    try:
        white_elo = int(headers.get("WhiteElo", "0"))
        black_elo = int (headers.get("BlackElo", "0"))

    except ValueError:
        return False
    
    if white_elo < MIN_RATING or black_elo < MIN_RATING:
        return False
    
    termination = headers.get("Termination", "")
    if termination in ("time forfeit", "abandoned", "rules infraction"):
        return False
    
    return True


def _material_balance(board: chess.Board) -> int:
    """Calculate the material balance of the board from White's perspective."""
    balance = 0
    for piece_type, value in PIECE_VALUES.items():
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))

        balance += (white_count - black_count) * value

    return balance


def position_filter_reason(board: chess.Board, ply: int, next_move: chess.Move) -> str | None:
    """Check a position against the filter criteria. Returns None if the
    position passes, otherwise a short string naming which check failed
    (used for reject-rate diagnostics)."""
    if ply < MIN_PLY:
        return "min_ply"
    if RNG.random() >= POSITION_SAMPLE_RATE:
        return "sample_rate"
    if next_move is not None and board.is_capture(next_move):
        return "is_capture"
    if abs(_material_balance(board)) > MAX_MATERIAL_BALANCE:
        return "material_balance"
    return None


def passes_position_filter(board: chess.Board, ply: int, next_move: chess.Move) -> bool:
    """Filter positions based on ply, material balance, and randomness. Returns 
    True if the position passes the filter, False otherwise."""
    return position_filter_reason(board, ply, next_move) is None


def stm_label(result: str, side_to_move: bool) -> float | None:
    """Convert game result and side to move into a label for training. 
    Returns 1.0 for a win, 0.0 for a loss, 0.5 for a draw, and None for 
    an unknown result."""
    if result == "1/2-1/2":
        return 0.5
    elif result == "1-0":
        return 1.0 if side_to_move == chess.WHITE else 0.0
    elif result == "0-1":
        return 0.0 if side_to_move == chess.WHITE else 1.0
    return None


def open_pgn(path: Path) -> IO[str]:
    """Open a PGN file, which may be compressed with zstd. Returns a 
    text stream."""
    if path.suffix == ".zst":
        raw = path.open("rb")
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(raw)
        return io.TextIOWrapper(stream_reader, encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def iter_games(pgn_stream: IO[str]) -> Iterator[chess.pgn.Game]:
    """Yield games from a PGN stream."""
    while True:
        game = chess.pgn.read_game(pgn_stream)
        if game is None:
            return
        yield game


def parse_pgn_to_binary(pgn: Path, out_path: Path, max_positions: int | None = None) -> None:
    """Parse a PGN file and write filtered positions to a binary file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    games_seen = 0
    games_kept = 0

    # Reject-rate diagnostics: tallies WHY positions/games got dropped, so a
    # stalled-looking run (low write rate) can be told apart from "working
    # as intended, just filtering aggressively." Categories mirror the
    # checks in passes_game_filter / position_filter_reason / stm_label /
    # the oversized-feature-set guard.
    game_reject_reasons: Counter = Counter()
    position_reject_reasons: Counter = Counter()

    start_time = time.monotonic()
    last_heartbeat = start_time
    HEARTBEAT_SECONDS = 10

    with open_pgn(pgn) as pgn_in, out_path.open("wb") as out:
        pbar = tqdm(unit=" pos", desc=out_path.name)

        for game in iter_games(pgn_in):
            games_seen += 1
            if not passes_game_filter(game.headers):
                # Cheap re-check to classify *why*, for the breakdown --
                # passes_game_filter itself stays a simple bool for callers
                # that don't need the reason.
                try:
                    we = int(game.headers.get("WhiteElo", "0"))
                    be = int(game.headers.get("BlackElo", "0"))
                except ValueError:
                    game_reject_reasons["bad_elo_tag"] += 1
                    continue
                if we < MIN_RATING or be < MIN_RATING:
                    game_reject_reasons["min_rating"] += 1
                elif game.headers.get("Termination", "") in ("time forfeit", "abandoned", "rules infraction"):
                    game_reject_reasons["termination"] += 1
                else:
                    game_reject_reasons["other"] += 1
                continue

            games_kept += 1
            result = game.headers.get("Result", "*")
            board = game.board()
            moves = list(game.mainline_moves())

            for ply, move in enumerate(moves):
                next_move = moves[ply + 1] if ply + 1 < len(moves) else None
                reason = position_filter_reason(board, ply, next_move)
                if reason is None:
                    label = stm_label(result, board.turn)
                    if label is None:
                        position_reject_reasons["unknown_result"] += 1
                    else:
                        stm, nstm = encode_both(board)
                        if len(stm) <= 32 and len(nstm) <= 32:
                            out.write(pack_record(stm, nstm, int(not board.turn), label))
                            written += 1
                            if written % 10 == 0:
                                pbar.update(10)
                            if max_positions and written >= max_positions:
                                pbar.close()
                                print(f"\nStopped at max_positions={max_positions}")
                                _print_breakdown(games_seen, games_kept, written, game_reject_reasons, position_reject_reasons)
                                return written
                        else:
                            position_reject_reasons["oversized_feature_set"] += 1
                else:
                    position_reject_reasons[reason] += 1
                board.push(move)

            # Heartbeat: fires on a timer regardless of write rate, so a
            # genuinely slow-but-working run (heavy filtering) doesn't look
            # identical to a stall.
            now = time.monotonic()
            if now - last_heartbeat >= HEARTBEAT_SECONDS:
                elapsed = now - start_time
                games_per_sec = games_seen / elapsed if elapsed > 0 else 0
                top_reject = position_reject_reasons.most_common(1)
                top_str = f"{top_reject[0][0]}={top_reject[0][1]:,}" if top_reject else "none yet"
                pbar.set_postfix_str(
                    f"games={games_seen:,} ({games_per_sec:.0f}/s), "
                    f"kept={games_kept:,}, top_reject={top_str}"
                )
                last_heartbeat = now

        pbar.close()

    _print_breakdown(games_seen, games_kept, written, game_reject_reasons, position_reject_reasons)
    return written


def _print_breakdown(
    games_seen: int,
    games_kept: int,
    written: int,
    game_reject_reasons: Counter,
    position_reject_reasons: Counter,
) -> None:
    """Print a final summary with reject-rate breakdowns, so a low yield can
    be diagnosed (e.g. mostly min_rating vs mostly is_capture) rather than
    just seen as a single opaque ratio."""
    print(f"\nGames seen: {games_seen:,}   games kept (passed game filter): {games_kept:,}")
    if game_reject_reasons:
        print("  Game-level rejections:")
        for reason, count in game_reject_reasons.most_common():
            pct = 100 * count / games_seen if games_seen else 0
            print(f"    {reason:24s} {count:>10,}  ({pct:5.1f}% of all games)")

    total_positions_seen = written + sum(position_reject_reasons.values())
    print(f"\nPositions seen (from kept games): {total_positions_seen:,}   positions written: {written:,}")
    if position_reject_reasons:
        print("  Position-level rejections:")
        for reason, count in position_reject_reasons.most_common():
            pct = 100 * count / total_positions_seen if total_positions_seen else 0
            print(f"    {reason:24s} {count:>10,}  ({pct:5.1f}% of positions seen)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="Path to a .pgn or .pgn.zst file")
    ap.add_argument("output", type=Path, help="Output binary file path")
    ap.add_argument("--max", type=int, default=None, help="Stop after N positions (for testing)")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1
    
    n = parse_pgn_to_binary(args.input, args.output, max_positions=args.max)
    print(f"Wrote {n:,} positions to {args.output} "
          f"({args.output.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    