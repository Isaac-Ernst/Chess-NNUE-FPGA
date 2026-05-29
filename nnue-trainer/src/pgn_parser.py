import argparse
import io
import random
import struct
import sys
from pathlib import Path
from typing import IO, Iterator

import chess
import chess.pgn
import zstandard as zstd
import tqdm import tqdm

from feature_encoder import encode_both

RECORD_SIZE = 138
PAD_INDEX = 65535
RECORD_STRUCT = struct.Struct(f"<32H32HBBBe5x")

def pack_record(stm: list[int], nstm: list[int], side_to_move: int, label: float) -> bytes:
    stm_padded = stm + [PAD_INDEX] * (32 - len(stm))
    nstm_padded = nstm + [PAD_INDEX] * (32 - len(nstm))
    return RECORD_STRUCT.pack(*stm_padded, *nstm_padded, len(stm), len(nstm), side_to_move, label)


MIN_RATING = 2000
MIN_PLY = 16
MAX_MATERIAL_BALANCE = 500
POSITION_SAMPLE_RATE = 0.125
RNG = random.Random(0xC4E55)

def passes_game_filter(headers: chess.pgn.Headers) -> bool:
    
    ###

    return NotImplementedError


def _material_balance(board: chess.Board) -> int:

    ###

    return NotImplementedError


def passes_position_filter(board: chess.Board, ply: int, next_move: chess.Move) -> bool:
    if ply < MIN_PLY:
        return False
    if abs(_material_balance(board)) > MAX_MATERIAL_BALANCE:
        return False
    if next_move is not None and board.is_capture(next_move):
        return False
    if RNG.random() >= POSITION_SAMPLE_RATE:
        return False
    return True


def stm_label(result: str, side_to_move: bool) -> float | None:
    if result == "1/2-1/2":
        return 0.5
    elif result == "1-0":
        return 1.0 if side_to_move == chess.WHITE else 0.0
    elif result == "0-1":
        return 0.0 if side_to_move == chess.WHITE else 1.0
    return None


def open_pgn(path: Path) -> IO[str]:
    if path.suffix == ".zst":
        raw = path.open("rb")
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(raw)
        return io.TextIOWrapper(stream_reader, encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def iter_games(pgn_stream: IO[str]) -> Iterator[chess.pgn.Game]:
    while True:
        game = chess.pgn.read_game(pgn_stream)
        if game is None:
            return
        yield game


def parse_pgn_to_binary(pgn: Path, out_path: Path, max_positions: int | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    games_seen = 0

    with open_pgn(pgn_path) as pgn_in, out_path.open("wb") as out:
        pbar = tqdm(unit=" pos", desc=out_path.name)

        for game in iter_games(pgn_in):
            games_seen += 1
            if not passes_game_filter(game.headers):
                continue

            result = game.headers.get("Result", "*")
            board = game.board()
            moves = list(game.mainline_moves())

            for ply, move in enumerate(moves):
                next_move = moves[ply + 1] if ply + 1 < len(moves) else None
                if passes_position_filter(board, ply, next_move): 
                    label = stm_label(result, board.turn)
                    if label is not None:
                        stm, nstm = encode_both(board)
                        if len(stm) <= 32 and len(nstm) <= 32:
                            out.write(pack_record(stm, nstm, int(not board.turn), label))
                            written += 1
                            if written % 1000 == 0:
                                pbar.update(1000)
                            if max_positions and written >= max_positions:
                                pbar.close()
                                print(f"\nStopped at max_positions={max_positions}")
                                return written
                board.push(move)

        pbar.close()

    print(f"\nGames seen: {games_seen:,}   positions written: {written:,}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path, help="Path to a .pgn or .pgn.zst file")
    ap.add_argument("output", type+Path, help="Output binary file path")
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
