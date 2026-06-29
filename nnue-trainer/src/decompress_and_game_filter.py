import os
import queue
import subprocess
import sys
import threading
import time

zstd_exe = r"nnue-trainer\src\zstd.exe"
pgn_extract_exe = r"nnue-trainer\src\pgn-extract.exe"

# Hardcoded strictly to your 2026-04 file
input_zst = r"data\raw\lichess_db_standard_rated_2026-04.pgn.zst"
output_pgn = r"data\raw\elite_lichess_db_standard_rated_2026-04.pgn"
tag_filter_file = r"nnue-trainer\src\elo_filter.txt"

# pgn-extract's tag-based filtering reads criteria from a small text file
# passed via -t. This is the ONLY way to filter by Elo with this tool --
# there is no "-w2000 / -b2000 = min rating" shortcut. (-w sets output
# line width, -b restricts game length in *moves* -- both unrelated to
# rating, and -b2000 in particular was silently dropping almost every
# game because it required EXACTLY 2000 moves.)
if not os.path.exists(tag_filter_file):
    os.makedirs(os.path.dirname(tag_filter_file), exist_ok=True)
    with open(tag_filter_file, "w") as f:
        # Regex explanation:
        # ^         = Must start at the beginning of the tag string
        # [23]      = The first digit must be a 2 or a 3
        # [0-9] x3  = Must be followed by exactly three numeric digits
        f.write('WhiteElo /^[23][0-9][0-9][0-9]/\n')
        f.write('BlackElo /^[23][0-9][0-9][0-9]/\n')
    print(f"Created tag filter file: {tag_filter_file}")

print(f"Starting extraction for single file: 2026-04\n")

# 1. Start zstd
p_zstd = subprocess.Popen([zstd_exe, "-cdq", input_zst], stdout=subprocess.PIPE)

# 2. Start pgn-extract
#    -t <file>      : keep only games matching the tag criteria (Elo, here)
#    --minmoves N   : drop very short games/aborted games (real move-count filter)
#    -s             : silent mode, don't echo each extracted game's tags to stdout/stderr
#    --quiet        : suppress the running game counter pgn-extract normally prints
#    (Don't redirect stderr to DEVNULL while developing the filter --
#     you want to see pgn-extract's own game-matched / game-rejected counts.)
p_pgn = subprocess.Popen(
    [pgn_extract_exe, "-t", tag_filter_file, "--minmoves", "10", "-s"],
    stdin=p_zstd.stdout,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
p_zstd.stdout.close()

bytes_written = 0
games_seen = 0
MB_CHUNK = 1024 * 1024
next_flush = MB_CHUNK
start_time = time.monotonic()
last_print = start_time
HEARTBEAT_SECONDS = 10  # print *something* even during long silent stretches

# read1() blocks until data is actually available, so polling it on a timer
# from the main thread doesn't work -- the call itself just sits there during
# a long run of rejected games. Instead, read in a background thread and push
# chunks through a queue, so the main loop can use queue.get(timeout=...) to
# get a real heartbeat even when nothing has arrived yet.
chunk_queue: "queue.Queue[bytes | None]" = queue.Queue()

def _reader():
    try:
        while True:
            chunk = p_pgn.stdout.read1(65536)
            chunk_queue.put(chunk)
            if not chunk:
                return  # EOF
    except Exception as e:
        chunk_queue.put(None)
        print(f"\n[reader thread error] {e}", file=sys.stderr)

reader_thread = threading.Thread(target=_reader, daemon=True)
reader_thread.start()

# CRITICAL: stderr must also be drained continuously, not just read once at
# the end. pgn-extract can write warnings/diagnostics to stderr while running.
# If nothing reads that pipe concurrently, the stderr pipe buffer fills up
# (Windows pipe buffers are small) and pgn-extract BLOCKS trying to write to
# it -- which freezes the whole pipeline at a fixed point, every run, with
# zstd then blocking too once pgn-extract stops reading its stdin. This is
# almost certainly why the script was stalling at the same byte count.
stderr_lines: list[str] = []
total_games_parsed = 0

def _stderr_reader():
    global total_games_parsed
    try:
        for line in iter(p_pgn.stderr.readline, b""):
            decoded = line.decode(errors="replace").strip()
            if decoded:
                stderr_lines.append(decoded)
                # Catch the live counter pgn-extract outputs
                if decoded.startswith("Games:"):
                    try:
                        total_games_parsed = int(decoded.split("Games:")[1].strip())
                    except ValueError:
                        pass
    except Exception:
        pass

stderr_thread = threading.Thread(target=_stderr_reader, daemon=True)
stderr_thread.start()

with open(output_pgn, "wb") as outfile:
    while True:
        try:
            chunk = chunk_queue.get(timeout=HEARTBEAT_SECONDS)
        except queue.Empty:
            # Nothing arrived in HEARTBEAT_SECONDS. If pgn-extract is alive
            # but grinding through a long run of rejected (low-rated) games,
            # this is exactly what you'd expect -- not a hang. If it's NOT
            # alive, something actually broke.
            elapsed = time.monotonic() - start_time
            alive = p_pgn.poll() is None
            status = "pgn-extract still running" if alive else "pgn-extract EXITED -- this is NOT normal, check stderr below"
            last_stderr = stderr_lines[-1].strip() if stderr_lines else "(none)"
            print(
                f"\r[Heartbeat] {elapsed:6.0f}s -- {status}; "
                f"{games_seen:,} games matched, {bytes_written / MB_CHUNK:.1f} MB so far; "
                f"last stderr: {last_stderr}"
                + (" " * 10),
                end="", flush=True,
            )
            if not alive:
                break
            continue

        if not chunk:
            break  # real EOF from pgn-extract's stdout

        outfile.write(chunk)
        bytes_written += len(chunk)
        games_seen += chunk.count(b"[Event ")

        now = time.monotonic()
        if bytes_written >= next_flush or now - last_print >= HEARTBEAT_SECONDS:
            outfile.flush()
            os.fsync(outfile.fileno())
            elapsed = now - start_time
            print(
                f"\r[Live Progress] {elapsed:6.0f}s -- Saved {bytes_written / MB_CHUNK:.1f} MB "
                f"({games_seen:,} elite matched / {total_games_parsed:,} total scanned)...",
                end="", flush=True,
            )
            if bytes_written >= next_flush:
                next_flush += MB_CHUNK
            last_print = now

p_pgn.wait()
stderr_thread.join(timeout=2)

# Surface anything pgn-extract printed to stderr while running, so you can
# sanity-check warnings / its own match-count summary.
if stderr_lines:
    print("\n\n--- pgn-extract stderr output ---")
    print("".join(stderr_lines).strip())

print(f"\n\nFinished! Total size saved: {bytes_written / MB_CHUNK:.2f} MB")
print(f"Approx. games matched (by [Event] tag count): {games_seen:,}")
print(f"Total games scanned: {total_games_parsed:,}")

if bytes_written == 0:
    print(
        "\n[WARNING] Zero bytes written. This usually means the filter file path is "
        "wrong, the tag syntax didn't match, or zstd/pgn-extract failed silently. "
        "Re-run without piping (zstd -cdq input | pgn-extract -t filter.txt > out.pgn) "
        "directly in a terminal to see any error output live.",
        file=sys.stderr,
    )
