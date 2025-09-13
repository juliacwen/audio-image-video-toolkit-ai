#!/usr/bin/env python3
import wave
import struct
import math
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate a sine wave WAV file")
    parser.add_argument("outfile", help="Output WAV file path")
    parser.add_argument("--freq", type=float, default=1000, help="Frequency in Hz")
    parser.add_argument("--sr", type=int, default=8000, help="Sample rate in Hz")
    parser.add_argument("--duration", type=float, default=1.0, help="Duration in seconds")
    parser.add_argument("--amplitude", type=int, default=10000, help="Amplitude (max 32767)")
    args = parser.parse_args()

    n_samples = int(args.sr * args.duration)

    with wave.open(args.outfile, "wb") as wf:
        wf.setnchannels(1)       # mono
        wf.setsampwidth(2)       # 16-bit PCM
        wf.setframerate(args.sr)

        for i in range(n_samples):
            sample = int(args.amplitude * math.sin(2 * math.pi * args.freq * i / args.sr))
            wf.writeframesraw(struct.pack("<h", sample))

    print(f"WAV generated: {args.outfile} ({args.freq}Hz, {args.sr}Hz, {args.duration}s)")

if __name__ == "__main__":
    main()

