"""Check a GGUF against a recipe table: same tensor names, ggml types and data SHA-256 for every tensor.

    pip install gguf
    python verify_tensors.py magi-1.0.gguf magi-1.0.tensor-types.tsv

Exit code 0 when all tensors match. Metadata is not compared: a rebuild writes its own
imatrix path into `quantize.imatrix.file` / `quantize.imatrix.dataset` (see BUILD.md).
"""
import hashlib
import sys

import gguf

gguf_path, tsv_path = sys.argv[1], sys.argv[2]
want = {}
for line in open(tsv_path, encoding="utf-8").read().splitlines()[1:]:
    c = line.split("\t")
    want[c[0]] = (c[1], c[6])

reader = gguf.GGUFReader(gguf_path)
bad = []
for t in reader.tensors:
    if t.name not in want:
        bad.append(f"{t.name}: not in the recipe table")
        continue
    ttype, digest = want.pop(t.name)
    if t.tensor_type.name != ttype:
        bad.append(f"{t.name}: type {t.tensor_type.name}, expected {ttype}")
    elif hashlib.sha256(t.data.tobytes()).hexdigest() != digest:
        bad.append(f"{t.name}: data differs")
bad += [f"{name}: missing from the file" for name in want]

n = len(reader.tensors)
if bad:
    print(f"{len(bad)} problem(s) in {n} tensors:")
    print("\n".join("  " + b for b in bad[:20]))
    sys.exit(1)
print(f"OK: all {n} tensors match {tsv_path} (type and data SHA-256)")
