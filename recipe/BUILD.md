# Rebuilding magi-1.0 and magi-beta-1

Both models are Qwen3.5-4B with no training. Every weight tensor of the official model is re-quantized to a type chosen for its tensor class. A rebuild is one `llama-quantize` call that runs on the CPU in about two minutes.

## Inputs

| Input | What it is | Check |
|---|---|---|
| `Qwen3.5-4B-BF16.gguf` | Unsloth's BF16 GGUF of [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), from [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) | 8,424,393,632 bytes, SHA-256 `9e6e2841a75f503ccb330831832fd7861266e187e0dbf149a954219ccb8c197a` |
| `imatrix-own-traces.gguf` | the importance matrix used for both releases (this folder) | 248 entries, 22 chunks of 8,192 tokens |
| `magi-1.0.overrides.txt`, `magi-beta-1.overrides.txt` | the type for each tensor class (this folder) | 13 rules; each tensor matches at most one rule, so rule order does not matter |
| `llama-quantize` | We used the PrismML-Eng/llama.cpp fork, `version: 0.2.0-dev (build 10772, commit e8fd209c9)`, Windows x64 binary. The types involved (IQ3_XXS, Q4_K, Q6_K, Q8_0) are standard llama.cpp types. Other builds should work, but we have not tested whether they give identical bytes. | |

```sh
hf download unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-BF16.gguf --local-dir .
sha256sum Qwen3.5-4B-BF16.gguf
# 9e6e2841a75f503ccb330831832fd7861266e187e0dbf149a954219ccb8c197a
```

## Build

From the repository root:

```sh
llama-quantize --pure \
  --imatrix recipe/imatrix-own-traces.gguf \
  --tensor-type-file recipe/magi-1.0.overrides.txt \
  Qwen3.5-4B-BF16.gguf magi-1.0.gguf IQ2_XXS

llama-quantize --pure \
  --imatrix recipe/imatrix-own-traces.gguf \
  --tensor-type-file recipe/magi-beta-1.overrides.txt \
  Qwen3.5-4B-BF16.gguf magi-beta-1.gguf IQ2_XXS
```

- `--pure` turns off llama-quantize's own per-tensor adjustments, so every type comes from the override file.
- `IQ2_XXS` is only the fallback type for tensors that no rule matches. Every quantizable tensor matches a rule, so no tensor gets it. It also sets `general.file_type` (19) as in the release.
- The other 177 tensors (norms, the 4-tap DeltaNet convolution, A, Δt bias) stay in F32.
- The thread count does not change the output: builds with 8 and 12 threads gave identical tensors.
- magi-beta-1 differs from magi-1.0 only in the 24 full-attention key, value and output projections, which are Q6_K instead of Q4_K.

| | Tensors | Tensor data | Bits per weight |
|---|---|---|---|
| magi-1.0 | 426 | 1,918,808,064 bytes (1,829.9 MiB) | 3.650 |
| magi-beta-1 | 426 | 1,951,248,384 bytes (1,860.9 MiB) | 3.712 |

Bits per weight are over all 4,205,751,296 parameters. `magi-1.0.tensor-types.tsv` and `magi-beta-1.tensor-types.tsv` list every tensor with these columns:
- `ggml_type`
- `ne`: shape, in ggml order (innermost dimension first)
- `n_params` and `n_bytes`
- `group`: the grouping used in the paper's composition table (Table 3)
- `data_sha256`: SHA-256 of the tensor's data bytes in the released file

## Verify

The released files:

| File | Bytes | SHA-256 |
|---|---|---|
| magi-1.0 | 1,929,776,256 | `26b4f5b3eadb72eb51d605910a80b35e29b5c33711e404da53037b2f00ca9d0b` |
| magi-beta-1 | 1,962,216,576 | `7b70afd257974cafa9d024b405b8d238a9b0f855aa1d040f834d4a0901ffb911` |

The released files were built with exactly the commands above, run from the repository root. A rebuild done the same way should therefore match the whole-file SHA-256.

To check the tensors of a rebuild, for example one made with other paths:

```sh
pip install gguf
python recipe/verify_tensors.py magi-1.0.gguf recipe/magi-1.0.tensor-types.tsv
python recipe/verify_tensors.py magi-beta-1.gguf recipe/magi-beta-1.tensor-types.tsv
```

Both should print `OK: all 426 tensors match`.

llama-quantize writes two strings into the metadata:
- `quantize.imatrix.file`: the imatrix path given on the command line, here `recipe/imatrix-own-traces.gguf`.
- `quantize.imatrix.dataset`: the calibration file name stored in the imatrix, here `imatrix-tr.txt`.

If you give a different imatrix path, only the whole-file hash changes. The tensors stay identical.

Our check on 2026-10-23 (CPU only): both models were rebuilt with the commands above. All 426 tensors matched the original builds (type and data SHA-256), and the results are the released files.

## The importance matrix

The matrix was computed once from the BF16 model:

```sh
llama-imatrix -m Qwen3.5-4B-BF16.gguf -f imatrix-tr.txt -c 8192 --process-output -o imatrix-tr.gguf
```

GPU offload was on in our run. `imatrix-tr.txt` holds 32 thinking traces that Qwen3.5-4B wrote with thinking on (Unsloth Q8_0) for LiveCodeBench problems dated before 2025-01-18. That date is outside the benchmark's contest window (2025-01-18 to 2025-04-06).

The traces are not distributed because they contain the problem statements, so this folder ships the matrix itself. It is the file used for both releases with one change: its dataset-name string is shortened to `imatrix-tr.txt`. The tensor data is byte-identical. A recomputed matrix will be close to this one but may not be bit-identical, so use this file for an exact rebuild.

## Serving as benchmarked

The paper's runs served the GGUF with Ollama 0.40.3 at `num_ctx` 32768 with greedy decoding. The 8-bit KV cache configuration uses the same file, with `OLLAMA_KV_CACHE_TYPE=q8_0` and `OLLAMA_FLASH_ATTENTION=1` set on the Ollama server.
