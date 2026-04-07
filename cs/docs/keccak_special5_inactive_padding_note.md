# Keccak Special5 Inactive Padding Note

This note explains a local concern in `keccak_special5` around inactive rows.

## The Intended Padding Story

The circuit comment says that when we do **not** process a delegation request, the row should remain harmless:

- `execute = 0`
- memory writes are masked elsewhere
- the circuit itself should not become unsatisfiable just because the row is inactive

Relevant comment:

- [`cs/src/delegation/keccak_special5/mod.rs:215`](../src/delegation/keccak_special5/mod.rs:215)

## What `control` Encodes

`x10.low` carries the packed control word:

- low 3 bits: `precompile`
- next 3 bits: `iter`
- next 5 bits: `round`

This is unpacked in the main circuit at:

- [`cs/src/delegation/keccak_special5/mod.rs:223`](../src/delegation/keccak_special5/mod.rs:223)
- [`cs/src/delegation/keccak_special5/mod.rs:520`](../src/delegation/keccak_special5/mod.rs:520)

## What the Witness Does

There are two important witness computations driven by `control`.

### 1. `control_next` witness

The witness reads raw `control`, derives `precompile_value`, `iter_value`, and `round_value`, and then computes the next control word.

Relevant code:

- [`cs/src/delegation/keccak_special5/mod.rs:253`](../src/delegation/keccak_special5/mod.rs:253)

Important detail:

- this computation is **not** gated by `execute`
- so even if `execute = 0`, a nonzero `control` still produces a nontrivial `control_next`

### 2. Bitmask witness

Later, the circuit decomposes the same raw `control` into:

- `precompile_bitmask[0..7]`
- `iter_bitmask[0..5]`
- `round_bits[0..5]`

Relevant code:

- [`cs/src/delegation/keccak_special5/mod.rs:555`](../src/delegation/keccak_special5/mod.rs:555)

There is one special patch for inactive rows:

- [`cs/src/delegation/keccak_special5/mod.rs:584`](../src/delegation/keccak_special5/mod.rs:584)

Specifically, the witness does:

- `precompile_bitmask_bools[0] &= execute`
- `iter_bitmask_bools[0] &= execute`

This means:

- on inactive rows, `p0` and `i0` are forcibly cleared
- but the rest of the decoding still comes from raw `control`

## What the Constraints Require

After the witness assigns those bits, the circuit enforces:

1. recomposition:
   - `precompile + 8 * iter + 64 * round = control`
2. active rows:
   - if `execute = 1`, then `sum(precompile_bitmask) = 1`
   - if `execute = 1`, then `sum(iter_bitmask) = 1`
3. inactive rows:
   - if `execute = 0`, then `sum(precompile_bitmask) = 0`
   - if `execute = 0`, then `sum(iter_bitmask) = 0`

Relevant code:

- [`cs/src/delegation/keccak_special5/mod.rs:607`](../src/delegation/keccak_special5/mod.rs:607)
- [`cs/src/delegation/keccak_special5/mod.rs:615`](../src/delegation/keccak_special5/mod.rs:615)

## Why `execute = 0, control != 0` Is Dangerous

The issue is that `execute` does **not** fully gate the control semantics.

### Failure mode 1: bitmask contradiction

Suppose:

- `execute = 0`
- `control = 1`

Then raw decoding says:

- `precompile = 1`
- `iter = 0`
- `round = 0`

The witness will set:

- `precompile_bitmask[1] = 1`

But the inactive-row constraints require:

- `sum(precompile_bitmask) = 0`

That is immediately unsatisfiable.

### Failure mode 2: `control_next` contradiction

Even if the flag sums happen to be zero, `control_next` is still computed from raw `control`.

Suppose:

- `execute = 0`
- `control = 64`

Then:

- `precompile = 0`
- `iter = 0`
- `round = 1`

Because of the witness patch, the inactive row can still end up with zero precompile/iter flags. But the witness for `control_next` still uses `round = 1` and computes a nontrivial next control word, while the algebraic transition is driven by the decoded flags:

- [`cs/src/delegation/keccak_special5/mod.rs:641`](../src/delegation/keccak_special5/mod.rs:641)

On an inactive row, those flags are supposed to represent "no active precompile". So the row can still become inconsistent through `control_next`.

## Why the Table Code Matters

The permutation-index tables make the intended convention explicit:

- there is one dedicated padding case for `control == 0`
- everything else that is not a valid active configuration is treated as junk

Relevant code:

- [`cs/src/tables/keccak_precompile_related.rs:127`](../src/tables/keccak_precompile_related.rs:127)

In particular:

- `control == 0` maps to `[0, 0, 0, 0, 0, 0]`
- other invalid cases fall through to `[0, 1, 2, 3, 4, 5] // THIS IS JUNK!!!!`

So the implementation is really assuming a stronger invariant:

- if `execute = 0`, then `control = 0`

## Bottom Line

The current constraints are correct only under the external convention:

- inactive row = `execute = 0` and `control = 0`

They are **not** locally inert for arbitrary `control` values on inactive rows.

So the concern is:

- the circuit comments suggest inactive rows should be harmless
- but in practice, harmlessness relies on an extra invariant about `control`
- that invariant is not fully enforced locally in `keccak_special5`

## Practical Interpretation

If the surrounding delegation ABI guarantees:

- `execute = 0 => control = 0`

then the circuit is probably fine.

If that guarantee is not airtight, then:

- `execute = 0, control != 0`

can make the circuit unsatisfiable for reasons that are not obvious from the top-level comment alone.
