# Blake2 Round Bitmask One-Hot Note

This note explains a constraint concern in the Blake delegation circuits.

The issue is not that the normal software path is obviously wrong. The issue is that the circuit appears to **assume** one-hot round selection without **locally enforcing** it.

## Short Version

The Blake delegation circuits use a `round_bitmask` to decide:

- which `SIGMA` permutation to select
- whether this is the first round
- whether this is the final round in extended-control mode

However, the circuits only decompose the round word into booleans. They do **not** appear to enforce that exactly one round bit is set.

That means the circuits seem underconstrained with respect to malformed control input:

- honest software starts from a one-hot round cursor
- honest software updates it in a way that preserves one-hotness
- but the circuit itself does not seem to reject multi-hot round masks

## Where This Happens

### Single-round Blake delegation

The circuit reads one word for the round bitmask:

- [`cs/src/delegation/blake2_single_round/mod.rs:18`](../src/delegation/blake2_single_round/mod.rs:18)

Then it splits the low word into boolean bits:

- [`cs/src/delegation/blake2_single_round/mod.rs:171`](../src/delegation/blake2_single_round/mod.rs:171)

Those bits are then used as selectors:

- first-round logic:
  - [`cs/src/delegation/blake2_single_round/mod.rs:185`](../src/delegation/blake2_single_round/mod.rs:185)
- `SIGMA` permutation selection:
  - [`cs/src/delegation/blake2_single_round/mod.rs:218`](../src/delegation/blake2_single_round/mod.rs:218)

### Extended-control Blake delegation

The extended-control circuit reads the round bits out of `x12`:

- [`cs/src/delegation/blake2_round_with_extended_control/mod.rs:27`](../src/delegation/blake2_round_with_extended_control/mod.rs:27)
- [`cs/src/delegation/blake2_round_with_extended_control/mod.rs:380`](../src/delegation/blake2_round_with_extended_control/mod.rs:380)

Those bits are then used for:

- final-round detection:
  - [`cs/src/delegation/blake2_round_with_extended_control/mod.rs:433`](../src/delegation/blake2_round_with_extended_control/mod.rs:433)
- first-round handling:
  - [`cs/src/delegation/blake2_round_with_extended_control/mod.rs:443`](../src/delegation/blake2_round_with_extended_control/mod.rs:443)
- message permutation selection:
  - [`cs/src/delegation/blake2_round_with_extended_control/mod.rs:652`](../src/delegation/blake2_round_with_extended_control/mod.rs:652)

## What `split_into_bitmask` Gives You

The important point is that `split_into_bitmask` is a **binary decomposition**, not a one-hot encoding.

So if the encoded integer is:

- `1`, the bits are `[1, 0, 0, ...]`
- `2`, the bits are `[0, 1, 0, ...]`
- `3`, the bits are `[1, 1, 0, ...]`

That means multi-hot patterns are allowed unless some additional exact-one constraint is added.

## Why This Matters

The circuits use `round_bitmask` as if it were identifying the current round.

For example, in `blake2_single_round`, the selected message word is built by accumulating over all active round bits:

- [`cs/src/delegation/blake2_single_round/mod.rs:214`](../src/delegation/blake2_single_round/mod.rs:214)

If multiple round bits are `1`, then the selected message word becomes a linear combination of several different `SIGMA` rows, which does not correspond to any actual Blake round.

Likewise:

- `round_bitmask[0]` drives first-round initialization
- the extended-control circuit derives `perform_final_xor` from particular round bits

So multi-hot masks can activate incompatible phase logic at the same time.

## Why This May Not Show Up In Normal Execution

The software-facing path strongly suggests that the intended representation is one-hot.

### Initial control values are one-hot

The Blake control constants seed the round field with a single active bit:

- [`common_constants/src/delegation_types/blake2s_with_control.rs:86`](../../common_constants/src/delegation_types/blake2s_with_control.rs:86)
- [`common_constants/src/delegation_types/blake2s_with_control.rs:87`](../../common_constants/src/delegation_types/blake2s_with_control.rs:87)

### The runtime invokes the circuit once per round

The software helper repeatedly triggers the delegation CSR:

- [`blake2s_u32/src/state_with_extended_control.rs:131`](../../blake2s_u32/src/state_with_extended_control.rs:131)
- [`blake2s_u32/src/state_with_extended_control.rs:141`](../../blake2s_u32/src/state_with_extended_control.rs:141)
- [`blake2s_u32/src/state_with_extended_control.rs:288`](../../blake2s_u32/src/state_with_extended_control.rs:288)
- [`blake2s_u32/src/state_with_extended_control.rs:299`](../../blake2s_u32/src/state_with_extended_control.rs:299)

### The extended-control circuit preserves cursor-style behavior

The next `x12` control word is rebuilt by shifting the current round bits forward:

- [`cs/src/delegation/blake2_round_with_extended_control/mod.rs:818`](../src/delegation/blake2_round_with_extended_control/mod.rs:818)

So along the honest path:

- one-hot starts true
- one-hot stays true

## The Concern

That is still weaker than local enforcement.

What seems to be missing is a constraint of the form:

- `sum(round_bitmask) = 1`

or, if the padding/inactive case is allowed:

- `sum(round_bitmask) <= 1`

Without that, the circuit appears to rely on an external invariant:

- the caller provides a valid one-hot round cursor

rather than proving it internally.
