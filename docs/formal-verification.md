# Formal verification of optimization passes

The repository has a small formal proof suite for two structural rewrites,
plus a conditional semantic check for `fuse_reduce_keep_dims`:

- `remove_noop_slice_update`, which replaces a `slice_update` that covers the
  complete destination with its update value.
- `remove_broadcast_tiles`, which removes a `tile` when the consumer's
  elementwise broadcast produces the same result shape without that tile.

The suite also records conditional canonical softcap multiply coverage and
scalar contracts or counterexamples for select widening, softcap reciprocal
rounding, GELU coefficient tolerance, and RMSNorm underflow. These findings
are deliberately scoped to their modeled or MIL-reference contracts; they do
not establish whole-pass correctness or native-backend IEEE behavior.

The proofs use [Z3's SMT solver](https://microsoft.github.io/z3guide/) to show
that the modeled rewrite has no counterexample for arbitrary positive tensor
dimensions. They encode the shape and index constraints that the matcher is
intended to establish, then ask Z3 whether a valid input can make the before
and after expressions differ. Those lemmas quantify over unbounded symbolic
dimensions. The tests also build before and after graphs with concrete finite
shapes using the production Core ML MIL pass implementations. Those fixtures
check the graph mutation on representative cases; they do not turn a finite
sample into a proof over every MIL graph.

The keep-dims check has a universal singleton-insertion flat-index lemma and
concrete MIL fixtures for supported reductions. Reduction slices are bounded
to 64 elements per output and modeled with an uninterpreted reducer shared by
the before and after routes. This establishes route and index preservation
under the same reduction algorithm; it does not establish floating-point
accumulation equivalence across different backends or algorithms.
This assumption also covers canonicalizing negative or reordered reduction
axes: the model uses a canonical ordered slice, not a backend execution trace.

`tests/formal/proof.py` is an MIL-dependent model checker, not a second copy of
the repository matchers. Registered-pass fixtures supply the before and after
programs; once an operation is supported, the same harness can also compare a
fixture for a Core ML Tools pass. This makes the approach extensible without
claiming blanket coverage of upstream passes. If coverage matures, the model
can be extracted into a reusable harness; no new packaging is required now.

Run the dedicated suite locally with:

```bash
hatch run proofs:check
```

The `proofs` environment uses the normal project dependencies, pins
`z3-solver==4.15.4.0`, and runs with Python 3.12 so it can import the
production pass modules. It does not require Apple's Core ML runtime or macOS;
the workflow runs on Linux. The normal test environment intentionally does not
collect `tests/formal`, since it does not install Z3. The explicit
`tests/formal` argument in `proofs:check` still collects and runs the formal
tests.

## What the checks establish

These checks establish a conditional statement about each modeled rewrite:
when the same preconditions used by the fixture and the matcher hold, the
modeled tensor result is preserved. For the two pilot passes, the dimensions
are symbolic in the Z3 lemmas and strictly positive. The MIL fixture tests are
finite examples, so they validate the implementation path without claiming to
cover every legal MIL graph.

The result depends on several trusted components: Z3's solver and model, the
formalization of MIL semantics, the assumptions encoded by the fixtures, and
Core ML Tools' MIL implementation. The tests require an `unsat` result for
each universal lemma; `sat`, `unknown`, solver exceptions, and configured
timeouts fail the test and therefore fail CI. CI trusts Z3's reported result
and does not independently replay or check a proof certificate. The suite does
not prove the Python matcher for every graph, all graph mutation corner cases,
Core ML backend behavior, or the numerical equivalence of the other
optimization passes. It is therefore a proof-backed regression check for the
covered contracts, not a complete verification of the optimizer.

For pointwise MIL operations, the checker represents tensor elements as raw
bits and uses uninterpreted functions keyed by operation, dtype, and operand
bits. This assumes that the same operation with the same operand bits produces
the same result at each output index, including the relevant backend shape
behavior. A satisfiable counterexample in that abstraction can therefore be a
model-level counterexample without being an executable arithmetic counterexample;
it must be interpreted with the modeled operation contract and backend
assumptions.

## Current pass inventory

The following inventory records the scope honestly so a new proof cannot be
mistaken for a proof of every optimization in the pipeline.

| Pass | Kind | Formal status |
| --- | --- | --- |
| `remove_noop_slice_update` | Structural tensor rewrite | SMT lemmas and finite production MIL fixtures |
| `remove_broadcast_tiles` | Structural shape/broadcast rewrite | SMT lemmas and finite production MIL fixtures |
| `broadcast_select_operands` | Mixed; inserts `add(x, 0)` to widen an operand | Scalar int/bool exactness and relaxed float contracts only; no whole-pass proof (`-0.0 + 0.0` changes sign, and NaN payload behavior is relevant) |
| `fuse_reduce_keep_dims` | Mixed; reduction shape and backend reduction behavior | Conditional modeled proof for concrete fixtures plus a universal singleton-indexing lemma; assumes the same reduction algorithm |
| `replace_decomposed_softmax` | Numerical fusion | No proof |
| `fuse_attention_to_sdpa` | Numerical fusion | No proof |
| `fuse_logit_softcap` | Numerical fusion | Conditional modeled proof for canonical `alpha * tanh(x * beta)` multiply fixtures plus scalar reciprocal facts; no unconditional backend IEEE proof |
| `fuse_gelu_erfc` | Numerical fusion | MIL reference-evaluation counterexample to blanket exactness; no backend IEEE proof |
| `fuse_gelu_tanh` | Numerical approximation fusion | MIL reference-evaluation counterexample to blanket exactness; no backend IEEE proof |
| `fuse_rmsnorm` | Numerical fusion | Z3 IEEE/MIL reference-evaluation counterexample to blanket exactness; no native-backend or FTZ proof |

The inventory covers the custom optimization passes in this repository. Core
ML Tools passes are outside the broad proof boundary, but
`tests/formal/test_upstream_passes.py` has two isolated
`common::noop_elimination` smoke fixtures (rank-1 noop-reshape plus `add`, and
rank-2 noop-reshape plus `mul`). Those fixtures exercise the shared checker;
they are not a whole-pass proof or blanket coverage of upstream passes.

### Next-pass contract findings: `broadcast_select_operands`

Separate scalar lemmas show that IEEE fp16 and fp32 addition of `+0` preserves
`fpEQ`, or produces NaN when the input is NaN, across all bit patterns; raw-bit
identity is refuted by `-0`, and a reciprocal gives a downstream `-inf` versus
`+inf` counterexample. A stronger exact-bit lemma holds after excluding NaNs
and negative zero; positive zero remains in the domain. Int32 addition of zero
and boolean OR with false are bit-exact. The relaxed float relation is
therefore not contextual equivalence. These are numerical contracts for the
inserted scalar operations, not a proof of the production matcher, graph
mutation, backend behavior, or flush-to-zero modes, and they do not change the
pass table's formal status.

### Next-pass contract findings: softcap reciprocal scaling

The scalar IEEE lemmas prove that division by `2` or `4` matches multiplication
by the exactly representable reciprocal for every non-NaN fp16/fp32 input
encoding, including infinities, signed zeros, and subnormals; the lemma proves
both operation outputs are non-NaN before comparing raw bits. For the
production `c = 30` spelling,
fixed finite witnesses show that division and multiplication by the Python
reciprocal cast to the input dtype can produce different input bits: fp16
`x=0x1a25` yields `0x068e` versus `0x068d`, and fp32 `x=0x3a83126f` yields
`0x380bcf65` versus `0x380bcf66`. Production MIL fixtures confirm the emitted
`beta` uses that cast reciprocal. These are input-rounding counterexamples
only; `tanh` may collapse the difference, so they do not establish a final
softcap output mismatch or a backend claim.

These are deliberately exact scalar subsets; the project does not make a
global approximation claim for the softcap pass from them.

The production softcap proof fixtures cover fp16/fp32 `alpha, beta` pairs
`(0.5, 2.0)` and `(3.0, 0.5)` in coremltools' executable order,
`alpha * tanh(x * beta)`. Their equivalence is conditional on the fused op
using the same typed multiply rounding and tanh implementation as the three
separate ops. Constant-left inner multiplication and a missing inner multiply
remain outside this currently proved model subset: the uninterpreted model has
no commutativity or identity axioms, so those SAT results are abstract model
counterexamples, not claims about concrete arithmetic. The remaining softcap
spellings require separate IEEE/backend evidence.

### Next-pass contract findings: GELU coefficient tolerances

The GELU matchers accept perturbed coefficients within their absolute or
relative/ULP tolerances, then replace the symbolic pattern with native GELU.
Fixed MIL reference-evaluation fixtures build both sides, explicitly cast both
outputs to declared fp32, and record finite differences at `x = 20`: erfc
half=`nextafter(0.5, +inf)` yields bits `1101004801` versus native EXACT
`1101004800`, while tanh half=`0.50004` yields `1101005639` versus native
TANH_APPROXIMATION `1101004800`. These are MIL reference-evaluation
counterexamples to a blanket exact claim, not proofs of any backend IEEE
behavior; the coefficient tolerances require a separately stated numerical
contract.

### Next-pass contract findings: RMSNorm underflow

Under IEEE binary32 round-to-nearest-even with gradual underflow, the fixed
input `x=[2^-74, 0, 0, 0]` and `epsilon=0` gives `x²=2^-148`, whose sum divided
by four rounds to zero. The original route therefore produces `+inf` in its
first lane, while the fused `l2_norm * 2` route produces `2.0`. Z3's fixed IEEE
replay and MIL reference value inference agree on those bits. This is a
counterexample to a blanket bit-exact identity, and it makes no claim about a
native Core ML backend or flush-to-zero mode.

## Coverage roadmap

Per-axis integer lemmas quantify over all positive dimensions. Production
fixtures use concrete shapes and symbolic input bits/output indices, proving
all values at every index for those graphs; dynamic shapes and nested blocks
are not yet translated.

The next stages should proceed in this order:

1. **Keep-dims reduction.** Extend the current bounded reduction fixtures and
   singleton-indexing lemma across more real MIL graph families. Retain the
   explicit assumption that before and after use the same reduction algorithm:
   shape identity alone cannot establish equal floating-point accumulation
   order or backend behavior.
2. **Select operand widening.** Establish exact contracts for boolean and
   integer operands. For floating point, choose deliberately between an IEEE
   contract that specifies signed zero, NaNs, overflow, and underflow, and a
   tolerance contract with stated finite-value and backend assumptions.
3. **Numerical fusions.** For GELU, softmax, attention, RMSNorm, softcap, and
   related fusions, state whether the contract is IEEE-level or approximate.
   Include assumptions for domains, overflow, underflow, NaNs, accumulation
   order, and backend elementary functions. A universal exact proof is not
   currently available for these passes; property tests are complementary
   evidence, not proofs, and useful error bounds require explicit assumptions.
   Do not infer a bounded-error contract from these checks: any tolerance claim
   must state its domain, error budget, and backend assumptions explicitly.
   The current softcap result is limited to the canonical exact subset above;
   the other spellings remain unresolved and need concrete IEEE/backend
   evidence rather than an implicit approximation contract.
4. **Broader integration.** Add translation-validation checks over more real
   MIL graph families, with an independent translator for operation semantics.
   Add semantic anchoring tests that compare the routing interpreter with MIL's
   constant-value inference after the semantic model is mature. A full matcher
   proof also requires formalizing matcher control flow and guards, and graph
   mutation over arbitrary valid MIL graphs; mutation tests are complementary
   evidence, not a substitute for that formalization. Keep checks fail-closed
   when translation or solver status is inconclusive.

The workflow runs on every `main` push and pull request, but it blocks merging
only when a repository administrator configures its status as a required check.
This project change does not modify repository settings or claim that merge
protection is enabled.

## Adding a proof

Start with a semantic contract that says what the rewrite preserves and names
the preconditions that make it true. Keep the contract separate from the
matcher implementation so that a bug in the matcher cannot silently become a
bug in the specification. For shape rewrites, quantify dimensions as positive
integers and model the relevant indexing or broadcasting rule directly.

Then add two layers of checks in `tests/formal`:

1. A Z3 lemma that asks for a counterexample to the contract. Assert the
   rewrite preconditions and prove the before and after expressions equal for
   arbitrary symbolic dimensions.
2. A small finite fixture that constructs the before graph using MIL, runs the
   production pass, and checks the resulting graph and evaluated tensor
   values. Include a non-match or boundary fixture where the pass must leave
   the graph unchanged.

Keep solver setup deterministic and avoid depending on a Core ML runtime.
When a pass changes floating-point arithmetic, treat exact SMT integer or real
arithmetic as a model that needs additional IEEE-754 and backend evidence; it
does not by itself prove numerical equivalence. Update this inventory and the
scope paragraph when coverage expands.

The authoritative references for the two layers are the [Z3 guide](https://microsoft.github.io/z3guide/),
Apple's [Model Intermediate Language guide](https://apple.github.io/coremltools/docs-guides/source/model-intermediate-language.html),
and the [Core ML Tools MIL graph-pass documentation](https://apple.github.io/coremltools/docs-guides/source/graph-passes-intro.html).
