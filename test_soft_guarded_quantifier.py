"""Tests for the soft guarded quantifier. CPU only, no data files, seconds.

    python test_soft_guarded_quantifier.py

Checks that a crisp guard reproduces LTN's guarded quantification (computed
with ltn's own Quantifier and AggregPMeanError), the guard-weighted mean at
p = 1, that the guard receives no gradient, invariance to scaling the guard,
that unguarded individuals do not contribute, the empty-guard value, batched
quantification over one dimension, and that the value cannot be raised by
lowering the guard through gradient descent while it can under the
Reichenbach implication.
"""

import sys

import ltn
import torch

from soft_guarded_quantifier import AggregSoftGuardedPMeanError

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'OK  ' if condition else 'FAIL'} {name}{('  -- ' + detail) if detail else ''}")


torch.manual_seed(0)
N = 50
b = torch.rand(N)                        # consequent B(x)
a = torch.rand(N)                        # fuzzy guard A(x)

# ---------------------------------------------------- crisp guard = ltn guard --
print("\na crisp guard reproduces ltn's guarded quantification")
crisp = a > 0.5
x = ltn.Variable("x", torch.stack([b, a], dim=1))
B = ltn.Predicate(func=lambda v: v[:, 0])
for p in (1, 2, 5):
    forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=p), quantifier="f")
    ltn_val = forall(x, B(x), cond_vars=[x], cond_fn=lambda v: v.value[:, 1] > 0.5).value
    soft_val = AggregSoftGuardedPMeanError(p=p)(b, crisp.float())
    check(f"p={p}: equals ltn Forall with cond_fn",
          torch.allclose(ltn_val.float(), soft_val, atol=1e-6),
          f"ltn={ltn_val.item():.6f} soft={soft_val.item():.6f}")

# ------------------------------------------------------- p = 1 weighted mean ---
print("\np = 1 is the guard-weighted mean")
val = AggregSoftGuardedPMeanError(p=1, stable=False)(b, a)
ref = (a * b).sum() / a.sum()
check("equals sum A*B / sum A", torch.allclose(val, ref, atol=1e-6),
      f"{val.item():.6f} vs {ref.item():.6f}")

# ------------------------------------------------------------ guard gradient --
print("\nthe guard receives no gradient")
a_req = a.clone().requires_grad_(True)
b_req = b.clone().requires_grad_(True)
AggregSoftGuardedPMeanError(p=2)(b_req, a_req).backward()
check("no gradient reaches the guard", a_req.grad is None)
check("gradient reaches the consequent", b_req.grad is not None and b_req.grad.abs().sum() > 0)
b_req = b.clone().requires_grad_(True)
AggregSoftGuardedPMeanError(p=1, stable=False)(b_req, a).backward()
check("p = 1: consequent gradient is A(x) / sum A", torch.allclose(b_req.grad, a / a.sum(), atol=1e-7))

# --------------------------------------------------------- scale / zero guard --
print("\nonly the relative guard values matter")
op = AggregSoftGuardedPMeanError(p=2)
base = op(b, a)
check("scaling the guard leaves the value unchanged",
      all(torch.allclose(op(b, c * a), base, atol=1e-6) for c in (0.01, 0.5, 3.0)))
extra_b = torch.cat([b, torch.zeros(10)])
extra_a = torch.cat([a, torch.zeros(10)])
check("individuals with guard 0 do not contribute", torch.allclose(op(extra_b, extra_a), base, atol=1e-6),
      f"{op(extra_b, extra_a).item():.6f} vs {base.item():.6f}")

# ------------------------------------------------------------------ empty guard --
print("\nan empty guard gives 1 with a finite gradient")
b_req = b.clone().requires_grad_(True)
val = op(b_req, torch.zeros(N))
val.backward()
check("value is 1", val.item() == 1.0)
check("gradient is finite and zero", torch.isfinite(b_req.grad).all().item()
      and b_req.grad.abs().sum().item() == 0.0)

# ----------------------------------------------------------- batched quantifier --
print("\nquantifying over one dimension of a batch")
bb, aa = torch.rand(4, 7), torch.rand(4, 7)
aa[2] = 0.0
rows = op(bb, aa, dim=1)
check("one value per row, each equal to the row on its own",
      rows.shape == (4,) and all(torch.allclose(rows[i], op(bb[i], aa[i]), atol=1e-6) for i in range(4)))
check("an empty row gives 1", rows[2].item() == 1.0)

# -------------------------------------------------------------------- large p --
print("\nlarge p approaches the minimum over the guarded individuals")
val = AggregSoftGuardedPMeanError(p=200, stable=False)(b, a)
check("close to min of B where A > 0", abs(val.item() - b[a > 0].min().item()) < 0.05,
      f"{val.item():.4f} vs min {b[a > 0].min().item():.4f}")

# --------------------------------------------------------- vacuous satisfaction --
print("\nlowering the guard does not pay off")
# One individual satisfies B, one violates it. delta lowers the guard of the
# violating individual. Under the Reichenbach implication 1 - A + A*B gradient
# descent lowers that guard (vacuous satisfaction); the soft guarded quantifier
# gives it no gradient.
b2 = torch.tensor([0.95, 0.05], requires_grad=True)
for name, fn in (
        ("soft guarded", lambda g: AggregSoftGuardedPMeanError(p=2)(b2, g)),
        ("Reichenbach", lambda g: ltn.fuzzy_ops.AggregPMeanError(p=2)(1 - g + g * b2, dim=0))):
    delta = torch.zeros((), requires_grad=True)
    guard = torch.stack([torch.tensor(0.9), 0.9 * torch.exp(delta)])
    (1 - fn(guard)).backward()
    grad = 0.0 if delta.grad is None else delta.grad.item()
    if name == "soft guarded":
        check("soft guarded: no gradient toward lowering the guard", grad == 0.0, f"d loss/d delta = {grad:+.4f}")
    else:
        check("Reichenbach: gradient lowers the guard", grad > 0.0, f"d loss/d delta = {grad:+.4f}")

# ------------------------------------------------------------------ summary --
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for name in FAIL:
        print(f"  FAILED: {name}")
    sys.exit(1)
print("ALL TESTS PASSED")
