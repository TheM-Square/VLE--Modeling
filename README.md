# Vapor-Liquid Equilibrium for Binary Mixtures

**Models:** Antoine Equation, Raoult's Law, Margules Activity Coefficients  
**Stack:** Python, NumPy, Matplotlib, SciPy

---

## What this does

Computes VLE phase diagrams for two systems:

| System | Behavior |
|---|---|
| Benzene–Toluene | Nearly ideal (similar molecules) |
| Ethanol–Water | Non-ideal, strong positive deviations |

For each system it draws T-xy, x-y diagrams and compares ideal vs non-ideal predictions.

---

## How to run

```bash
pip install numpy matplotlib scipy
python vle_modeling.py
```

---

## Models used

**Antoine equation** — gives vapor pressure at any temperature:
```
log10(Psat) = A - B / (C + T)
```

**Raoult's Law (ideal):**
```
P * y_i = x_i * Psat_i(T)
```

**Modified Raoult's Law (non-ideal):**
```
P * y_i = x_i * gamma_i * Psat_i(T)
```

**Margules model** — calculates activity coefficients γ:
```
ln(g1) = x2² * [A12 + 2*(A21-A12)*x1]
ln(g2) = x1² * [A21 + 2*(A12-A21)*x2]
```

---

## Bubble point calculation

Given liquid composition `x1`, find temperature `T` such that:
```
x1*g1*Psat1(T) + x2*g2*Psat2(T) = P
```
Solved using `scipy.optimize.brentq` (root-finding).

---

## Output

`vle_results.png` — 4 plots:
(/vle_plots.png)
1. T-xy diagram: Benzene–Toluene
2. T-xy diagram: Ethanol–Water
3. x-y equilibrium diagram (both systems)
4. Antoine vapor pressure curves

---

## Why ethanol-water looks different

Ethanol and water have very different polarities. Water-water interactions are stronger than ethanol-water → the mixture has higher vapor pressure than Raoult predicts → positive deviations → γ > 1. This is why you can't distill pure ethanol past 96 wt% (azeotrope).
