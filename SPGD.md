Algorithm 1 PGD-Sparse Attack

Input:
    Image x
    Ground-truth label y
    Classifier f
    Saliency function G
    Step size α
    Number of iterations T
    Sparsity ratio p
    Threshold τ
    Weights λ, β

Output:
    Sparse perturbation δ_sparse

1:  Initialize dense perturbation δ ← 0
2:  Compute reference saliency S_clean ← G(x)

3:  for t = 1 ... T do

4:      δ_soft ← σ(δ)

5:      x_adv ← Clip(x + δ_soft)

6:      Compute prediction loss
            L_margin ← f(x_adv, ŷ) − max_{j≠ŷ} f(x_adv, j)

7:      Compute saliency map
            S_adv ← G(x_adv)

8:      Compute saliency loss
            L_sal ← SoftIOU(S_adv, S_clean)

10:     Compute total loss
            L ← −L_margin + λL_sal

11:     Compute gradient
            g ← ∇δL
        ▷ Backpropagate through G using create_graph=True
        ▷ (second-order gradients)

12:     Update perturbation
            δ ← δ + α sign(g)

13: end for

14: δ_soft ← σ(δ)

15: mask ← I(δ_soft > τ)

16: δ_sparse ← mask ⊙ δ

17: return δ_sparse