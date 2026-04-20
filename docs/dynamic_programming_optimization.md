
# $\epsilon$-Dominance Dynamic Programming for SUSI Multiple-Choice Pareto frontier estimation

## Problem motivation

Say we simulate several different scenarios for a stand.
The scenarios cover different management options, which might include different choices of ditch network maintenance, fertilization, logging, etc.
We are naturally interested in the following question: given a set of target variables $v$, what is the optimal management choice?

With a single stand, the answer to that question should be straightforward: set your target variables from the SUSI simulation output, then choose the management option that results in the best value of those variables.

However, the question gets surprisingly more difficult when we have $n$ different stands that we have to manage with a common set of targets.
If we are interested in, say, maximizing the amount of volume while minimizing the C emissions and nutrient export to water courses, how should we manage each of our $n$ stands? Should we lower the WT in the best-performing sites to enhance productivity there, while leaving the poorer sites to maximize C capture? Or vice-versa? Or should each stand strive to be individually balanced?

This might sound like something that a computer should be able to figure out.
Simply go over all the possible options, and then choose the ones that optimize the target variables.
It turns out that the number of possible combinations of stands $n$ and scenarios (a.k.a. management options) $m$ is mind-bogglingly large, even for computers.

Say that for each stand $i \in (1, ..., n)$ we have computed a varying number of scenarios, $m_i$.
Then the space of possibilities is purely combinatorial, with cardinality:

$$
\prod_{i=1}^n m_i.
$$

So, if we have $n=21$ stands with $m=5$ choices each, we are left with about $3^{21} = 4.7 \cdot 10^{14}$ options to choose from. And if we instead have $n=300$ sites, we end up with $5^{300} = 4.9 \cdot 10^{209}$ choices.
Exhaustive seach is obviously impossible.

## Additivity: the key simplification

If there was no more structure to the optimization problem, we would have no choice but to rely on heuristic algorithms such as genetic algorithms, simulated annealing, etc.
And in fact, we started by implementing one of those algorithms.

But there is an important property that renders it much more tractable: additivity.
I first introduce a bit more of notation.

A solution vector for the optimization problem sketched above has the following shape in design space:
  $$
  x = (x_1, \dots, x_n), \quad x_i \in {1,\dots,m_i}.
  $$

In plain language: we choose one management option for each stand, which is indexed by a number from $1$ to $m_i$, and a solution is a combination of those individual choices.

We have $v$ target variables.
So each combination of a stand (indexed by $i$) and a scenario for that stand (indexed by $x_i$) corresponds to a $v$ dimensional vector in target space:
$$
  a_{i,x_i} \in \mathbb{R}^v
$$

In our case, the $a_{i,x_i}$ are precomputed; they are the solutions of the Susi simulations for the different stands and management options.
So we do not need to compute the  $a_{i,x_i}$ for each $(i, x_i)$.
Instead, we simply query the value from the precomputed table.

Note also that our goal is to approximate the **Pareto frontier** in $\mathbb{R}^v$.
This is because we have $v$ (probably conflicting) variables, but we want to stay maximally agnostic about how to compare them before making a decision (how much "points" do we assign to wood volume compared to C balance?).

The key property is the following.
For each design-space vector $x$, the total optimization target depends on the $a_{i, x_i}$ only through their sum:
  $$
  F(x) = \sum_{i=1}^n a_{i,x_i}.
  $$
In other words, it is *additive*.

Here's the intuitive idea for how we can exploit this property to improve the optimization algorithm.
Say we have 3 management scenarios for each stand: A, B and C.
And let's say that, for stand 1, A and B have conflicting objectives, but that C is completely Pareto-dominated by A.
This means that for all objective variables, A is either better or equal than C.
We don't need to consider the option where stand 1 has the choice C, because we know we could improve it by simply switching to the option A, in all variables.
Note that this is only possible because there is no "interaction" between the stands, i.e., the total optimization objective decomposes as a sum over independent stands.

This implies, first of all, that we can rule out a handfull of possibilities before running any optimization algorithm, based simply on the values of the stands individually.
That is: we can look for the Pareto-dominated management choices for each stand individually, and drop them.

Second, and more importantly, we can exploit this property incrementally at each iteration of the $\epsilon$-dominance dynamic programming algorithm.
Here's how.

## Dynamic Programming: incremental Pareto front construction

We start the algorithm by considering the options for stand 1 only.
We define a partial Pareto front which stores all the non-dominated points that we gather along the way:

$$
P_i
$$
is the set of non-dominated objective vectors using groups (1,\dots,i).

So we initialize $P_i$ as $P_1$.

Now we consider stand 2.
For each of the points in $P_1$, we make all possible combinations using the values from stand 2.
This will result in several points, some of them Pareto-dominated by others.
We proceed to prune all those Pareto-dominated points.
Thus we have created $P_2$.

We repeat this process for all $n$ stands. At each step $i$:
1. We compute all the combinations from joining points in $P_{i-1}$ with the choices for stand $i$.
2. We prune the dominated ones. This leaves us with the partial Pareto frontier at step $i$, $P_i$.

The final result is $P_n$, which is no longer a partial Pareto frontier, but the total Pareto frontier that we were looking for.

## Still, a challenge: frontier explosion
Even if at each step we prune the frontier and considerably reduce the amount of possible options, we might end up with too many points in the partial Pareto frontier and an intractable problem.
This happens because our variable values, being floating point numbers, can be arbitrarily close to each other without being Pareto-dominated.

The solution is what *$\epsilon$-dominance* stands for in our algorithm's fancy name.
Instead of checking for exact Pareto-dominance between points, we ask for a stricter thing:
does the new point dominate the existing point even if we add $\epsilon$ to it?

That is: a vector $q$ $\epsilon$-dominates $p$ if:
$$
q_k \le (1+\varepsilon) p_k \quad \forall k
$$
Note that in the case $epsilon=0$ the exact domination criterion is recovered.

This has a nice geometrical interpretation.
For each point $p$ in target space, we draw a box of size $\epsilon$ times the value of the coordinate of $p$[^1].
If the point $q$ is inside that box, it never dominates $p$.
That is: instead of using perfect resolution (exact dominance) of the target space, we are voluntarily creating a coarser resolution so that we do not have to store so many points in the partial Pareto fronts.
The thickness  of the "pixels" is governed by $\epsilon$.

[^1]: This fact, by the way, is useful because it automatically accounts for different scales in the variables. If the box size would be just $\epsilon$ and the dominance criterion $q_k \le p_k + \epsilon$, we would run into trouble with variables of different scales. One solution would be to create one epsilon per dimension $\epsilon_k$, but the chosen approach leads to simpler interface: just choose a single value to tune the algorithm's behaviour.

Still, at the end of the day, we hope to retain a good approximation of the Pareto front.
And we can always get a finer resolution by choosing a smaller $\epsilon$.
This allows controlled approximation of the Pareto frontier.


CONTINUE HERE!

## 8. $\epsilon$-Pruning via Grid Discretization

To efficiently prune, we discretize objective space.

### Bucket Mapping

For each vector ( p ), define:
[
b_k = \left\lfloor \frac{\log(p_k)}{\log(1+\varepsilon)} \right\rfloor
]

* Each vector maps to a grid cell
* Only one representative is kept per cell

### Effect

* Reduces number of stored solutions
* Guarantees $\epsilon$-approximation of Pareto frontier

---

## 9. Preprocessing Optimization (Critical)

Before running the DP:

### Remove Dominated Choices Per Group

For each group ( i ), remove any ( a_{i,j} ) such that:
[
a_{i,j} \text{ is dominated by another } a_{i,j'}
]

### Impact

* Reduces ( m_i )
* Significantly lowers computational cost
* Improves pruning effectiveness downstream

---

## 10. Complexity Behavior

### Worst Case

* Still exponential (theoretical)

### Practical Case

* Controlled by $\epsilon$
* Typically:

  * Hundreds to a few thousand points in frontier
  * Runtime: seconds to minutes

---

## 11. Advantages of This Approach

### Deterministic

* Same input → same output

### Structure-Exploiting

* Fully leverages additivity and separability

### Scalable

* Handles large ( n ) effectively

### General-Purpose

* Works across all ( (n, m_i, v) ) without redesign

### Tunable

* Single parameter ( \varepsilon ) controls:

  * accuracy
  * runtime
  * memory

---

## 12. Choosing $\epsilon$

Typical values:

| $\epsilon$         | Effect                         |
| --------- | ------------------------------ |
| 0.01      | High accuracy, larger frontier |
| 0.03–0.05 | Good balance                   |
| 0.1       | Aggressive pruning, fast       |

---

## 13. Optional Enhancements

### Scalarization Sampling

* Solve weighted sums:
  [
  \min_x \sum_k w_k F_k(x)
  ]
* Very fast due to separability
* Helps fill convex regions of the frontier

### Hybrid Approach

* Combine scalarization + $\epsilon$-DP
* Improves coverage

---

## 14. Limitations

* Approximate (controlled by $\epsilon$)
* Performance depends on:

  * correlation between objectives
  * effectiveness of pruning

---

## 15. Summary

This approach transforms an intractable combinatorial problem into a **manageable incremental construction of an approximate Pareto frontier**.

The key enablers are:

* additive structure
* independence across groups
* $\epsilon$-dominance pruning

### Final Takeaway

> $\epsilon$-dominance dynamic programming provides a robust, scalable, and general solution for multi-objective multiple-choice optimization problems with additive structure.

---

## 16. Implementation Notes

* Use efficient data structures for pruning (hash maps for buckets)
* Normalize objectives if scales differ significantly
* Monitor frontier size to detect pathological cases
* Consider logging intermediate frontier sizes for diagnostics

---

End of document.
