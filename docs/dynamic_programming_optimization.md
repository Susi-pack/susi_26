
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
is the set of non-dominated objective vectors using groups $(1,\dots,i)$.

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
q_k \le (1+\epsilon) p_k \quad \forall k
$$
Note that in the case $\epsilon=0$ the exact domination criterion is recovered.


This has a nice geometrical interpretation.
For each point $p$ in target space, we draw a box of size $\epsilon$ times the value of the coordinate of $p$[^1].
If the point $q$ is inside that box, it never dominates $p$.
That is: instead of using perfect resolution (exact dominance) of the target space, we are voluntarily creating a coarser resolution so that we do not have to store so many points in the partial Pareto fronts.
The thickness  of the "pixels" is governed by $\epsilon$.

[^1]: This fact, by the way, is useful because it automatically accounts for different scales in the variables. If the box size would be just $\epsilon$ and the dominance criterion $q_k \le p_k + \epsilon$, we would run into trouble with variables of different scales. One solution would be to create one $\epsilon$ per dimension $\epsilon_k$, but the chosen approach leads to simpler interface: just choose a single value to tune the algorithm's behaviour.

Still, at the end of the day, we hope to retain a good approximation of the Pareto front.
And we can always get a finer resolution by choosing a smaller $\epsilon$.
This allows controlled approximation of the Pareto frontier.

## The algorithm
We already know everything we need to understand the implementation in `analysis/optimization/dynamic_programming.py`. The algorithm entry point is `find_pareto_front()`, and the structure is something like this (pseudocode):


```
pareto_front = partial_pareto_front(of stand #1)

for i in 1..n:
    pareto_front_new = []
    
    for p in pareto_front:
        for j in 1..m_i:
            P_new.append(p + a[i][j])
    
    pareto_front = pareto_epsilon_prune(pareto_front_new)
```

The meat of the algorithm is swept under the rug of `pareto_epsilon_prune()`. But that is a really simple function:

```python
def pareto_epsilon_prune(
    points: list[PartialParetoPoint], epsilon: float
) -> tuple[PartialParetoPoint, ...]:
    compressed_points = compress_into_buckets(points=points, epsilon=epsilon)
    return pareto_prune(compressed_points)
```

The misterious-looking `compress_into_buckets()`, does the following.
It discretizes objective space into "buckets" or "pixels" of size controlled by $\epsilon$.
And then maps each input point to one of those buckets.
This is doing the $\epsilon$ part of the algorithm.

The output of that is passed to `pareto_prune()`, which implements a normal Pareto pruning algorithm.

This is the whole algorithm.

## A few more implementation details

### `PartialParetoPoint` data structure
The algorithm as we have described only stores the solution in target space.
If we want to retain the desing vectors that give rise to that vector, we need to explicitly do so.
We could naively store all intermediate partial Pareto vectors, but that is a huge waste of space, since the vector at step $i$ contains the vector at step $i-1$.
That is why we created a tree structure in `PartialParetoPoint`.
Each point has a `parent_point` (a pointer to the `PartialParetoPoint` from which it originated) and the `current_scenario_choice`.
With that we can reconstruct the tree all the way back to recover the design vector of any `PartialParetoPoint`.

### Shift to strictly positive targets
The putting in to buckets function has a $\log$ in it.
In order for this to work, we need that all the target variables have positive values.
Since the optimization problem is invariant to translations, we first shift the values to be all positive, and then run the optimization.
To recover the original values back, we need to shift them back.

