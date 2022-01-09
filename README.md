# tinyoptimizer
A Non-Linear Optimizer in 500 lines of code (check out the [wiki](https://github.com/tomstewart89/tinyoptmizer/wiki) to see how!).

![optimisation](https://user-images.githubusercontent.com/2457362/147831250-5a9f0d96-e6e3-4044-9ae7-01e5d498fcde.gif)

This project was written in the spirit of [tinyrenderer](https://github.com/ssloy/tinyrenderer); its main purpose was to teach myself (and maybe you too) how non-linear solvers like [Ipopt](https://github.com/coin-or/Ipopt) or [NLopt](https://github.com/stevengj/nlopt) work, by writing a really simple clone, completely from scratch.

I've added a bunch of notes to the [wiki](https://github.com/tomstewart89/tinyoptmizer/wiki) to explain how I wrote this solver so if you want to write one too, then be sure to check that out. If you just randomly stumbled on this repo and don't know what non-linear optimization is, then read on!

## What is Non-Linear Optimization?

Non-Linear Optimizer aim to solve problems that look like this:

![nlp](https://user-images.githubusercontent.com/2457362/147829074-374342f3-dc27-4e1e-ab12-eef66abc5767.png)

In other words, find the `x` that returns the lowest value from `f_0` while also making sure that the output of `p` equality constraints is zero and the output of `m` inequality constraints is less than zero.

Here's an example:

```
minimise (x1 - 2) ** 2 + (x2 - 2) ** 2

subject to : x1 - 1 <= 0
             cos(pi / 5) x1 + sin(pi / 5) x2 - 1 <= 0
             cos(2 * pi / 5) x1 + sin(2 * pi / 5) x2 - 1 <= 0
             cos(3 * pi / 5) x1 + sin(3 * pi / 5) x2 - 1 <= 0
             cos(4 * pi / 5) x1 + sin(4 * pi / 5) x2 - 1 <= 0
```

It might be obvious that the cost function above will be minimised when `x1 == 2` and `x2 == 2` but due to those pesky inequality constraints that point is infeasible (i.e out of bounds). Instead we'll have to settle for some point hard up against the constraints that is nearest `(2,2)` which when we run the solver turns out to be around `(0.62, 0.85)` (you can see this play out in the gif above)

