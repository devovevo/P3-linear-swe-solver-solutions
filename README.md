# P3 Overview - (Linearized) Shallow Water Simulation

Welcome to P3 everyone! We hope that you enjoyed simulating particle hydrodynamics, which was a problem that introduced you to less than uniform computation. What we mean by this is that particles were not evenly divided in space, so simply looping through all cells would often lead to wasted work (as most cells were empty). In this assignment, you will work with a structured grid computation, where every cell will be (roughly) the same, so you can't be too clever in how you divide the work.

# What are the Linearized Shallow Water Equations (SWE)?

The original shallow water equations, in non-conservative form, are given by
$$
\begin{align*}
\frac{\partial h}{\partial t} + \frac{\partial}{\partial x} \left((H + u) u\right) + \frac{\partial}{\partial y} \left((H + u) v\right) = 0, \\
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -g \frac{\partial h}{\partial x} - ku + \nu \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right), \\
\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = -g \frac{\partial h}{\partial y} - kv + \nu \left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right)
\end{align*},
$$
where $(u, v)$ describes the velocity field of a body of water, $H$ is the mean depth of the water (i.e. the average) at a particular point, $h$ is how much the height deviates from the mean at a given time, $k$ describes the viscous drag (basically a friction force), and $\nu$ is the viscosity (how much a liquid wants to flow away from itself). In some other sources you might also see a term dealing with $f$, the Coriolis parameter, which takes into account the rotation of the Earth. However, since we assume we are on a scale much smaller than this, we ignore this term. As you can see, these equations are quite the doozy, though if you'd like to know more please see the [Wikipedia](https://en.wikipedia.org/wiki/Shallow_water_equations) or ask Professor Bindel (he knows way more than us TAs!).

Since this is so complicated, and I have no background in PDEs, I decided to use the simpler, linearized shallow water equations. Essentially, if we assume that our height deviation is small relative to the average height (i.e. $h \ll H$) and that the higher order terms of velocity are quite small, through some derivations we have
$$
\begin{align*}
\frac{\partial h}{\partial t} + H \left(\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right) = 0, \\
\frac{\partial u}{\partial t} = -g \frac{\partial h}{\partial x} - ku, \\
\frac{\partial v}{\partial t} = -g \frac{\partial h}{\partial y} - kv
\end{align*}
$$
Since we want our equations to be energy conserving (as a check of our correctness), we assume that $k = 0$, so we don't dissipate energy due to friction. Now, we have a much simpler looking set of equations, though again, how do we solve this? If you already know about PDEs you can skip this section, but if not (like me), then please read on!

# Discretizing the Problem

In order to actually solve this problem numerically, we need to be able to compute derivatives with respect to space as well as time. Looking at the definition of a derivative
$$
\begin{align*}
f'(x) = \lim_{h \rightarrow 0} \frac{f(x + h) - f(x)}{h}
\end{align*}
$$
we see that it is the limit of a difference quotient. Given that, we could imagine that for a fixed (but small) $h$, the corresponding finite difference quotient should be a good approximation. Therefore, given a rectangular domain $R \subseteq \mathbb{R}^2$, we can discretize it into evenly spaced partitions $(x_1, \dots, x_n)$ and $(y_1, \dots, y_n)$, and then take the difference quotient of adjacent points to approximate spatial derivatives. In other words, we would say that the spatial derivative in the $x$ direction at $(x_i, y_j)$ is given by
$$
\begin{align*}
\hat{\frac{\partial f}{\partial x}}(x_i, y_j) = \frac{f(x_{i + 1}, y_j) - f(x_i, y_j)}{x_{i + 1} - x_i} = \frac{f(x_{i + 1}, y_j) - f(x_i, y_j)}{\Delta x}
\end{align*}
$$
with a similar formula for $y_i$. Using this idea, we can approximate all of the spatial derivatives for our functions $h$, $u$, and $v$. To be particularly clever, since our $u$ and $v$ functions govern the horizontal and vertical velocities of our field, we can imagine that they exist on the boundaries of our cells. This is called an Arakawa C grid, and an image is shown below ([source](https://www.researchgate.net/figure/The-Arakawa-C-grid-layout-of-the-variables-in-our-numerical-scheme-The-domain-is-divided_fig4_267118062)):
![Arakawa C Grid](image.png)
As you can see, we take the $u$ values on the horizontal edges of our cells, and the $v$ values on the vertical edges of our cells. The $h$ values, on the other hand, are in the center of our cells. I did this because it was popular for other SWE solvers, though I don't know if there's a particular numerical reason why it is used. This becomes a little difficult because if we have $n$ points in our horizontal partition for the $h$ function, we will have $n + 1$ points in our partition for the $u$ function (since we have the last edge as well). Therefore, we need to rely on boundary conditions to tell us what happens there.

# 