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

