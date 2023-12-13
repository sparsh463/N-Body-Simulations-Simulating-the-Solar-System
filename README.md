# PIC16B_Project
A Repository to house all the Project Files for PIC 16B


## I: Project Overview
For this project, we created and optimized a simulation and accompanying visualization of gravitational dynamics in Earth’s solar system. Beginning with a script for computing the changing acceleration of bodies governed by the Newtonian laws of gravitation: 

$$
\vec{F}_{21} = -G \frac{m_1 m_2}{|\vec{r}_{21}|^2} \hat{r}_{21}
$$

where

- $\vec{F}_{21}$ is the force applied on object 2 exerted by object 1,
- $G$ is the gravitational constant,
- $m_1$ and $m_2$ are respectively the masses of objects 1 and 2,
- $|\vec{r}_{21}| = |\vec{r}_2 - \vec{r}_1|$ is the distance between objects 1 and 2, and
- $\hat{r}_{21} = \frac{\vec{r}_2 - \vec{r}_1}{|\vec{r}_2 - \vec{r}_1|}$ is the unit vector from object 1 to object 2.

We used astropy to create a 
simulation of the celestial bodies in the solar system, generating continuous position, force, and acceleration arrays. With a given simulation run time, the arrays were visualized using Plotly to generate an interactive animation of the orbits and changing parameters of the bodies throughout a run.

A major component of our project was focused on optimization techniques and profiling for the overall project. We ran cProfile and line-by-line assessments of the program, and were able to find bottlenecks, such as the matrix multiplication of the changing accelerations, and the visualization with Plotly. Using techniques from the course, we used numba to optimize the calculations within an explicit for-loop, and use just-in-time compilation to significantly reduce the run time for the simulation.

There are many possible future directions for our product - one line of work can be focused on improving accessibility, through hosting and adding GUI features to the simulation and visualization. Additionally, testing the program on more complex celestial systems and verifying with empirical orbit data could be promising to further refine the accuracy of the simulation.
