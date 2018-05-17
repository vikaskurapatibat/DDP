# DDP

This repository is for implementation of MultiPhase SPH schemes concentrating on the formulation of Surface Tension in SPH.

1) Unable to reproduce the wrong results with Morris setup even with gamma 1 and 7 and no background pressure. 

2) Adami gives wrong results with gamma 1 and 7 and no background pressure. (check)

3) Gives the right result for Adami with gamma 1 and background pressure. 
4) Wrong result with adami's case and no background pressure with tensile correction from PySPH. 
5) Adami case2 gives wrong results even with reduced time step and hence time step is not the issue.
6) Adami implementation in PySPH is wrong. Needs a fix.


27-04-2018:
1) Checked all the equations for anamolies in s_idx and d_idx. None noticed.
2) Adami scheme still mixes without background pressure.
3) Adami scheme works with background pressure.
4) Adami scheme works with momentum equation with tensile correction with background pressure.
5) Adami scheme doesn't work with momentum equation with tensile correction without background pressure.
6) Adami case 2 works now.


To do:

1) morris background pressure (1)
2) Post Processing (1)
3) PySPH cases fix (2)
4) Shadloo case fix (2)
5) Adami Cases: three phase interaction (3)
6) TVF + ST (4)
7) Test kernel gradient correction without background pressure
8) Pull Requests
9) Tartakovsky
10) Splishsplash schemes for Surface Tension.
11) 3D case from Adami's curvature paper. (4)


Finished:
1) Tensile correction - doesn't work
2) Morris cases fix
3) Clean up Code
4) Adami Cases : variable density


Skipped:
1) Other TIS
2) Generalized TVF

30-04-2018:
1) No background pressure mentioned in that paper.
2) Post Processing done. 
3) Adami Scheme in PySPH fixed. It's not right in both the papers, it's supposed to be divided by density. Tell sir the same. 
4) Shadloo Ydlis scheme has some issue. : The nx and ny seem okay but with some anomolies. Edit: They are right. There is some other issue in the problem. 



Ask sir about the rod case using Adami Divergence problem.

Shadloo case still has the issue after fixing the equations accordingly.


User options to be set in all the problems.


Things to ask sir: 

Reason why the square droplet isn't moving even when the accelerations are there. Solved, they were moving with low accelerations.

