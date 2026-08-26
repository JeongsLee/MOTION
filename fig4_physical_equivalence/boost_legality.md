Galilean boost legality per family (why each cell is measured or blank)
  legal   : periodic domain + advective dynamics -> u'(x,t)=u(x+Vt,t)+V is an exact solution
  illegal : walls/inflow fix the frame, or the equation is not Galilean invariant

6-family (panel b)
  SWE            legal    periodic, shallow-water advection is Galilean invariant
  Com. NS        legal    periodic compressible NS
  Incom. NS      legal    periodic incompressible NS
  PDEarena       ILLEGAL  Dirichlet walls + spatially fixed forcing (confirmed by the
                          translation probe: 9.7-24% defect where translation is also illegal)
  CFDbench       ILLEGAL  inflow/outflow + no-slip walls fix the laboratory frame
  PDEarena unc   ILLEGAL  same as PDEarena

IVP transfer (panel c)
  NS-PwC         legal    periodic incompressible NS  <- the one head-to-head cell
  ACE            ILLEGAL  reaction-diffusion has no advection; u(x-Vt,t) is not a solution
  Poisson-Gauss  ILLEGAL  steady problem, no time axis
  Wave           ILLEGAL  the wave equation is not Galilean invariant (speed is frame-dependent)
