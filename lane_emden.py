#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estructura y evolución estelar: Ecuación de Lane-Emden y modelos politrópicos
solares.

Created on Tue Nov 12 15:36:42 2024

@author: alejandro
"""

from scipy . integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from astropy.constants import M_sun, R_sun, G, R
from astropy import units as u
from scipy.interpolate import interp1d
import matplotlib.lines as mlines


# Enable LaTeX rendering
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')  # Use a serif font for LaTeX rendering
plt.rc('font', size=16)  # Adjust size to your preference
# Define the LaTeX preamble with siunitx
plt.rcParams['text.latex.preamble'] = r'''
            \usepackage{siunitx}
            \sisetup{
              detect-family,
              separate-uncertainty=true,
              output-decimal-marker={.},
              exponent-product=\cdot,
              inter-unit-product=\cdot,
            }
            \DeclareSIUnit{\cts}{cts}
            \DeclareSIUnit{\dyn}{dyn}
            \DeclareSIUnit{\mag}{mag}
            \usepackage{sansmath}  % Allows sans-serif in math mode
            \sansmath
            '''
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Computer Modern Serif",
})

xi_min = 0.001
xi_max = 200

theta_0_ini = 1 # Valor de theta en xi=0
theta_1_ini = 0 # Valor de theta' en xi=0

    
nn = np.linspace(0, 5, 11)


def lane_emden(xi,theta,n):
    return [theta[1],
            -theta[0]**n -(2/xi)*theta[1]]

def zero_reached (xi, theta, n) :
    return (theta[0])

zero_reached.terminal = True


def solve_LE():
    """
    Función que integra la eq. de L-E para distintos n y devuelve un plot

    Returns
    -------
    soluciones.

    """

    soluciones_xx = []
    soluciones_tt = []
    derivadas_tt = []
    
    for n in nn:
        sol = solve_ivp (lane_emden, [xi_min, xi_max], [theta_0_ini, theta_1_ini],
                              args =(n,), t_eval=np.linspace(xi_min, xi_max, int(1e6)), events=zero_reached,
                              rtol=1e-7, atol=1e-10)
    
        xx = sol.t
        tt = sol.y[0]
        der_tt = sol.y[1]
        
        soluciones_xx.append(xx)
        soluciones_tt.append(tt)
        derivadas_tt.append(der_tt)
    
    return np.array(soluciones_xx, dtype=object), np.array(soluciones_tt, dtype=object), np.array(derivadas_tt, dtype=object)

###############################################################################
######################## 1 - Plot de la solución ##############################
###############################################################################

xx, tt, dtdt = solve_LE()
fig1, ax1=plt.subplots(figsize=(17, 10))

cmap = cm.get_cmap("viridis", len(nn))
for i in range(len(xx)):
    ax1.plot(xx[i],tt[i], color=cmap(i))

ax1.set_xlabel(r"$\xi$", fontsize=30)
ax1.set_ylabel(r"$\theta$", fontsize=30)
ax1.tick_params(axis='both', which='major', labelsize=24)
ax1.set_xlim(0 , 15)
ax1.set_ylim(0, 1)

# Add color bar to indicate which color corresponds to which `n` value
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=nn[0], vmax=nn[-1]))
sm.set_array([])
cbar = fig1.colorbar(sm, ax=ax1, spacing='proportional', aspect=30)
cbar.set_label(r"Polytropic index ($n$)", rotation=270, labelpad=50, fontsize=30)
tick_locs = (max(nn)/len(nn))*(np.arange(len(nn))+0.5)
cbar.set_ticks(tick_locs)
cbar.set_ticklabels(nn, fontsize=24)
plt.show()

###############################################################################
################# 2 - Comparación con soluciones exactas ######################
###############################################################################

fig2, ax2=plt.subplots(figsize=(13, 8))

# Para n=0:
tt_n_0 = 1-1/6*xx[nn==0][0]**2

# Para n=1:
tt_n_1 = np.sin(xx[nn==1][0])/xx[nn==1][0]  

# Para n=5:
tt_n_5 = (1+xx[nn==5][0]**2/3)**(-0.5)  

plt.plot(xx[nn==0][0], tt[nn==0][0],"orange", linewidth=5, alpha=0.5, label=r"Numerical solution")
plt.plot(xx[nn==0][0], tt_n_0, "black", linestyle=(0, (5, 7)), linewidth=3, label=r"Analytical solution")
ax2.text(2.43, 0.068, r"$n = 0$", fontsize=24, color="black", ha='left', va='top')

plt.plot(xx[nn==1][0], tt[nn==1][0],"orange", linewidth=5, alpha=0.5)
plt.plot(xx[nn==1][0], tt_n_1, "black", linestyle=(0, (5, 7)), linewidth=3)
ax2.text(2.75, 0.19 , r"$n = 1$", fontsize=24, color="black", ha='left', va='top')

plt.plot(xx[nn==5][0], tt[nn==5][0],"orange", linewidth=5, alpha=0.5)
plt.plot(xx[nn==5][0], tt_n_5, "black", linestyle=(0, (5, 7)), linewidth=3)
ax2.text(3.39, 0.50 , r"$n = 5$", fontsize=24, color="black", ha='left', va='top')

ax2.set_xlabel(r"$\xi$", fontsize=30)
ax2.set_ylabel(r"$\theta$", fontsize=30)
ax2.tick_params(axis='both', which='major', labelsize=24)
ax2.set_xlim(0 , 5)
ax2.set_ylim(0, 1)
ax2.legend(loc='best', frameon=False, fontsize=26)
plt.show()
            
    
###############################################################################
# 3 - Cálculo de Dn, Mn, Rn, Bn

t_1 = np.array([t[-1] for t in tt]) # Últimos valores de theta en cada n
xi_1 = np.array([xi[-1] for xi in xx]) # Valores de xi que anulan theta en cada n --> R_n
dt_dxi_1 = np.array([dt[-1] for dt in dtdt]) # Valores de la derivada de theta en xi_1

#for i in range(len(nn)):
    #print(f"n={nn[i]}")
    #print(f"theta[-1]={t_1[i]}")
    #print(f"xi_1={xi_1[i]}")
    #print()
    
M_n = -xi_1**2*dt_dxi_1
B_n = 1/(nn+1)*(-xi_1**2*dt_dxi_1)**(-2/3)
D_n = 1/(-3/xi_1*dt_dxi_1)

print(f"{'n':^5} | {'M_n':^10} | {'D_n':^10} | {'R_n':^10} | {'B_n':^10}")
print("-" * 49)
for i in range(len(nn)):
    print(f"{nn[i]:^5} | {M_n[i]:^10.4f} | {D_n[i]:^10.4f} | {xi_1[i]:^10.4f} | {B_n[i]:^10.4f}")


###############################################################################
# 3 - Densidad, masa, presión, temperatura para n=3

# Modelo solar complejo:
standard = np.loadtxt('A300_solar_model.dat', skiprows=2)
R_standard = standard[:,1]
M_standard = standard[:,0]
rho_standard = standard[:,3]
P_standard = standard[:,4]
T_standard = standard[:,2]

fig, (ax3, ax4, ax5, ax6) = plt.subplots(4, 1, figsize=(8.27, 11.69), sharex=True)

# Densidad: 
# Convertimos xi a coordenada radial r/R:
xi=xx[nn==3][0]
r_R = xi/xi_1[nn==3][0]
#Calculamos la densidad a partir de theta:
average_density_sun = M_sun / ((4/3) * np.pi * R_sun**3)
average_density_sun_in_cgs = average_density_sun.to(u.g / u.cm**3).value
central_density = D_n[nn==3]*average_density_sun
density=central_density*tt[nn==3][0]**3

ax3.plot(r_R, np.log10([d.to(u.g / u.cm**3).value for d in density]), color='darkviolet', label=r"Solar polytrope, $n=3$")
ax3.plot(R_standard, np.log10(rho_standard), color='limegreen', label=r"Standard Solar Model", linestyle=(0, (5, 1)))
ax3.set_ylim(-4, 3)
ax3.set_xlim(0, 1)
ax3.set_ylabel(r"$\log_{10}(\rho\, [\unit{\gram\per\centi\meter\cubed}])$ ", fontsize=14)
ax3.legend(loc='best', frameon=False)

# Masa relativa:
M = -xi**2*dtdt[nn==3][0]/M_n[nn==3][0]
ax4.plot(r_R, M, color='darkviolet')
ax4.plot(R_standard, M_standard, color='limegreen', linestyle=(0, (5, 1)))
ax4.set_ylabel(r"$M/M_\odot$", fontsize=14)

# Presión:
# Calculamos la constante K:
K = (M_sun/M_n[nn==3][0]/4/np.pi)**(2/3)*G*np.pi 
# Calculamos la presión:
P = K*density**(4/3)
ax5.plot(r_R, np.log10([p.to(u.dyn / u.cm**2).value for p in P]), color='darkviolet')
ax5.plot(R_standard, np.log10(P_standard), color='limegreen', linestyle=(0, (5, 1)))
ax5.set_ylim(10, 18)
ax5.set_ylabel(r"$\log_{10}(P\, [\unit{\dyn\per\centi\meter\squared}])$ ", fontsize=14)

# Temperatura:
    
# Resolvemos la ecuación cuártica de Eddington:
mu=0.61
roots = np.roots([mu**4*0.003,0,0,1,-1])
beta = float([sol for sol in roots if sol.imag==0 and sol>0][0])

T = P*beta*mu/(R/(1*u.g/u.mol)*density)
ax6.plot(r_R, np.log10([t.to(u.K).value for t in T]), color='darkviolet')
ax6.plot(R_standard, np.log10(T_standard), color='limegreen', linestyle=(0, (5, 1)))
ax6.set_ylim(4, 8)
ax6.set_ylabel(r"$\log_{10}(T\, [\unit{\kelvin}])$", fontsize=14)
ax6.set_xlabel(r"$r/R$", fontsize=14)


# Adjust layout to prevent overlap
fig.tight_layout()
plt.show()

###############################################################################
# 4 - Densidad para n=2.5, n=3, n=3.5

# Cálculo de Chi cuadrado para evaluar los modelos politrópicos.
def chi_squared(polytropic_model:tuple, good_model:tuple):
    """
    Esta función calcula el chi cuadrado reducido de un modelo politrópico
    comparándolo con el Standard Solar Model

    Parameters
    ----------
    polytropic_model : tuple
        (R, rho) para el modelo politrópico.
    good_model : tuple
        (R, rho) para el Standard Solar Model.

    Returns
    -------
    None.

    """
    R_poly = polytropic_model[0]
    rho_poly = polytropic_model[1]
    R_good = good_model[0]
    rho_good = good_model[1]
    
    # Interpolate the good model at the x-values of the polytrope model
    interp_func = interp1d(R_good, rho_good, kind='linear', fill_value="extrapolate")
    rho_good_interp = interp_func(R_poly)
    sigma_good = 1e-3*rho_good_interp # Incertidumbre estimada en los valores del Standard Solar Model
    
    # Compute the chi-squared
    chi_squared = np.sum(((rho_poly - rho_good_interp) / sigma_good) ** 2)
    
    # Degrees of freedom
    N = len(rho_poly)  # Number of data points in the polytrope model
    reduced_chi_squared = chi_squared / N
    return reduced_chi_squared

reduced_chi_squared = [] # Lista donde guardaremos los valores de Chi² reducido
reduced_chi_squared_02_08 = [] # Lista donde guardaremos los valores de Chi² reducido para valores
                               # de R entre 0.2 y 0.8

fig, ax=plt.subplots(figsize=(15, 10))
# Standard Solar Model:
ax.plot(R_standard, np.log10(rho_standard), color='red' , linestyle=(0, (5, 1)), linewidth=5)

for i, n in enumerate(nn):
    xi_n=xx[nn==n][0]
    r_R_n = xi_n/xi_1[nn==n][0]
    central_density_n = D_n[nn==n]*average_density_sun
    density_n=central_density_n*tt[nn==n][0]**3
    ax.plot(r_R_n, np.log10([d.to(u.g / u.cm**3).value for d in density_n]), color=cmap(i), alpha=0.8)
    reduced_chi_squared.append(chi_squared(((r_R_n, [d.to(u.g / u.cm**3).value for d in density_n])), 
                                           (R_standard, rho_standard)))
    
    interior_mask = (r_R_n<0.8) & (r_R_n>0.2)
    reduced_chi_squared_02_08.append(chi_squared(((r_R_n[interior_mask], np.array([d.to(u.g / u.cm**3).value for d in density_n])[interior_mask])), 
                                           (R_standard, rho_standard)))

# Add color bar to indicate which color corresponds to which `n` value
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=nn[0], vmax=nn[-1]))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, spacing='proportional', aspect=30)
cbar.set_label(r"Polytropic index ($n$)", rotation=270, labelpad=50, fontsize=30)
tick_locs = (max(nn)/len(nn))*(np.arange(len(nn))+0.5)
cbar.set_ticks(tick_locs)
cbar.set_ticklabels(nn, fontsize=24)
ax.set_ylim(-3,3)
ax.set_xlim(0,1)
ax.set_xlabel(r"$r/R$", fontsize=30)
ax.set_ylabel(r"$\log_{10}(\rho\, [\unit{\gram\per\centi\meter\cubed}])$ ", fontsize=30)
ax.tick_params(axis='both', which='major', labelsize=24)

# Custom legend handles
solar_handle = mlines.Line2D([], [], color="red", linestyle=(0, (5, 1)), label=r"Standard Solar Model",linewidth=5)
poly_handle = mlines.Line2D([], [], color="black", linestyle="-", label="Polytropes")

# Add legend with only these two handles
ax.legend(handles=[solar_handle, poly_handle], fontsize=26,loc='best', frameon=False)

plt.show()

# Print header
print(f"{'n':<8}{'Reduced Chi^2':<20}{'Reduced Chi^2 (0.2-0.8)':<20}")
print("-" * 40)

# Print rows
for n, chi, chi_0208 in zip(nn, reduced_chi_squared, reduced_chi_squared_02_08):
    print(f"{n:<8.1f}{chi:<20.3f}{chi_0208:<20.3f}")

