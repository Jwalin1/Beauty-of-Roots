import numpy as np
import mpmath as mpm
import matplotlib.pyplot as plt

from . import poly_utils
from . import pisot_search

tau = 2 * np.pi
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_approximation_error_1(num, n_pows, axs=None):
  axs = plt.gca() or axs
  vals = num ** np.arange(1,n_pows)
  plt.plot(range(1,1+len(vals)), np.array(list(map(mpm.nint,vals))) - vals, c='b', marker='.', mfc='r', mec='y')
  plt.xlabel('i')
  plt.ylabel('$x^i - [x^i]$')
  plt.axhline(0, ls='--', lw=1)
  disp_digits = min(20, len(str(num)))
  axs.text(0.5, 1.05, f'x={float(num):.{disp_digits}f}...', transform=axs.transAxes, ha='center', size=12)

def plot_approximation_error_2(num, n_pows):
  axs = plt.gca() or axs
  vals = num ** np.arange(1,n_pows)
  plt.plot(range(1,1+len(vals)), abs(np.array(list(map(mpm.nint,vals))) - vals), c='b', marker='.', mfc='r', mec='y')
  plt.xlabel('i')
  plt.ylabel('$|x^i - [x^i]|$')
  plt.axhline(0, ls='--', lw=1)
  disp_digits = min(20, len(str(num)))
  axs.text(0.5, 1.05, f'x={float(num):.{disp_digits}f}...', transform=axs.transAxes, ha='center', size=12)


def get_poly_str(polynomial, coeff_digits=3):
  eqn = ''
  degree = len(polynomial) - 1
  for j, coeff in enumerate(polynomial):
    if coeff == 0:
      continue
    sign = '-' if coeff < 0 else ('+' if j>0 else '')
    if (coeff != -1 and coeff != 1):
      coeff = f'{abs(coeff):.{coeff_digits}f}' if coeff != int(coeff) else abs(coeff)
    else:
      coeff = ''
    pow = f'$^{{{degree - j}}}$' if degree-j > 1 else ''
    var = 'x' if degree-j > 0 else '1'
    eqn += f' {sign} {coeff}{var}{pow} '
  return eqn

theta = np.linspace(0, tau, 100)
def plot_roots(roots):
  roots_np = np.array(roots, dtype=complex)
  plt.scatter(roots_np.real, roots_np.imag, c='r')
  for root in roots_np:
    plt.plot((0,root.real), (0, root.imag), c='b')
  plt.scatter(0, 0, zorder=2, c='w')
  circle = np.exp(1j * theta)
  plt.plot(circle.real, circle.imag, ls='--', c=colors[0])

n_pows   = [40, 100, 90, 30, 150, 100, 200, 200, 60, 250, 13, 525, 800]
n_digits = [ 8,  17, 16,  7,  29,  20,  42,  42, 14,  52,  5, 120, 250]

def plot_poly_convergence(coeffs, n_digits, axs, convergence_params={}):
  mpm.mp.dps = n_digits
  roots = mpm.polyroots(coeffs)
  plt.sca(axs[0])
  eqn = get_poly_str(coeffs)
  # plt.title(f'roots of {eqn}')
  fs = 12 if len(eqn) <= 84 else 12 * (84 / len(eqn))
  axs[0].text(0.5, 1.05, f'roots of {eqn}', transform=axs[0].transAxes, ha='center', fontsize=fs)
  plot_roots(roots)

  max_root = poly_utils.get_max_root(coeffs, n_digits)
  n_pows = poly_utils.check_convergence(max_root, **convergence_params)
  plt.sca(axs[1])
  disp_digits = min(15, n_digits)
  # plt.title(f'x={float(max_root):.{disp_digits}f}...')
  plot_approximation_error_1(max_root, n_pows)
  plt.sca(axs[2])
  # plt.title(f'x={float(max_root):.{disp_digits}f}...')
  plot_approximation_error_2(max_root, n_pows)


def plot_nint_errs(x, y, n_pows, dips=None):
  plt.plot(x, y, c='b')
  plt.xlabel('x')
  # Normalized weighted error sum.
  ylabel_text = f'$ \\left( \\frac{{8}}{{{n_pows} \cdot {n_pows+1}}} \\right)  \sum_{{i=1}}^{{{n_pows}}} i \cdot (x^i - [x^i])^2 $'
  # substitute for plt.ylabel()
  plt.text(-0.15, 0.5, ylabel_text, transform=plt.gca().transAxes, va='center', rotation=90)
  plt.text(0.5, 1.05, f'n_pows = {n_pows}', transform=plt.gca().transAxes, ha='center')  # substitute for plt.title()
  if dips is not None:
    plt.scatter(x[dips], y[dips], c='r', s=15)




def plot_roots_spirals(paired_single_roots, pows):
  fig, axs = plt.subplots(len(paired_single_roots), 4, figsize=(24, 6*len(paired_single_roots)))
  if axs.ndim == 1:
    axs = np.expand_dims(axs, 0)
  for i, root in enumerate(paired_single_roots):
    modulus = f'r = {abs(root):.5f}'
    argument = f'$\\theta = {np.arctan2(root.imag, root.real):5f}$'
    if root.imag == 0:
      plt.sca(axs[i,0])
      plt.scatter(root.real, root.imag, label='start', c='b')
      root_pows = root**pows
      plt.plot(root_pows.real, root_pows.imag, c='b')
      plt.xlabel('Re', fontsize=14)
      plt.ylabel('Im', fontsize=14)
      plt.title(f'$z^t = ({root.real:.5f})^t$', fontsize=14)
      plt.legend()
      plt.sca(axs[i,1])
      plt.title(f'Its conjugate is not one of the roots.')
      plt.sca(axs[i,2])
      plt.title(f'For integer powers, the imaginary component disappears.')
      plt.sca(axs[i,3])
      root_pows = root**pows
      plt.plot(pows, root_pows.real, c='violet')
      plt.xlabel('t', fontsize=14)
      eqn = f'$z^t + \\bar{{z}}^t = \Re{{(z^t)}} = r^t\\cos(t\\theta)$'
      title = eqn + '\n' + modulus + ', ' + argument
      plt.title(title, fontsize=14)
    else:
      plt.sca(axs[i,0])
      plt.scatter(root.real, root.imag, label='start', c='b')
      root_pows = root**pows
      plt.plot(root_pows.real, root_pows.imag, c='b')
      plt.xlabel('Re', fontsize=14)
      plt.ylabel('Im', fontsize=14)
      plt.title(f'$z^t = ({root:.5f})^t$', fontsize=14)
      plt.legend()
      plt.sca(axs[i,1])
      plt.scatter(root.real, -root.imag, label='start', c='r')
      root_conj_pows = root.conj()**pows
      plt.plot(root_conj_pows.real, root_conj_pows.imag, c='r')
      plt.xlabel('Re', fontsize=14)
      plt.ylabel('Im', fontsize=14)
      plt.title(f'$\\bar{{z}}^t = ({root.conj():.5f})^t$', fontsize=14)
      plt.legend()
      plt.sca(axs[i,2])
      plt.plot(root_conj_pows.real, root_conj_pows.imag, c='r')
      plt.plot(root_pows.real, root_pows.imag, c='b')
      plt.xlabel('Re', fontsize=14)
      plt.ylabel('Im', fontsize=14)
      plt.title(f'$z^t, \\bar{{z}}^t$', fontsize=14)
      plt.sca(axs[i,3])
      root_sum = root_pows + root_conj_pows
      plt.plot(pows, root_sum.real, c='violet')
      plt.xlabel('t', fontsize=14)
      eqn = f'$z^t + \\bar{{z}}^t = 2 \Re{{(z^t)}} = 2r^t\\cos(t\\theta)$'
      title = eqn + '\n' + modulus + ', ' + argument
      plt.title(title, fontsize=14)
  plt.subplots_adjust(hspace=.3)


def plot_err_fourier(x):
  fig, axs = plt.subplots(1,3, figsize=(18,6))
  plt.sca(axs[0])
  plt.plot(x,pisot_search.nint_dist(x), label='[x]-x')
  plt.plot(x,pisot_search.nint_dist_appr(x,10), alpha=0.7, label='appr')
  plt.plot(x,pisot_search.nint_dist_appr_deriv(x,10)/30, alpha=0.5, label='deriv')
  plt.gca().set_title('distance from nearest integer')
  plt.legend()

  plt.sca(axs[1])
  plt.plot(x,np.round(x), label='[x]')
  plt.plot(x,pisot_search.nint_dist_appr(x,10)+x, alpha=0.7, label='appr')
  plt.plot(x,pisot_search.nint_dist_appr_deriv(x,10)/30+1, alpha=0.5, label='deriv')
  plt.gca().set_title('nearest integer')
  plt.legend()

  plt.sca(axs[2])
  plt.plot(x,pisot_search.nint_dist(x)**2, label=r'$([x]-x)^2$')
  plt.plot(x,(pisot_search.nint_dist_appr(x,10))**2, alpha=0.7, label='appr')
  plt.plot(x,(pisot_search.nint_dist_appr_deriv(x,10)/30)**2, alpha=0.5, label='deriv')
  plt.gca().set_title('sqaured distance from nearest integer')
  plt.legend()