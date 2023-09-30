import itertools
import numpy as np
import mpmath as mpm
from tqdm.auto import tqdm


def get_max_root(coeffs, n_digits):
  mpm.mp.dps = n_digits
  roots = mpm.polyroots(coeffs)
  real_pos_roots = [root.real for root in roots if root.imag == 0
                    and root.real > 0]
  return max(real_pos_roots)

def check_convergence(num, window_size=5, threshold=1e-2, limit=2000):
  """Keeps raising the number to higher powers until n consecutive numbers lie within
  the given threshold or the powers exceed a certain limit."""
  x, pow = 1, 0
  below_thresh = 0
  while True:
    x *= num
    pow += 1
    err = abs(x - mpm.nint(x))
    if err <= threshold:
      below_thresh += 1
    else:
      below_thresh = 0
    if below_thresh == window_size:
      break
    if pow == limit:  # Did not converge.
      break  
  return pow


def _crop_leading_zeros(coeffs: tuple):
  return coeffs[next((i for i, x in enumerate(coeffs) if x != 0), len(coeffs)):]

def _get_index(num, values):
  for i, value in enumerate(values):
    if np.isclose(num, value):
      return i

def filtered_coeffs(max_degree, coeff_choices):
  for combo in itertools.product(coeff_choices, repeat=max_degree):
    if combo[-1] != 0:  # Must not end with 0.
        yield combo

def get_pisot_coeffs_data(max_degree=10, coeff_choices=[-1,0,1]):
  """Compute the pisot numbers resulting from littlewood polynomials upto a certain degree.
  Note that polynomials with other integer coefficients apart from +/-1
  can also yield pisot numbers but these arent covered by this function."""
  PV_nums = {}
  # all_coeffs = itertools.product([-1,0,1], repeat=max_degree)  # Only littelwood polynomials.
  # total = 3**max_degree
  all_coeffs = filtered_coeffs(max_degree-1, coeff_choices)
  total = (len(coeff_choices)**(max_degree-2)) * (len(coeff_choices)-1)
  for coeffs in tqdm(all_coeffs, total=total):
    coeffs = (1, *_crop_leading_zeros(coeffs))  # Must begin with 1.
    poly = np.poly1d(coeffs)
    roots = poly.roots
    roots_out = roots[abs(roots)>1]
    # No roots on unit circle, all inside except one root.
    roots_on = roots[np.isclose(abs(roots),1)]
    if (len(roots_out)==1) and (len(roots_on)==0):
      root_out = roots_out[0]
      if root_out.imag == 0:
        root_out = root_out.real
        if root_out > 0:
          index = _get_index(root_out, PV_nums.values())
          if index is None:
            PV_nums[coeffs] = root_out
          else:
            root_out_coeffs = list(PV_nums.keys())[index]
            if len(coeffs) < len(root_out_coeffs):
              del PV_nums[root_out_coeffs]
              PV_nums[coeffs] = root_out
  return PV_nums
