import numpy as np
import mpmath as mpm
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from celluloid import Camera

from . import visualize, pisot_search, poly_utils


def powers_animate(n_digits, n_pows):
  fig, axs = plt.subplots()
  plt.subplots_adjust()
  camera = Camera(fig)

  for i in tqdm(range(2, n_pows+1)):
    x, y_errs = pisot_search.get_nint_errs_numpy(n_digits, i)
    visualize.plot_nint_errs(x, y_errs, i)
    camera.snap()
  plt.close()

  anim = camera.animate()
  return anim

def powers_animate_single(num, n_pows):
  fig, axs = plt.subplots()
  camera = Camera(fig)

  for i in tqdm(range(2, n_pows)):
    visualize.plot_approximation_error_1(num, i)
    camera.snap()
  plt.close()

  anim = camera.animate()
  return anim


# https://math.stackexchange.com/a/3253471
def sigmoid_lerp(t, start, end, k):
  return end - ((end-start)/(1+(1/t-1)**-k))

def interp_animate(coeffs_1, coeffs_2, n_digits):
  fig, axs = plt.subplots()
  camera = Camera(fig)

  mpm.mp.dps = n_digits
  roots = mpm.polyroots(coeffs_1)
  root_1 = max(list(map(abs, roots)))
  n_pows_1 = poly_utils.check_convergence(root_1)
  roots = mpm.polyroots(coeffs_2)
  root_2 = max(list(map(abs, roots)))
  n_pows_2 = poly_utils.check_convergence(root_2)
  n_pows = max(n_pows_1, n_pows_2)

  if root_1 > root_2:
    root_1, root_2 = root_2, root_1

  t = np.linspace(0, 1, 400)
  nums = sigmoid_lerp(t, root_1, root_2, k=17*(n_pows/100))
  for num in tqdm(nums):
    visualize.plot_approximation_error_1(num, n_pows)
    camera.snap()
  plt.close()

  anim = camera.animate()
  return anim