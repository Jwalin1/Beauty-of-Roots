import pygad
import numpy as np
import numba as nb
import mpmath as mpm
from pprint import pprint
from tqdm.auto import tqdm
from IPython import display
from functools import partial

tau = 2 * np.pi


def _normalized_weighted_abs_sum(x, n_pows):
  pows = np.arange(1, 1+n_pows)
  weights = np.arange(n_pows)
  Y = x ** pows
  y_errs = np.array([abs(y - mpm.nint(y)) for y in Y])
  y_err = (y_errs * weights).sum()
  y_err /= (n_pows) * (n_pows+1) / 4
  return y_err

def _normalized_weighted_squared_sum(x, n_pows):
  pows = np.arange(1, 1+n_pows)
  weights = np.arange(n_pows)
  Y = x ** pows
  y_errs = np.array([(y - mpm.nint(y)**2) for y in Y])
  y_err = (y_errs * weights).sum()
  y_err /= (n_pows) * (n_pows+1) / 8
  return y_err  


# Fastest approach but not memory efficient.
def get_nint_errs_numpy(n_digits=6, n_pows=None):
  X = np.linspace(1,2, num=1+10**n_digits)
  n_pows = n_digits*5 if n_pows is None else n_pows
  pows = np.arange(1,1+n_pows)
  weights = np.arange(n_pows)
  Ys = X[:,None] ** pows
  Y_errs = abs(Ys - np.round(Ys))
  Y_errs = (Y_errs*weights).sum(axis=1)
  Y_errs /= (n_pows) * (n_pows+1) / 4
  return X, Y_errs

# Relatively fast and doesnt consume very large memory, but no tqdm.
@nb.njit()
def get_nint_errs_numba(n_digits=8, n_pows=None):
  X = np.linspace(1,2, 1+10**n_digits)
  n_pows = n_digits*5 if n_pows is None else n_pows
  pows = np.arange(1, 1+n_pows, 1, np.float64)
  out = np.empty_like(pows)
  Y_errs = []
  weights = np.arange(len(pows))
  for x in X:
    Y = x ** pows
    y_err = np.abs(Y - np.round(Y,0, out))
    y_err = (y_err*weights).sum()
    y_err /= (n_pows) * (n_pows+1) / 4
    Y_errs.append(y_err)
  return X, np.array(Y_errs)

# Very slow, useless.
def get_nint_errs_mpmath(n_digits=6, n_pows=None, accuracy_digits=15):
  mpm.mp.dps = accuracy_digits
  n_pows = n_digits*5 if n_pows is None else n_pows
  x = mpm.mpf(1)
  dx = mpm.mpf(1 / 10**n_digits)
  nums = 1 + 10**n_digits
  X, Y_errs = [], []
  for _ in tqdm(range(nums)):
    y_err = _normalized_weighted_squared_sum(x, n_pows)
    Y_errs.append(y_err)
    X.append(x)
    x += dx
  return np.array(X), np.array(Y_errs)


def _print_dict(top_n, perc):
  display.clear_output()
  print(f'{perc:.5}% checked')
  pprint(top_n)

def random_search_mpm(num_range=(1.34,1.96), n_digits=20, nums_to_check=10**5,
                      top_n=5, min_apart=0.1, min_near=0.01,
                      print_every=5):
  mpm.mp.dps = n_digits
  n_pows = int(n_digits * 3)
  top_n = {k+2:n_digits**2 for k in range(top_n)}

  perc = 0
  for i in tqdm(range(1,1+nums_to_check)):
    x = num_range[0] + mpm.rand() * (num_range[1] - num_range[0])
    y_err = _normalized_weighted_squared_sum(x, n_pows)

    found = False
    for num, err in top_n.items():
      if (y_err <= err):
        dists = abs(np.array(list(top_n.keys())) - x)
        if (dists.min() > min_apart) or (abs(num-x) < min_near):
          del top_n[num]
          top_n[x] = y_err
          found = True
          break
    if found:
      _print_dict(top_n, 100*i/nums_to_check)
    elif 100 * i / nums_to_check >= perc:
      _print_dict(top_n, 100*i/nums_to_check)
      perc += print_every
  return top_n



def guided_search(n_digits=20, num_generations=2000):
  mpm.mp.dps = n_digits
  window_size = 5
  n_pows = int(n_digits * 3)

  def fitness_function(ga_instance, solution, solution_idx):
    num = '0.' + ''.join(str(num) for num in solution)
    num = mpm.mpf('1.1') + mpm.mpf(num)
    y_err = _normalized_weighted_squared_sum(num, n_pows)
    return float(1  / y_err)

  def on_generation(ga_instance):
    pbar.update()

  ga_instance = pygad.GA(num_generations = num_generations,
                         num_parents_mating = 4,
                         fitness_func = fitness_function,
                         sol_per_pop = 8,
                         num_genes = n_digits,
                         gene_space = list(range(10)),
                         gene_type = int,
                         on_generation = on_generation)    

  with tqdm(total=num_generations) as pbar:
    ga_instance.run()

  solution, solution_fitness, solution_idx = ga_instance.best_solution()
  num = '0.' + ''.join(str(num) for num in solution)
  num = mpm.mpf('1.1') + mpm.mpf(num)

  print("Parameters of the best solution : {solution}".format(solution=solution))
  print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
  ga_instance.plot_fitness()
  return num


def nint_dist(x):
  return np.round(x) - x

def nint_dist_deriv(x):
  return -1

def nint_dist_appr(x, n=100):
  sum_ = 0
  for k in range(1, n):
    sum_ += ((-1)**k) * np.sin(tau*k*x) / k
  return 2 * sum_ / tau

def nint_dist_appr_deriv(x, n=100):
  sum_ = 0
  for k in range(1, n):
    sum_ += ((-1)**k) * np.cos(tau*k*x)
  return 2 * sum_

def loss(x, n):
  sum_ = 0
  for i in range(1, n):
    sum_ += i * (nint_dist(x**i) ** 2)
  return sum_ / 2

def loss_deriv(x, n):
  sum_ = 0
  for i in range(1, n):
    sum_ += (i*i) * (x**(i-1)) * nint_dist(x**i) * nint_dist_deriv(x**i)
  return sum_

def grad_desc(x, lr, deriv_func, n_iters):
  steps = [x]
  for _ in tqdm(range(n_iters)):
    x -= lr * deriv_func(x)
    steps.append(x)
  return steps