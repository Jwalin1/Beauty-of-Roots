import numpy as np


tau = 2 * np.pi

phi = (1 + 5**.5) * .5
# Also given as the biggest positive root of the polynomial
# golden_poly = np.poly1d([1, -1, -1])
# phi = max(abs(golden_poly.roots))

psi = (1/3) * (1 + ((29+3*(93**.5))/2)**(1/3) + ((29-3*(93**.5))/2)**(1/3))
# super_golden_poly = np.poly1d([1, -1, 0, -1])
# psi = max(abs(super_golden_poly.roots))

plastic_const = ((9 + 69**.5)/18)**(1/3) + ((9 - 69**.5)/18)**(1/3)
# Also given as the biggest positive root of the polynomial
# plastic_poly = np.poly1d([1, 0, -1, -1])
# plastic_const = max(abs(plastic_poly.roots))

lehmer_poly_coeffs = [1, 1, 0, -1, -1, -1, -1, -1, 0, 1, 1]
salem_const = max(abs(np.poly1d(lehmer_poly_coeffs).roots))


pisot_poly_coeffs = [
  [1, 0, -1, -1],  # Plastic ratio.
  [1, -1, 0, 0, -1],
  [1, -1, -1, 1, 0, -1],
  [1, -1, 0, -1],  # Supergolden ratio.
  [1, -1, -1, 0, 1, 0, -1],
  [1, 0, -1, -1, -1, -1],
  [1, -1, -1, 0, 0, 1, 0, -1],
  [1, -2, 1, 0, -1, 1, -1],
  [1, -1, 0, -1, 0, -1],
  [1, -1, -1, 0, 0, 0, 1, 0, -1],
  [1, -1, -1],  # Golden ratio.
  [1, -2, 2, -3, 2, -2, 1, 0, 0, 1, -1, 2, -2, 2, -2, 1, -1],
  [1, -2, 0 ,1 ,-1, 0, 1, -1, 0, 1, 0, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1]
]