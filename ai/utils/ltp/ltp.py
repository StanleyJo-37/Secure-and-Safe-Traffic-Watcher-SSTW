import numpy as np

def uniformity(bits):
  transitions = 0
  curr_bit = bits[0]

  for bit in bits[1:]:
    if bit != curr_bit:
      transitions += 1
    curr_bit = bit

  return transitions

def bits_to_integer(bits: list) -> int:
	total = 0

	for i, bit in enumerate(bits):
		total += pow(2, i) * bit

	return total

class LTP(object):
  def __init__(self, k: int = 10):
    self.uniform_lookup = np.array([58] * 256)
    bin_id = 0
    
    for i in range(256):
      bits = np.array([(i >> j) & 1 for j in range(8)])

      if uniformity(bits) <= 2:
        self.uniform_lookup[i] = bin_id
        bin_id += 1
      else:
        self.uniform_lookup[i] = 58
  
    self.k = k
    
  def __call__(self, img, *args, **kwds):
    hist_upper = np.zeros(59, int)
    hist_lower = np.zeros(59, int)

    offsets = [
      (-1, -1), (-1, 0), (-1, 1),
      (0, -1),           (0, 1),
      (1, -1), (1, 0), (1, 1),
    ]

    for i in range(1, img.shape[0] - 1):
      for j in range(1, img.shape[1] - 1):

        upper_bits, lower_bits = [], []
        center = img[i, j]

        center = int(center)
        k = int(k)

        lower_bound = max(0, center - k)
        upper_bound = min(255, center + k)

        for dy, dx in offsets:
          val = img[i + dy, j + dx]

          upper_bits.append(val > upper_bound)
          lower_bits.append(val < lower_bound)

        hist_upper[self.uniform_lookup[bits_to_integer(upper_bits)]] += 1
        hist_lower[self.uniform_lookup[bits_to_integer(lower_bits)]] += 1

    return np.concatenate([hist_upper, hist_lower])