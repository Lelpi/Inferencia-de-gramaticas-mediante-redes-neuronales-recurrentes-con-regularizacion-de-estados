# Incorpora funciones para generar los datos a partir de una cadena de texto 
# arbitraria

import numpy as np
import numpy.matlib

#-------------------------------------------------------------------------------
# Funcion que parte los datos en batches:
#-------------------------------------------------------------------------------
def prepare_sequences(x, y, wlen):
  (n, dim) = x.shape
  nchunks = dim//wlen
  xseq = np.array(np.split(x, nchunks, axis=1))
  xseq = xseq.reshape((n*nchunks, wlen))
  yseq = np.array(np.split(y, nchunks, axis=1))
  yseq = yseq.reshape((n*nchunks, wlen))
  return xseq, yseq

#-------------------------------------------------------------------------------
# Funcion que genera los datos para el problema de la paridad:
#-------------------------------------------------------------------------------
def get_data_parity(n_symbols, batch_size, wlen, p_a, p_b):
  # Nota: wlen debe ser divisor de n_symbols!!
  p_dolar = 1. - p_a - p_b

  # Genero la entrada x:
  p = np.random.rand(batch_size, n_symbols)
  x = np.full((batch_size, n_symbols), '$')
  ix_a = p < p_a
  x[ix_a] = 'a'
  ix_b = (p >= p_a) & (p < p_a + p_b)
  x[ix_b] = 'b'

  # El primer simbolo siempre tiene que ser un $:
  x[:, 0] = '$'

  x_as_string = [''.join(z) for z in x]

  # Genero la salida y:
  y = np.zeros((batch_size, n_symbols))
  state = 0
  for j in range(batch_size):
    for i, s in enumerate(x[j]):
      if s == '$':
        state = 0
      elif s == 'a':
        state +=1
      y[j, i] = state % 2
  y_as_string = [''.join(z.astype('<U1')) for z in y]

  # Paso a codificacion one-hot:
  xs, ys = prepare_sequences(x, y, wlen)

  symbols = np.unique(x)
  char_to_ix = {s: i for i, s in enumerate(symbols)}
  ix_to_char = {i: s for i, s in enumerate(symbols)}

  xs_numeric = np.zeros(xs.shape)

  for s in symbols:
    xs_numeric[xs == s] = char_to_ix[s]

  one_hot_values = np.array(list(ix_to_char.keys()))
  xs_one_hot = 1 * (xs_numeric[:, :, None] == one_hot_values[None, None, :])

  # Transformacion de la clase a variable categorica:
  ys_categorical = np.concatenate((ys[:, :, None], ys[:, :, None] != 1), axis=2)

  return x, y, x_as_string, y_as_string, xs, ys, xs_one_hot, ys_categorical, symbols

#-------------------------------------------------------------------------------
# ...
#-------------------------------------------------------------------------------
def get_array_addition(n_symbols, num_dolar, base):
  # Sumandos sin $:
  sumando1 = np.random.randint(base, size=(n_symbols,))
  sumando2 = np.random.randint(base, size=(n_symbols,))

  # Indices de $:
  dolar_ix = np.random.permutation(n_symbols)[:num_dolar]
  dolar_ix

  # Generacion de sumandos con $ y en el formato adecuado:
  carry = 0
  inputs = np.asarray(sumando1, dtype='str')
  outputs = np.asarray(sumando1, dtype='str')
  for i, (s1, s2) in enumerate(zip(sumando1, sumando2)):  
    if i == 0:
      inputs[i] = '$'
      outputs[i] = '0'
      carry = 0
    elif i == (n_symbols - 1):
      inputs[i] = '$'
      outputs[i] = "%d" % carry
      carry = 0
    elif i in dolar_ix:
      inputs[i] = '$'
      outputs[i] = "%d" % carry
      carry = 0
    else:
      inputs[i] = "%d+%d" % (s1, s2)
      res = s1 + s2 + carry
      s = res % base
      outputs[i] = "%d" % s
      carry = res // base

  return inputs, outputs

#-------------------------------------------------------------------------------
# Funcion que genera los datos para el problema de las sumas:
#-------------------------------------------------------------------------------
def get_data_addition(n_symbols, batch_size, wlen, num_dolar, base):
  lista_x = []
  lista_y = []

  for i in range(batch_size):
    inputs, outputs = get_array_addition(n_symbols, num_dolar, base)
    lista_x.append(inputs)
    lista_y.append(outputs)

  xs_one_hot, ys_one_hot, xs_sparse, ys_sparse, xs_symbols, ys_symbols = get_data_from_arrays(lista_x, lista_y, wlen)
  return xs_one_hot, ys_one_hot, xs_sparse, ys_sparse, xs_symbols, ys_symbols

#-------------------------------------------------------------------------------
# Funcion que genera los datos de entrada y salida a partir de un conjunto de
# cadenas.
# Entrada: - lista de cadenas x
#          - lista de cadenas y
#          - wlen
# Salida: - datos organizados por batches  
#-------------------------------------------------------------------------------
def get_data_from_strings(data_str_x, data_str_y, wlen):
  # En batch_size es en numero de cadenas en x e y:
  batch_size = len(data_str_x)
  
  # Corto todas las cadenas a longitud igual al mayor multiplo de wlen menor que
  # todas las longitudes de cadena:
  minlen = len(data_str_x[0])
  for c in data_str_x:
    if len(c) < minlen:
      minlen = len(c)
  while minlen % wlen != 0:
    minlen -=1
  data_str_x = [c[:minlen] for c in data_str_x]
  data_str_y = [c[:minlen] for c in data_str_y]
  
  # Transformo las cadenas a array de numpy:
  x = np.array([[c for c in m] for m in data_str_x])
  y = np.array([[c for c in m] for m in data_str_y])
  
  # Parto en batches:
  xs, ys = prepare_sequences(x, y, wlen)
  
  # Paso a one-hot:
  xs_one_hot, xs_symbols = one_hot_encoding(xs)
  ys_one_hot, ys_symbols = one_hot_encoding(ys)

  # Codificacion numerica (sparse):
  xs_sparse = np.argmax(xs_one_hot, axis=2)
  ys_sparse = np.argmax(ys_one_hot, axis=2)

  # Devuelvo: 
  return xs_one_hot, ys_one_hot, xs_sparse, ys_sparse, xs_symbols, ys_symbols

#-------------------------------------------------------------------------------
# Funcion que genera los datos de entrada y salida a partir de un conjunto de
# arrays de numpy. Los arrays contienen cadenas, cada cadena es un simbolo.
# Entrada: - lista de arrays x
#          - lista de arrays y
#          - wlen
# Salida: - datos organizados por batches  
#-------------------------------------------------------------------------------
def get_data_from_arrays(data_arr_x, data_arr_y, wlen):
  # En batch_size es en numero de cadenas en x e y:
  batch_size = len(data_arr_x)
  
  # Corto todas las cadenas a longitud igual al mayor multiplo de wlen menor que
  # todas las longitudes de cadena:
  minlen = len(data_arr_x[0])
  for c in data_arr_x:
    if len(c) < minlen:
      minlen = len(c)
  while minlen % wlen != 0:
    minlen -=1
  data_arr_x = [c[:minlen] for c in data_arr_x]
  data_arr_y = [c[:minlen] for c in data_arr_y]
  
  # Transformo las cadenas a array de numpy:
  x = np.array([[c for c in m] for m in data_arr_x])
  y = np.array([[c for c in m] for m in data_arr_y])
  
  # Parto en batches:
  xs, ys = prepare_sequences(x, y, wlen)
  
  # Paso a one-hot:
  xs_one_hot, xs_symbols = one_hot_encoding(xs)
  ys_one_hot, ys_symbols = one_hot_encoding(ys)

  # Codificacion numerica (sparse):
  xs_sparse = np.argmax(xs_one_hot, axis=2)
  ys_sparse = np.argmax(ys_one_hot, axis=2)

  # Devuelvo: 
  return xs_one_hot, ys_one_hot, xs_sparse, ys_sparse, xs_symbols, ys_symbols

#-------------------------------------------------------------------------------
# Funcion que transforma a codificacion one-hot.
# Entrada: - data: numpy array con los datos como chars
# Salida: - numpy array con los datos en codificacion one-hot. Este array tiene
#           una dimension mas que data, con tantas posiciones como simbolos
#         - numpy array con los simbolos. El orden de los simbolos en el array
#           se corresponde con el indice en la representacion one-hot.
#-------------------------------------------------------------------------------
def one_hot_encoding(data):
  symbols = np.unique(data)
  
  char_to_ix = {s: i for i, s in enumerate(symbols)}
  ix_to_char = {i: s for i, s in enumerate(symbols)}

  data_numeric = np.zeros(data.shape)

  for s in symbols:
    data_numeric[data == s] = char_to_ix[s]

  one_hot_values = np.array(list(ix_to_char.keys()))
  data_one_hot = 1 * (data_numeric[:, :, None] == one_hot_values[None, None, :])

  return data_one_hot, symbols
