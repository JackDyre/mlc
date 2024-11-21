#ifndef DATA_H
#define DATA_H

// #define MAT_IMPL
#include "mat.h"

#define array_lenf(a) sizeof(a) / sizeof(DTYPE)

Mat *get_data() {
  size_t shape_prim[] = {1, 1};

  DTYPE data_a[] = {0, 1, 2, 3, 4, 5, 6, 7,

                    .1, .15, .25, .35, .55, .65, .85, .95};

  size_t r = shape_prim[0] + shape_prim[1];
  assert(array_lenf(data_a) % r == 0);
  size_t c = array_lenf(data_a) / r;

  Mat data_m = mat_alloc(r, c);
  mat_fill_from(data_m, data_a);

  Mat inp = mat_view(data_m, 0, shape_prim[0], 0, data_m.cols);
  Mat out = mat_view(data_m, shape_prim[0], data_m.rows, 0, data_m.cols);

  Mat *ret = (Mat *)malloc(2 * sizeof(Mat));
  assert(ret);
  ret[0] = inp;
  ret[1] = out;
  return ret;
}

#endif // DATA_H
