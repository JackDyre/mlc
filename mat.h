#ifndef MAT_H
#define MAT_H

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#define DTYPE float
#define mat_at(m, r, c) (m).elems[(r) * (m).stride + (c)]

typedef struct Mat {
  size_t rows;
  size_t cols;
  size_t stride;
  DTYPE *elems;
} Mat;

Mat mat_alloc(size_t rows, size_t cols);
void mat_fill(Mat m, DTYPE val);
void mat_fill_from(Mat m, DTYPE *elems);
void mat_rand(Mat m, DTYPE min, DTYPE max);
void mat_copy(Mat m, Mat other);
void mat_print(Mat m, const char *name);
Mat mat_view(Mat m, size_t rmin, size_t rmax, size_t cmin, size_t cmax);
void mat_scalar(Mat m, DTYPE s);
void mat_add(Mat m, Mat other);
void mat_mul(Mat m, Mat a, Mat b);
void mat_actf(Mat m);
DTYPE actf(DTYPE x);
DTYPE d_actf(DTYPE x);

#endif // MAT_H

#ifdef MAT_IMPL

Mat mat_alloc(size_t rows, size_t cols) {
  Mat m;
  m.rows = rows;
  m.cols = cols;
  m.stride = cols;
  m.elems = (DTYPE *)malloc(rows * cols * sizeof(DTYPE));
  assert(m.elems);
  return m;
}

void mat_fill(Mat m, DTYPE val) {
  for (size_t r = 0; r < m.rows; r++) {
    for (size_t c = 0; c < m.cols; c++) {
      mat_at(m, r, c) = val;
    }
  }
}

void mat_fill_from(Mat m, DTYPE *elems) {
  for (size_t r = 0; r < m.rows; r++) {
    for (size_t c = 0; c < m.cols; c++) {
      mat_at(m, r, c) = elems[r * m.cols + c];
    }
  }
}

void mat_rand(Mat m, DTYPE min, DTYPE max) {
  for (size_t r = 0; r < m.rows; r++) {
    for (size_t c = 0; c < m.cols; c++) {
      mat_at(m, r, c) = min + (max - min) * (DTYPE)rand() / (DTYPE)RAND_MAX;
    }
  }
}

void mat_copy(Mat m, Mat other) {
  assert(m.rows == other.rows);
  assert(m.cols == other.cols);
  for (size_t r = 0; r < m.rows; r++) {
    for (size_t c = 0; c < m.cols; c++) {
      mat_at(m, r, c) = mat_at(other, r, c);
    }
  }
}

void mat_print(Mat m, const char *name) {
  if (name[0]) {
    printf("%s=[\n", name);
  } else {
    printf("[\n");
  }
  for (size_t r = 0; r < m.rows; r++) {
    printf("  ");
    for (size_t c = 0; c < m.cols; c++) {
      printf("%f  ", mat_at(m, r, c));
    }
    printf("\n");
  }
  printf("]\n");
}

Mat mat_view(Mat m, size_t rmin, size_t rmax, size_t cmin, size_t cmax) {
  assert(0 <= rmin && rmin < rmax && rmax <= m.rows && 0 <= cmin &&
         cmin < cmax && cmax <= m.cols);
  size_t nrows = rmax - rmin;
  size_t ncols = cmax - cmin;
  Mat n;
  n.rows = nrows;
  n.cols = ncols;
  n.stride = m.stride;
  n.elems = m.elems + (rmin * m.stride + cmin);
  return n;
}

void mat_scalar(Mat m, DTYPE s) {
  for (size_t r = 0; r < m.rows; r++) {
    for (size_t c = 0; c < m.cols; c++) {
      mat_at(m, r, c) *= s;
    }
  }
}

void mat_add(Mat m, Mat other) {
  assert(m.rows == other.rows);
  assert(m.cols == other.cols);
  for (size_t r = 0; r < m.rows; r++) {
    for (size_t c = 0; c < m.cols; c++) {
      mat_at(m, r, c) += mat_at(other, r, c);
    }
  }
}

void mat_mul(Mat m, Mat a, Mat b) {
  assert(m.rows == a.rows);
  assert(m.cols == b.cols);
  assert(a.cols == b.rows);
  for (size_t r = 0; r < m.rows; r++) {
    for (size_t c = 0; c < m.cols; c++) {
      mat_at(m, r, c) = 0;
      for (size_t i = 0; i < a.cols; i++) {
        mat_at(m, r, c) += mat_at(a, r, i) * mat_at(b, i, c);
      }
    }
  }
}

void mat_actf(Mat m) {
  for (size_t r = 0; r < m.rows; r++) {
    for (size_t c = 0; c < m.cols; c++) {
      mat_at(m, r, c) = actf(mat_at(m, r, c));
    }
  }
}

// DTYPE actf(DTYPE x) { return 1.0 / (1.0 + expf(-x)); }
// DTYPE actf(DTYPE x) { return x > 0 ? x : 0; }
DTYPE actf(DTYPE x) { return x > 0 ? 1.5 * x : .5 * x; }
// DTYPE actf(DTYPE x) { return sinf(x); }
// DTYPE actf(DTYPE x) { return expf(x) - 1; }

// DTYPE d_actf(DTYPE x) {
//   DTYPE ex = expf(-x);
//   return ex / ((1 + ex) * (1 + ex));
// }
// DTYPE d_actf(DTYPE x) { return x > 0 ? 1 : 0; }
DTYPE d_actf(DTYPE x) { return x > 0 ? 1.5 : .5; }
// DTYPE d_actf(DTYPE x) { return cosf(x); }
// DTYPE d_actf(DTYPE x) { return expf(x); }

#endif // MAT_IMPL
