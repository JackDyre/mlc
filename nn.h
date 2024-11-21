#ifndef NN_H
#define NN_H

#define MAT_IMPL
#include "mat.h"

#define nn_inp(nn) (nn).as[0]
#define nn_out(nn) (nn).as[(nn).lc]

typedef struct NN {
  size_t lc;
  size_t *shape; // lc + 1
  Mat *as;       // lc + 1
  Mat *ws;       // lc
  Mat *bs;       // lc
} NN;

NN nn_alloc(size_t *shape, size_t lc);
void nn_rand(NN nn, DTYPE min, DTYPE max);
void nn_print(NN nn);
void nn_forward(NN nn, Mat inp);
DTYPE nn_cost(NN nn, Mat inp, Mat out);
DTYPE nn_cost_many(NN nn, Mat inp, Mat out);
void nn_grad(NN g, NN nn, Mat inp, Mat out);
void nn_step(NN g, NN nn, Mat inp, Mat out);
void nn_train(NN g, NN nn, Mat inp, Mat out);

#endif // NN_H

#ifdef NN_IMPL

NN nn_alloc(size_t *shape, size_t lc) {
  NN nn;
  nn.lc = lc;
  nn.shape = (size_t *)malloc((lc + 1) * sizeof(size_t));
  nn.as = (Mat *)malloc((lc + 1) * sizeof(Mat));
  nn.ws = (Mat *)malloc(lc * sizeof(Mat));
  nn.bs = (Mat *)malloc(lc * sizeof(Mat));
  for (size_t i = 0; i <= lc; i++) {
    nn.shape[i] = shape[i];
    nn.as[i] = mat_alloc(shape[i], 1);
    if (i < lc) {
      nn.ws[i] = mat_alloc(shape[i + 1], shape[i]);
      nn.bs[i] = mat_alloc(shape[i + 1], 1);
    }
  }
  return nn;
}

void nn_rand(NN nn, DTYPE min, DTYPE max) {
  for (size_t i = 0; i < nn.lc; i++) {
    mat_rand(nn.ws[i], min, max);
    mat_rand(nn.bs[i], min, max);
  }
}

void nn_print(NN nn) {
  for (size_t i = 0; i < nn.lc; i++) {
    mat_print(nn.ws[i], "w");
    mat_print(nn.bs[i], "b");
  }
}

void nn_forward(NN nn, Mat inp) {
  assert(inp.rows == nn_inp(nn).rows);
  assert(inp.cols == nn_inp(nn).cols);
  mat_copy(nn_inp(nn), inp);
  for (size_t i = 0; i < nn.lc; i++) {
    mat_mul(nn.as[i + 1], nn.ws[i], nn.as[i]);
    mat_add(nn.as[i + 1], nn.bs[i]);
    mat_actf(nn.as[i + 1]);
  }
}

DTYPE nn_cost(NN nn, Mat inp, Mat out) {
  assert(inp.rows == nn_inp(nn).rows);
  assert(inp.cols == nn_inp(nn).cols);
  assert(out.rows == nn_out(nn).rows);
  assert(out.cols == nn_out(nn).cols);
  nn_forward(nn, inp);
  DTYPE cost = 0;
  for (size_t r = 0; r < nn_out(nn).rows; r++) {
    for (size_t c = 0; c < nn_out(nn).cols; c++) {
      cost += powf(mat_at(nn_out(nn), r, c) - mat_at(out, r, c), 2);
    }
  }
  cost /= (nn_out(nn).rows * nn_out(nn).cols);
  return cost;
}

DTYPE nn_cost_many(NN nn, Mat inp, Mat out) {
  assert(inp.rows == nn_inp(nn).rows);
  assert(out.rows == nn_out(nn).rows);
  assert(inp.cols == out.cols);
  DTYPE cost = 0;
  for (size_t i = 0; i < inp.cols; i++) {
    cost += nn_cost(nn, mat_view(inp, 0, inp.rows, i, i + 1),
                    mat_view(out, 0, out.rows, i, i + 1));
  }
  cost /= inp.cols;
  return cost;
}

void nn_grad(NN g, NN nn, Mat inp, Mat out) {
  assert(inp.rows == nn_inp(nn).rows);
  assert(inp.cols == nn_inp(nn).cols);
  assert(out.rows == nn_out(nn).rows);
  assert(out.cols == nn_out(nn).cols);
  nn_forward(nn, inp);
  for (size_t a = 0; a < out.rows; a++) {
    mat_at(nn_out(g), a, 0) =
        2 * (mat_at(nn_out(nn), a, 0) - mat_at(out, a, 0));
  }
  for (size_t idx = 0; idx < g.lc; idx++) {
    size_t l = g.lc - 1 - idx;
    for (size_t r = 0; r < g.ws[l].rows; r++) {
      for (size_t c = 0; c < g.ws[l].cols; c++) {
        DTYPE z = mat_at(nn.bs[l], r, 0);
        for (size_t a = 0; a < nn.as[l].rows; a++) {
          z += mat_at(nn.as[l], a, 0) * mat_at(nn.ws[l], r, a);
        }
        mat_at(g.ws[l], r, c) =
            mat_at(nn.as[l], c, 0) * d_actf(z) * mat_at(g.as[l + 1], r, 0);
      }
    }
    for (size_t b = 0; b < g.bs[l].rows; b++) {
      DTYPE z = mat_at(nn.bs[l], b, 0);
      for (size_t a = 0; a < nn.as[l].rows; a++) {
        z += mat_at(nn.as[l], a, 0) * mat_at(nn.ws[l], b, a);
      }
      mat_at(g.bs[l], b, 0) = d_actf(z) * mat_at(g.as[l + 1], b, 0);
    }
    for (size_t a = 0; a < g.as[l].rows; a++) {
      mat_at(g.as[l], a, 0) = 0;
      for (size_t i = 0; i < g.as[l + 1].rows; i++) {
        DTYPE z = mat_at(nn.bs[l], i, 0);
        for (size_t a = 0; a < nn.as[l].rows; a++) {
          z += mat_at(nn.as[l], a, 0) * mat_at(nn.ws[l], i, a);
        }
        mat_at(g.as[l], a, 0) +=
            mat_at(nn.ws[l], i, a) * d_actf(z) * mat_at(g.as[l + 1], a, 0);
      }
    }
  }
}

void nn_step(NN g, NN nn, Mat inp, Mat out) {
  assert(inp.rows == nn_inp(nn).rows);
  assert(out.rows == nn_out(nn).rows);
  assert(inp.cols == out.cols);
  DTYPE lr = 1e-2;
  for (size_t i = 0; i < inp.cols; i++) {
    nn_grad(g, nn, mat_view(inp, 0, inp.rows, i, i + 1),
            mat_view(out, 0, out.rows, i, i + 1));
    for (size_t l = 0; l < nn.lc; l++) {
      mat_scalar(g.ws[l], -lr / ((DTYPE)inp.cols));
      mat_scalar(g.bs[l], -lr / ((DTYPE)inp.cols));
      mat_add(nn.ws[l], g.ws[l]);
      mat_add(nn.bs[l], g.bs[l]);
    }
  }
}

void nn_train(NN g, NN nn, Mat inp, Mat out) {
  assert(g.lc == nn.lc);
  for (size_t i = 0; i <= nn.lc; i++) {
    assert(g.shape[i] == nn.shape[i]);
  }
  assert(inp.rows == nn_inp(nn).rows);
  assert(out.rows == nn_out(nn).rows);
  assert(inp.cols == out.cols);
  DTYPE cost = nn_cost_many(nn, inp, out);
  printf("\nCost before training: %f\n", cost);
  for (size_t i = 0; i < 1e6; i++) {
    nn_step(g, nn, inp, out);
  }
  cost = nn_cost_many(nn, inp, out);
  printf("Cost after training:  %f\n\n", cost);
}

#endif // NN_IMPL
