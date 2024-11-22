#include <stdio.h>
#include <time.h>

#include "data.h"

#define NN_IMPL
#include "nn.h"

#define array_len_zu(a) sizeof(a) / sizeof(size_t)

int main(void) {
  srand(time(0));

  Mat *data = get_data();
  Mat inp = data[0];
  Mat out = data[1];

  size_t shape[] = {2, 8, 8, 8, 1};
  NN nn = nn_alloc(shape, array_len_zu(shape) - 1);
  NN g = nn_alloc(shape, array_len_zu(shape) - 1);
  nn_rand(nn, 0.1, 0.1);

  nn_train(g, nn, inp, out);

  for (size_t i = 0; i < inp.cols; i++) {
    nn_forward(nn, mat_view(inp, 0, inp.rows, i, i + 1));
    printf("%d * %d = %f\n", (int)mat_at(nn_inp(nn), 0, 0),
           (int)mat_at(nn_inp(nn), 1, 0), mat_at(nn_out(nn), 0, 0));
  }

  return 0;
}
