#pragma once
#include "workspace.hpp"

void gpu_scale_problem(const SharedWorkspace &ws, cudaStream_t stream);
void gpu_unscale(const SharedWorkspace &ws, const PrivateWorkspace &pws, double *x_out);
