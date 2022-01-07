#pragma once

#include <torch/torch.h>

struct GenerateDate{
    torch::Tensor data, target;
};