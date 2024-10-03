#pragma once

#include <cub/warp/warp_scan.cuh>

static constexpr auto gThreadBlockDim = 32;

template <typename integer>
constexpr inline integer ceil_div(integer n, integer m)
{
    return (n + m - 1) / m;
}

using WarpScan = cub::WarpScan<float2>;
