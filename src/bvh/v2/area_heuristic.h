#ifndef BVH_V2_AREA_HEURISTIC_H
#define BVH_V2_AREA_HEURISTIC_H

#include "bvh/v2/bbox.h"
#include "bvh/v2/utils.h"

#include <cstddef>

namespace bvh::v2 {

/// Surface Area Heuristic (SAH) build heuristic. If primitives are intersected in groups of `2^N`,
/// `LogClusterSize` can be set to `N` to reflect that fact in the cost function.
template <typename T, size_t LogClusterSize = 0>
class AreaHeuristic {
public:
    /// Create an area heuristic used by BVH builders to determine when to split a node.
    /// The parameter is the ratio of the cost of intersecting a node (a ray-box intersection)
    /// over the cost of intersecting a primitive.
    BVH_ALWAYS_INLINE AreaHeuristic(T cost_ratio = static_cast<T>(1.))
        : cost_ratio_(cost_ratio)
    {}

    BVH_ALWAYS_INLINE size_t get_cluster_count(size_t prim_count) const {
        return (prim_count + make_bitmask<size_t>(LogClusterSize)) >> LogClusterSize;
    }

    template <size_t N>
    BVH_ALWAYS_INLINE T get_child_cost(size_t prim_count, const BBox<T, N>& bbox) const {
        return bbox.get_half_area() * static_cast<T>(get_cluster_count(prim_count));
    }

    BVH_ALWAYS_INLINE T get_split_cost(T left_child_cost, T right_child_cost) const {
        return left_child_cost + right_child_cost;
    }

    template <size_t N>
    BVH_ALWAYS_INLINE bool should_split(T split_cost, size_t prim_count, const BBox<T, N>& bbox) const {
        return split_cost < bbox.get_half_area() * (static_cast<T>(get_cluster_count(prim_count)) - cost_ratio_);
    }

private:
    T cost_ratio_;
};

} // namespace bvh::v2

#endif
