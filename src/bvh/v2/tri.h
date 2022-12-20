#ifndef BVH_V2_TRI_H
#define BVH_V2_TRI_H

#include "bvh/v2/vec.h"
#include "bvh/v2/ray.h"
#include "bvh/v2/bbox.h"

#include <limits>
#include <utility>
#include <optional>

namespace bvh::v2 {

template <typename T, size_t N>
struct Tri {
    Vec<T, N> p0, p1, p2;

    Tri() = default;

    BVH_ALWAYS_INLINE Tri(const Vec<T, N>& p0, const Vec<T, N>& p1, const Vec<T, N>& p2)
        : p0(p0), p1(p1), p2(p2)
    {}

    BVH_ALWAYS_INLINE BBox<T, N> get_bbox() const { return BBox(p0).extend(p1).extend(p2); }
    BVH_ALWAYS_INLINE Vec<T, N> get_center() const { return (p0 + p1 + p2) * static_cast<T>(1. / 3.); }
	BVH_ALWAYS_INLINE std::pair<BBox<T, N>, BBox<T, N>> split(size_t axis, T pos) const;
};

/// A 3d triangle, represented as two edges and a point, with an (unnormalized, left-handed) normal.
template <typename T>
struct PrecomputedTri {
    Vec<T, 3> p0, e1, e2, n;

    PrecomputedTri() = default;

    BVH_ALWAYS_INLINE PrecomputedTri(const Vec<T, 3>& p0, const Vec<T, 3>& p1, const Vec<T, 3>& p2)
        : p0(p0), e1(p0 - p1), e2(p2 - p0), n(cross(e1, e2))
    {}

    BVH_ALWAYS_INLINE PrecomputedTri(const Tri<T, 3>& triangle)
        : PrecomputedTri(triangle.p0, triangle.p1, triangle.p2)
    {}

    BVH_ALWAYS_INLINE Tri<T, 3> convert_to_tri() const { return Tri<T, 3>(p0, p0 - e1, e2 + p0); }
    BVH_ALWAYS_INLINE BBox<T, 3> get_bbox() const { return convert_to_tri().get_bbox(); }
    BVH_ALWAYS_INLINE Vec<T, 3> get_center() const { return convert_to_tri().get_center(); }

    /// Returns a pair containing the barycentric coordinates of the hit point if the given ray
    /// intersects the triangle, otherwise returns nothing. The distance at which the ray intersects
    /// the triangle is set in `ray.tmax`. The tolerance can be adjusted to account for numerical
    /// precision issues.
    BVH_ALWAYS_INLINE std::optional<std::pair<T, T>> intersect(
        Ray<T, 3>& ray,
        T tolerance = -std::numeric_limits<T>::epsilon()) const;
};

template <typename T>
std::optional<std::pair<T, T>> PrecomputedTri<T>::intersect(Ray<T, 3>& ray, T tolerance) const {
    const auto c = p0 - ray.org;
    const auto r = cross(ray.dir, c);
    const auto inv_det = static_cast<T>(1.) / dot(n, ray.dir);

    const auto u = dot(r, e2) * inv_det;
    const auto v = dot(r, e1) * inv_det;
    const auto w = static_cast<T>(1.) - u - v;

    // These comparisons are designed to return false
    // when one of t, u, or v is a NaN
    if (u >= tolerance && v >= tolerance && w >= tolerance) {
        const auto t = dot(n, c) * inv_det;
        if (t >= ray.tmin && t <= ray.tmax) {
            ray.tmax = t;
            return std::make_optional(std::pair<T, T> { u, v });
        }
    }

    return std::nullopt;
}

template <typename T, size_t N>
std::pair<BBox<T, N>, BBox<T, N>> Tri<T, N>::split(size_t axis, T pos) const {
	auto split_edge = [=] (const Vec<T, N>& a, const Vec<T, N>& b) {
		auto t = (pos - a[axis]) / (b[axis] - a[axis]);
		return a + t * (b - a);
	};

	auto left  = BBox<T, N>::make_empty();
	auto right = BBox<T, N>::make_empty();
	const bool q0 = p0[axis] <= pos;
	const bool q1 = p1[axis] <= pos;
	const bool q2 = p2[axis] <= pos;
	if (q0) left .extend(p0);
	else    right.extend(p0);
	if (q1) left .extend(p1);
	else    right.extend(p1);
	if (q2) left .extend(p2);
	else    right.extend(p2);
	if (q0 ^ q1) {
		auto m = split_edge(p0, p1);
		left.extend(m);
		right.extend(m);
	}
	if (q1 ^ q2) {
		auto m = split_edge(p1, p2);
		left.extend(m);
		right.extend(m);
	}
	if (q2 ^ q0) {
		auto m = split_edge(p2, p0);
		left.extend(m);
		right.extend(m);
	}
	return std::make_pair(left, right);
}

} // namespace bvh::v2

#endif
