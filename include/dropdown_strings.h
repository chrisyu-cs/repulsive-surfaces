#pragma once

#include "remeshing/dynamic_remesher.h"

namespace rsurfaces
{

    template <typename Mode>
    inline std::string StringOfMode(Mode mode)
    {
        throw std::runtime_error("Can't use unspecialized template here.");
    }

    template <>
    inline std::string StringOfMode(GradientMethod mode)
    {
        switch (mode)
        {
        case GradientMethod::HsNCG:
            return "Hs-NCG";
        case GradientMethod::HsProjected:
            return "Hs-Projected";
        case GradientMethod::HsProjectedIterative:
            return "Hs-Projected (iterative)";
        case GradientMethod::HsExactProjected:
            return "Hs-ExactProjected";
        default:
            throw std::runtime_error("Unknown method type.");
        }
    }

    template <>
    inline std::string StringOfMode(remeshing::RemeshingMode mode)
    {
        switch (mode)
        {
        case remeshing::RemeshingMode::FlipOnly:
            return "Flip only";
        case remeshing::RemeshingMode::SmoothAndFlip:
            return "Smooth + flip";
        case remeshing::RemeshingMode::SmoothFlipAndCollapse:
            return "Smooth + flip + collapse";
        default:
            throw std::runtime_error("Unknown remeshing mode.");
        }
    }

    template <>
    inline std::string StringOfMode(remeshing::SmoothingMode mode)
    {
        switch (mode)
        {
        case remeshing::SmoothingMode::Circumcenter:
            return "Circumcenter";
        case remeshing::SmoothingMode::Laplacian:
            return "Laplacian";
        default:
            throw std::runtime_error("Unknown smoothing mode.");
        }
    }

    template <>
    inline std::string StringOfMode(remeshing::FlippingMode mode)
    {
        switch (mode)
        {
        case remeshing::FlippingMode::Delaunay:
            return "Delaunay";
        case remeshing::FlippingMode::Degree:
            return "Degree";
        default:
            throw std::runtime_error("Unknown flipping mode.");
        }
    }
} // namespace rsurfaces