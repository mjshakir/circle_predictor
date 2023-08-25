//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Utilities/CircleEquation.hpp"
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <algorithm>
#include <execution>
#include <limits>
//--------------------------------------------------------------
// Boost library
//--------------------------------------------------------------
#include <boost/multiprecision/cpp_dec_float.hpp>
//--------------------------------------------------------------
// public 
//--------------------------
torch::Tensor Utils::CircleEquation::distance_reward(const torch::Tensor& input, const torch::Tensor& output){
    //--------------------------
    return compute_distance_reward(input, output);
    //--------------------------
}// end torch::Tensor Utils::CircleEquation::distance_reward(const torch::Tensor& input, const torch::Tensor& output)
//--------------------------------------------------------------
torch::Tensor Utils::CircleEquation::diversity_reward(const torch::Tensor& output, const torch::Tensor& input){
    //--------------------------
    return compute_diversity_reward(input, output);
    //--------------------------
}// end torch::Tensor Utils::CircleEquation::diversity_reward(const torch::Tensor& output, const torch::Tensor& input)
//--------------------------------------------------------------
torch::Tensor Utils::CircleEquation::consistency_reward(const torch::Tensor& output){
    //--------------------------
    return compute_consistency_reward(output);
    //--------------------------
}// end torch::Tensor Utils::CircleEquation::consistency_reward(const torch::Tensor& output)
//--------------------------------------------------------------
torch::Tensor Utils::CircleEquation::distance_penalty(const torch::Tensor& input, const torch::Tensor& output){
    //--------------------------
    return compute_distance_penalty(input, output);
    //--------------------------
}// end torch::Tensor Utils::CircleEquation::distance_penalty(const torch::Tensor& input, const torch::Tensor& output)
//--------------------------------------------------------------
torch::Tensor Utils::CircleEquation::separation_penalty(const torch::Tensor& output, const double& min_distance){
    //--------------------------
    return compute_point_separation_penalty(output, min_distance);
    //--------------------------
}// end torch::Tensor Utils::CircleEquation::separation_penalty(const torch::Tensor& output, const double& min_distance)
//--------------------------------------------------------------
void Utils::CircleEquation::Aligned(torch::Tensor& reward, const torch::Tensor& points, const double& tolerance){
    //--------------------------
    arePointsAligned(reward, points, tolerance);
    //--------------------------
}// end void Utils::CircleEquation::Aligned(torch::Tensor& reward, const torch::Tensor& points, const double& tolerance)
//--------------------------------------------------------------
bool Utils::CircleEquation::Aligned(const torch::Tensor& points, const double& tolerance){
    //--------------------------
    return arePointsAligned(points, tolerance);
    //--------------------------
}//end torch::Tensor Utils::CircleEquation::Aligned(const torch::Tensor& points, const double& tolerance)
//--------------------------------------------------------------
void Utils::CircleEquation::Aligned(double& reward, const torch::Tensor& points, const double& tolerance){
    //--------------------------
    arePointsAligned(reward, points, tolerance);
    //--------------------------
}// end void Utils::CircleEquation::Aligned(double& reward, const torch::Tensor& points, const double& tolerance)
//--------------------------------------------------------------
void Utils::CircleEquation::CloseToCircumference(torch::Tensor& reward, const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius, const double& tolerance){
    //--------------------------
    arePointsCloseToCircumference(reward, points, center, radius, tolerance);
    //--------------------------
}// end void Utils::CircleEquation::CloseToCircumference(torch::Tensor& reward, const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius, const double& tolerance)
//--------------------------------------------------------------
bool Utils::CircleEquation::CloseToCircumference(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius, const double& tolerance){
    //--------------------------
    return arePointsCloseToCircumference(points, center, radius, tolerance);
    //--------------------------
}// end torch::Tensor Utils::CircleEquation::CloseToCircumference(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius, const double& tolerance)
//--------------------------------------------------------------
void Utils::CircleEquation::CloseToCircumference(double& reward, const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius, const double& tolerance){
    //--------------------------
    arePointsCloseToCircumference(reward, points, center, radius, tolerance);
    //--------------------------
}// end void Utils::CircleEquation::CloseToCircumference(double& reward, const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius, const double& tolerance)
//--------------------------------------------------------------
void Utils::CircleEquation::Equidistant(torch::Tensor& reward, const torch::Tensor& points, const double& tolerance){
    //--------------------------
    arePointsEquidistant(reward, points, tolerance);
    //--------------------------
}// end void Utils::CircleEquation::Equidistant(torch::Tensor& reward, const torch::Tensor& points, double tolerance)
//--------------------------------------------------------------
bool Utils::CircleEquation::Equidistant(const torch::Tensor& points, const double& tolerance){
    //--------------------------
    return arePointsEquidistant(points, tolerance);
    //--------------------------
}// end torch::Tensor Utils::CircleEquation::Equidistant(const torch::Tensor& points, const double& tolerance)
//--------------------------------------------------------------
void Utils::CircleEquation::Equidistant(double& reward, const torch::Tensor& points, const double& tolerance){
    //--------------------------
    arePointsEquidistant(reward, points, tolerance);
    //--------------------------
}// end void Utils::CircleEquation::Equidistant(double& reward, const torch::Tensor& points, const double& tolerance)
//--------------------------------------------------------------
void Utils::CircleEquation::AngleRatiosConsistent(torch::Tensor& reward, const torch::Tensor& points, const double& tolerance){
    //--------------------------
    areAngleRatiosConsistent(reward, points, tolerance);
    //--------------------------
}//end void Utils::CircleEquation::AngleRatiosConsistent(torch::Tensor& reward, const torch::Tensor& points, const double& tolerance)
//--------------------------------------------------------------
bool Utils::CircleEquation::AngleRatiosConsistent(const torch::Tensor& points, const double& tolerance){
    //--------------------------
    return areAngleRatiosConsistent(points, tolerance);
    //--------------------------
}//end torch::Tensor Utils::CircleEquation::AngleRatiosConsistent(const torch::Tensor& points, const double& tolerance)
//--------------------------------------------------------------
void Utils::CircleEquation::AngleRatiosConsistent(double& reward, const torch::Tensor& points, const double& tolerance){
    //--------------------------
    areAngleRatiosConsistent(reward, points, tolerance);
    //--------------------------
}//end void Utils::CircleEquation::AngleRatiosConsistent(double& reward, const torch::Tensor& points, const double& tolerance)
//--------------------------------------------------------------
void Utils::CircleEquation::Symmetric(torch::Tensor& reward, const torch::Tensor& points){
    //--------------------------
    arePointsSymmetric(reward, points);
    //--------------------------
}// end void Utils::CircleEquation::Symmetric(torch::Tensor& reward, const torch::Tensor& points)
//--------------------------------------------------------------
bool Utils::CircleEquation::Symmetric(const torch::Tensor& points){
    //--------------------------
    return arePointsSymmetric(points);
    //--------------------------
}// end void Utils::CircleEquation::Symmetric(torch::Tensor& reward, const torch::Tensor& points)
//--------------------------------------------------------------
void Utils::CircleEquation::Symmetric(double& reward, const torch::Tensor& points){
    //--------------------------
    arePointsSymmetric(reward, points);
    //--------------------------
}// end void Utils::CircleEquation::Symmetric(double& reward, const torch::Tensor& points)
//--------------------------------------------------------------
torch::Tensor Utils::CircleEquation::TriangleArea(const torch::Tensor& points, const long double& tolerance){
    //--------------------------
    return calculateTriangleArea(points, tolerance);
    //--------------------------
}// end torch::Tensor Utils::CircleEquation::TriangleArea(const torch::Tensor& points)
//--------------------------------------------------------------
void Utils::CircleEquation::TriangleArea(double& reward, const torch::Tensor& points, const long double& tolerance){
    //--------------------------
    calculateTriangleArea(reward, points, tolerance);
    //--------------------------
}// end torch::Tensor Utils::CircleEquation::TriangleArea(const torch::Tensor& points)
//--------------------------------------------------------------
torch::Tensor Utils::CircleEquation::CircleSmoothness(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius){
    //--------------------------
    return calculateCircleSmoothness(points, center, radius);
    //--------------------------    
}// end torch::Tensor Utils::CircleEquation::CircleSmoothness(const torch::Tensor& points, const torch::Tensor& center, torch::Tensor radius)
//--------------------------------------------------------------
void Utils::CircleEquation::CircleSmoothness(double& reward, const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius){
    //--------------------------
    calculateCircleSmoothness(reward, points, center, radius);
    //--------------------------    
}// end torch::Tensor Utils::CircleEquation::CircleSmoothness(const torch::Tensor& points, const torch::Tensor& center, torch::Tensor radius)
//--------------------------------------------------------------
torch::Tensor Utils::CircleEquation::PointLimiter(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius){
    //--------------------------
    return getMaxPointLimiter(points, center, radius);
    //--------------------------
}// end torch::Tensor Utils::CircleEquation::PointLimiter(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius)
//--------------------------------------------------------------
void Utils::CircleEquation::PointLimiter(double& reward, const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius){
    //--------------------------
    getMaxPointLimiter(reward, points, center, radius);
    //--------------------------
}// end torch::Tensor Utils::CircleEquation::PointLimiter(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius)
//--------------------------------------------------------------
bool Utils::CircleEquation::Distinct(const torch::Tensor& point1, const torch::Tensor& point2){
    //--------------------------
    return PointsDistinct(point1, point2);
    //--------------------------
}// end bool Utils::CircleEquation::Distinct(const torch::Tensor& point1, const torch::Tensor& point2)
//--------------------------------------------------------------
void Utils::CircleEquation::Distinct(double& reward, const torch::Tensor& point1, const torch::Tensor& point2){
    //--------------------------
    PointsDistinct(reward, point1, point2);
    //--------------------------
}// end bool Utils::CircleEquation::Distinct(const torch::Tensor& point1, const torch::Tensor& point2)
//--------------------------------------------------------------
// protected
//--------------------------
torch::Tensor Utils::CircleEquation::compute_distance_reward(const torch::Tensor& input, const torch::Tensor& output){
    
    torch::Tensor distances_squared;
    // Ensuring dimensions are compatible
    auto input_dims = input.dim();
    auto output_dims = output.dim();

    if (input_dims != 2 || input.size(1) < 3) {
        throw std::runtime_error("Expected input size: [batch_size, 3], found: " + std::to_string(input_dims));
    }
    
    if (output_dims == 2) {
        // Adjusting for the case where output is [batch_size, 2]
        torch::Tensor x_diff = output.select(1, 0) - input.select(1, 0);
        torch::Tensor y_diff = output.select(1, 1) - input.select(1, 1);
        distances_squared = x_diff.square() + y_diff.square();
    } else if (output_dims == 3) {
        // Adjusting for the case where output is [batch_size, points_size, 2]
        torch::Tensor x_diff = output.slice(2, 0, 1).squeeze() - input.select(1, 0).unsqueeze(-1);
        torch::Tensor y_diff = output.slice(2, 1, 2).squeeze() - input.select(1, 1).unsqueeze(-1);
        distances_squared = x_diff.square() + y_diff.square();
    } else {
        throw std::runtime_error("Unexpected output dimensions: " + std::to_string(output_dims));
    }
    
    torch::Tensor r_diffs = distances_squared - input.select(1, 2).unsqueeze(-1);
    torch::Tensor abs_errors = r_diffs.abs();
    return abs_errors.mean(-1);
}// end torch::Tensor Utils::CircleEquation::computeDistanceReward(const torch::Tensor& input, const torch::Tensor& output)
//--------------------------------------------------------------
torch::Tensor Utils::CircleEquation::compute_diversity_reward(const torch::Tensor& output, const torch::Tensor& input) {
    torch::Tensor y_diff = output.slice(-1, 1, 2).squeeze() - input.slice(1, 1, 2).unsqueeze(-1);
    torch::Tensor x_diff = output.slice(-1, 0, 1).squeeze() - input.slice(1, 0, 1).unsqueeze(-1);
    torch::Tensor angles = torch::atan2(y_diff, x_diff);
    // Placeholder. Adjust based on your specific diversity metric.
    return torch::ones({angles.size(0)});
}
//--------------------------------------------------------------
torch::Tensor Utils::CircleEquation::compute_consistency_reward(const torch::Tensor& output) {
    torch::Tensor x_diff_subsequent = output.slice(-1, 0, 1).squeeze() - torch::roll(output.slice(-1, 0, 1).squeeze(), 1, 1);
    torch::Tensor y_diff_subsequent = output.slice(-1, 1, 2).squeeze() - torch::roll(output.slice(-1, 1, 2).squeeze(), 1, 1);
    torch::Tensor subsequent_distances = torch::sqrt(x_diff_subsequent.square() + y_diff_subsequent.square());
    // Placeholder. Adjust based on your specific consistency metric.
    return torch::ones({subsequent_distances.size(0)});
}
//--------------------------------------------------------------
torch::Tensor Utils::CircleEquation::compute_distance_penalty(const torch::Tensor& input, const torch::Tensor& output){
    // Compute squared distances between points and circle center
    torch::Tensor circle_center = input.slice(1, 0, 2);  // Extract (X,Y) center
    torch::Tensor distances_squared = torch::sum((output - circle_center.unsqueeze(1)).square(), -1);
    
    // Compute penalty as the squared distance minus radius squared
    torch::Tensor penalty = torch::relu(distances_squared - input.select(1, 2).unsqueeze(-1));
    
    // Apply a ReLU to ensure penalties are non-negative
    // penalty = torch::clamp_min(penalty, 0.0);
    // penalty = torch::relu(penalty);


    // Compute the mean penalty over all points
    return penalty.mean(-1);
}// end torch::Tensor Utils::CircleEquation::compute_distance_penalty(const torch::Tensor& input, const torch::Tensor& output)
//--------------------------------------------------------------
torch::Tensor Utils::CircleEquation::compute_point_separation_penalty(const torch::Tensor& output, const double& min_distance){
    // Compute pairwise distances between points
    torch::Tensor diff = output.unsqueeze(1) - output.unsqueeze(2);
    torch::Tensor distances_squared = torch::sum(diff.square(), -1);

    // Create a mask to ignore comparisons between the same points
    torch::Tensor same_point_mask = torch::eye(output.size(1)).unsqueeze(0).expand_as(distances_squared);
    distances_squared.masked_fill_(same_point_mask, 1.0);  // Replace diagonal with 1.0 to avoid division by zero

    // Compute point separation penalty as the reciprocal of the minimum pairwise distance
    torch::Tensor min_distances;
    std::tie(min_distances, std::ignore) = torch::min(distances_squared, -1);
    min_distances = torch::clamp_min(min_distances, 1e-6); // Adjust the threshold as needed
    min_distances = torch::sqrt(min_distances);
    // torch::Tensor penalty = torch::clamp_min(min_distance / min_distances, 0.0);
    torch::Tensor penalty = torch::relu(min_distance / min_distances);

    // Compute the mean penalty over all points
    return penalty.mean(-1);
}
//--------------------------------------------------------------
void Utils::CircleEquation::arePointsAligned(torch::Tensor& reward, const torch::Tensor& points, const double& tolerance){
    //--------------------------
    if (points.size(0) < 2){
        //--------------------------
        reward += 1.;
        //--------------------------
        return;
        //--------------------------
    }// end if (points.size(0) < 2)
    //--------------------------
    torch::Tensor x = points.select(1, 0);
    torch::Tensor y = points.select(1, 1);
    //--------------------------
    torch::Tensor xDiff = x[1] - x[0];
    torch::Tensor yDiff = y[1] - y[0];
    //--------------------------
    reward[0] = torch::tensor(1.);
    //--------------------------
    at::parallel_for(1, points.size(0), 1, [&](int64_t start, int64_t end) {
        //--------------------------
        for (int64_t i = start; i < end; ++i) {
            //--------------------------
            if (torch::abs(((xDiff * (x[i] - x[0])) - (yDiff * (y[i] - y[0])))).item<double>() <= tolerance) {
                //--------------------------
                torch::add(reward.select(0, i), 1.);
                //--------------------------
            }// end if (torch::abs(((xDiff * (x[i] - x[0])) - (yDiff * (y[i] - y[0])))).item<double>() <= tolerance)
            //--------------------------
        }// end for (int64_t i = start; i < end; ++i)
        //--------------------------
    });
    //--------------------------
}// end void Utils::CircleEquation::arePointsAligned(torch::Tensor& reward, const torch::Tensor& center_points, const torch::Tensor& points, double tolerance)
//--------------------------------------------------------------
/*
torch::Tensor Utils::CircleEquation::arePointsAligned(const torch::Tensor& points, const double& tolerance) {
    if (points.size(0) < 2){
        //--------------------------
        return torch::ones({points.size(0)}, torch::kBool);
        //--------------------------
    }// end if (points.size(0) < 2)
    //--------------------------
    torch::Tensor x = points.select(1, 0);
    torch::Tensor y = points.select(1, 1);
    //--------------------------
    torch::Tensor xDiff = x[1] - x[0];
    torch::Tensor yDiff = y[1] - y[0];
    //--------------------------
    auto aligned = torch::ones({points.size(0)}, torch::kBool);
    //--------------------------
    at::parallel_for(1, points.size(0), 1, [&](int64_t start, int64_t end) {
        //--------------------------
        for (int64_t i = start; i < end; ++i) {
            //--------------------------
            torch::Tensor currentXDiff = x[i] - x[0];
            torch::Tensor currentYDiff = y[i] - y[0];
            //--------------------------
            torch::Tensor diff = xDiff * currentYDiff - yDiff * currentXDiff;
            //--------------------------
            if (torch::abs(diff).item<double>() > tolerance) {
                //--------------------------
                aligned[i] = false;
                //--------------------------
            }// end if (torch::abs(diff) > tolerance)
            //--------------------------
        }// end or (int64_t i = start; i < end; ++i)
        //--------------------------
    });
    //--------------------------
    return aligned;
    //--------------------------
}// end torch::Tensor Utils::CircleEquation::arePointsAligned(const torch::Tensor& points, const double& tolerance)
*/
//--------------------------------------------------------------
bool Utils::CircleEquation::arePointsAligned(const torch::Tensor& points, const double& tolerance) {
    //--------------------------
    if (points.size(0) < 2){
        //--------------------------
        return true;
        //--------------------------
    }// end if (points.size(0) < 2)
    //--------------------------
    torch::Tensor x = points.select(1, 0);
    torch::Tensor y = points.select(1, 1);
    //--------------------------
    torch::Tensor xDiff = x[1] - x[0];
    torch::Tensor yDiff = y[1] - y[0];
    //--------------------------
    bool aligned{true};
    //--------------------------
    at::parallel_for(1, points.size(0), 1, [&](int64_t start, int64_t end) {
        //--------------------------
        for (int64_t i = start; i < end; ++i) {
            //--------------------------
            if (torch::abs(((xDiff * (x[i] - x[0])) - (yDiff * (y[i] - y[0])))).item<double>() > tolerance) {
                //--------------------------
                aligned = false;
                //--------------------------
                break;
                //--------------------------
            }// end if (torch::abs(diff) > tolerance)
            //--------------------------
        }// end or (int64_t i = start; i < end; ++i)
        //--------------------------
    });
    //--------------------------
    return aligned;
    //--------------------------
}// end bool Utils::CircleEquation::arePointsAligned(const torch::Tensor& points, const double& tolerance)
//--------------------------------------------------------------
// void Utils::CircleEquation::arePointsAligned(double& reward, const torch::Tensor& points, const double& tolerance){
//     //--------------------------
//     if (points.size(0) < 2){
//         //--------------------------
//         reward += 1.;
//         //--------------------------
//         return;
//         //--------------------------
//     }// end if (points.size(0) < 2)
//     //--------------------------
//     torch::Tensor x = points.select(1, 0);
//     torch::Tensor y = points.select(1, 1);
//     //--------------------------
//     torch::Tensor xDiff = x[1] - x[0];
//     torch::Tensor yDiff = y[1] - y[0];
//     //--------------------------
//     double aligned{1.};
//     //--------------------------
//     at::parallel_for(1, points.size(0), 1, [&](int64_t start, int64_t end) {
//         //--------------------------
//         for (int64_t i = start; i < end; ++i) {
//             //--------------------------
//             if (torch::abs(((xDiff * (x[i] - x[0])) - (yDiff * (y[i] - y[0])))).item<double>() > tolerance) {
//                 //--------------------------
//                 aligned = 0.;
//                 //--------------------------
//                 break;
//                 //--------------------------
//             }// end if (torch::abs(diff) > tolerance)
//             //--------------------------
//         }// end or (int64_t i = start; i < end; ++i)
//         //--------------------------
//     });
//     //--------------------------
//     reward += aligned;
//     //--------------------------
// }// end void Utils::CircleEquation::arePointsAligned(double& reward, const torch::Tensor& points, const double& tolerance)
//--------------------------------------------------------------
void Utils::CircleEquation::arePointsAligned(double& reward, const torch::Tensor& points, const double& tolerance) {
    if (points.size(0) < 2) {
        reward += 1.;
        return;
    }

    torch::Tensor x = points.select(1, 0);
    torch::Tensor y = points.select(1, 1);
    torch::Tensor xDiff = x[1] - x[0];
    torch::Tensor yDiff = y[1] - y[0];

    std::atomic<bool> isAligned(true);

    at::parallel_for(1, points.size(0), 1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end && isAligned.load(); ++i) {
            if (torch::abs((xDiff * (x[i] - x[0])) - (yDiff * (y[i] - y[0]))).item<double>() > tolerance) {
                isAligned.store(false);
                break;
            }
        }
    });

    reward += (isAligned.load() ? 1. : 0.);
}
//--------------------------------------------------------------
void Utils::CircleEquation::arePointsCloseToCircumference(torch::Tensor& reward, const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius, const double& tolerance){
    //--------------------------
    torch::Tensor distances = torch::hypot(points.select(1, 0) - center.select(1, 0), points.select(1, 1) - center.select(1, 1));
    torch::Tensor diff = torch::abs(distances - radius);
    //--------------------------
    at::parallel_for(1, points.size(0), 1, [&](int64_t start, int64_t end) {
        //--------------------------
        for (int64_t i = start; i < end; ++i) {
            //--------------------------
            if (diff[i].item<double>() <= tolerance) {
                //--------------------------
                reward[i] += 1.;
                //--------------------------
            }// end if (diff[i].item<double>() <= tolerance)
            //--------------------------
        }// end or (int64_t i = start; i < end; ++i)
        //--------------------------
    });
    //--------------------------
}// void Utils::CircleEquation::arePointsCloseToCircumference(torch::Tensor& reward, const torch::Tensor& center, const torch::Tensor& radius, const double& tolerance)
//--------------------------------------------------------------
/*
torch::Tensor Utils::CircleEquation::arePointsCloseToCircumference(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius, const double& tolerance){
    //--------------------------
    auto close = torch::ones({points.size(0)}, torch::kBool);
    //--------------------------
    torch::Tensor distances = torch::hypot(points.select(1, 0) - center.select(1, 0), points.select(1, 1) - center.select(1, 1));
    torch::Tensor diff = torch::abs(distances - radius);
    //--------------------------
    at::parallel_for(1, points.size(0), 1, [&](int64_t start, int64_t end) {
        //--------------------------
        for (int64_t i = start; i < end; ++i) {
            //--------------------------
            if (diff[i].item<double>() > tolerance) {
                //--------------------------
                close[i] = false;
                //--------------------------
            }// end if (torch::abs(diff) > tolerance)
            //--------------------------
        }// end or (int64_t i = start; i < end; ++i)
        //--------------------------
    });
    //--------------------------
    return close;
}// end torch::Tensor Utils::CircleEquation::arePointsCloseToCircumference(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius, const double& tolerance)
*/
//--------------------------------------------------------------
bool Utils::CircleEquation::arePointsCloseToCircumference(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius, const double& tolerance){
    //--------------------------
    if (points.size(0) < 3) {
        //--------------------------
        return true;
        //--------------------------
    }// end if (points.size(0) < 3)
    //--------------------------
    torch::Tensor distances = torch::hypot(points.select(1, 0) - center.select(1, 0), points.select(1, 1) - center.select(1, 1));
    torch::Tensor diff = torch::abs(distances - radius);
    //--------------------------
    bool close{true};
    //--------------------------
    at::parallel_for(1, points.size(0), 1, [&](int64_t start, int64_t end) {
        //--------------------------
        for (int64_t i = start; i < end; ++i) {
            //--------------------------
            if (diff[i].item<double>() > tolerance) {
                //--------------------------
                close = false;
                //--------------------------
                break;
                //--------------------------
            }// end if (torch::abs(diff) > tolerance)
            //--------------------------
        }// end or (int64_t i = start; i < end; ++i)
        //--------------------------
    });
    //--------------------------
    return close;
    //--------------------------
}// end bool Utils::CircleEquation::arePointsCloseToCircumference(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius, const double& tolerance)
//--------------------------------------------------------------
// void Utils::CircleEquation::arePointsCloseToCircumference(  double& reward,
//                                             const torch::Tensor& points,
//                                             const torch::Tensor& center,
//                                             const torch::Tensor& radius,
//                                             const double& tolerance){
//     //--------------------------
//     if (points.size(0) < 3) {
//         //--------------------------
//         reward += 1.;
//         //--------------------------
//         return;
//         //--------------------------
//     }// end if (points.size(0) < 3)
//     //--------------------------
//     torch::Tensor distances = torch::hypot(points.select(1, 0) - center.select(1, 0), points.select(1, 1) - center.select(1, 1));
//     torch::Tensor diff = torch::abs(distances - radius);
//     //--------------------------
//     double close{1.};
//     //--------------------------
//     at::parallel_for(1, points.size(0), 1, [&](int64_t start, int64_t end) {
//         //--------------------------
//         for (int64_t i = start; i < end; ++i) {
//             //--------------------------
//             if (diff[i].item<double>() > tolerance) {
//                 //--------------------------
//                 close = 0.;
//                 //--------------------------
//                 break;
//                 //--------------------------
//             }// end if (torch::abs(diff) > tolerance)
//             //--------------------------
//         }// end or (int64_t i = start; i < end; ++i)
//         //--------------------------
//     });
//     //--------------------------
//     reward += close;
//     //--------------------------
// }// end void Utils::CircleEquation::arePointsCloseToCircumference
//--------------------------------------------------------------
void Utils::CircleEquation::arePointsCloseToCircumference(double& reward,
                                                          const torch::Tensor& points,
                                                          const torch::Tensor& center,
                                                          const torch::Tensor& radius,
                                                          const double& tolerance) {

    if (points.size(0) < 3) {
        reward += 1.;
        return;
    }

    torch::Tensor distances = torch::hypot(points.select(1, 0) - center.select(1, 0), 
                                           points.select(1, 1) - center.select(1, 1));
    torch::Tensor diff = torch::abs(distances - radius);

    std::atomic<bool> close(true);

    at::parallel_for(1, points.size(0), 1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end && close.load(); ++i) {
            if (diff[i].item<double>() > tolerance) {
                close.store(false);
                break;
            }
        }
    });

    reward += close.load() ? 1. : 0.;
}
//--------------------------------------------------------------
void Utils::CircleEquation::arePointsEquidistant(torch::Tensor& reward, const torch::Tensor& points, const double& tolerance){
    //--------------------------
    if (points.size(0) < 3){
        //--------------------------
        reward += 1.;
        //--------------------------
        return;
        //--------------------------
    }// end if (points.size(0) < 3)
    //--------------------------
    torch::Tensor referencePoint = points[0];
    torch::Tensor distance = torch::hypot(points.slice(/*dim=*/1, /*start=*/0, /*end=*/1) - referencePoint[0], points.slice(/*dim=*/1, /*start=*/1, /*end=*/2) - referencePoint[1]);
    //--------------------------
    at::parallel_for(2, points.size(0), 1, [&](int64_t start, int64_t end) {
        //--------------------------
        for (int64_t i = start; i < end; ++i) {
            //--------------------------
            torch::Tensor currentDistance = torch::hypot(points[i][0] - referencePoint[0], points[i][1] - referencePoint[1]);
            //--------------------------
            if (torch::abs(currentDistance - distance[i]).item<double>() <= tolerance) {
                //--------------------------
                reward[i] += 1.;
                //--------------------------
            }// end if (torch::abs(currentDistance - distance).item<double>() > tolerance)
            //--------------------------
        }// end for (int64_t i = start; i < end; ++i)
        //--------------------------
    });
    //--------------------------
}// end void Utils::CircleEquation::arePointsEquidistant(torch::Tensor& reward, const torch::Tensor& points, const double& tolerance)
//--------------------------------------------------------------
/*
torch::Tensor Utils::CircleEquation::arePointsEquidistant(const torch::Tensor& points, const double& tolerance){
    //--------------------------
    if (points.size(0) < 3){
        //--------------------------
        return torch::ones({points.size(0)}, torch::kBool);
        //--------------------------
    }// end if (points.size(0) < 3)
    //--------------------------
    torch::Tensor referencePoint = points[0];
    torch::Tensor distance = torch::hypot(points.slice(1, 0, 1) - referencePoint[0], points.slice(1, 1, 2) - referencePoint[1]);
    //--------------------------
    auto equidistant = torch::ones({points.size(0)}, torch::kBool);
    //--------------------------
    at::parallel_for(2, points.size(0), 1, [&](int64_t start, int64_t end) {
        //--------------------------
        for (int64_t i = start; i < end; ++i) {
            //--------------------------
            torch::Tensor currentDistance = torch::hypot(points[i][0] - referencePoint[0], points[i][1] - referencePoint[1]);
            //--------------------------
            if (torch::abs(currentDistance - distance).item<double>() > tolerance) {
                //--------------------------
                equidistant[i] = false;
                //--------------------------
            }// end if (torch::abs(currentDistance - distance).item<double>() > tolerance)
            //--------------------------
        }// end for (int64_t i = start; i < end; ++i)
        //--------------------------
    });
    //--------------------------
    return equidistant;
    //--------------------------
}// end torch::Tensor Utils::CircleEquation::arePointsEquidistant(const torch::Tensor& points, const double& tolerance)
*/
//--------------------------------------------------------------
bool Utils::CircleEquation::arePointsEquidistant(const torch::Tensor& points, const double& tolerance){
    //--------------------------
    if (points.size(0) < 3){
        //--------------------------
        return true;
        //--------------------------
    }// end if (points.size(0) < 3)
    //--------------------------
    torch::Tensor referencePoint = points[0];
    torch::Tensor distance = torch::hypot(points.slice(/*dim=*/1, /*start=*/0, /*end=*/1) - referencePoint[0], points.slice(/*dim=*/1, /*start=*/1, /*end=*/2) - referencePoint[1]);
    //--------------------------
    bool equidistant{true};
    //--------------------------
    at::parallel_for(2, points.size(0), 1, [&](int64_t start, int64_t end) {
        //--------------------------
        for (int64_t i = start; i < end; ++i) {
            //--------------------------
            torch::Tensor currentDistance = torch::hypot(points[i][0] - referencePoint[0], points[i][1] - referencePoint[1]);
            //--------------------------
            if (torch::abs(currentDistance - distance[i]).item<double>() > tolerance) {
                //--------------------------
                equidistant = false;
                //--------------------------
                break;
                //--------------------------
            }// end if (torch::abs(currentDistance - distance).item<double>() > tolerance)
            //--------------------------
        }// end for (int64_t i = start; i < end; ++i)
        //--------------------------
    });
    //--------------------------
    return equidistant;
    //--------------------------
}// end bool Utils::CircleEquation::arePointsEquidistant(const torch::Tensor& points, const double& tolerance)
//--------------------------------------------------------------
// void Utils::CircleEquation::arePointsEquidistant(double& reward, const torch::Tensor& points, const double& tolerance){
//     //--------------------------
//     if (points.size(0) < 3){
//         //--------------------------
//         reward += 1.;
//         //--------------------------
//         return;
//         //--------------------------
//     }// end if (points.size(0) < 3)
//     //--------------------------
//     torch::Tensor referencePoint = points[0];
//     torch::Tensor distance = torch::hypot(points.slice(/*dim=*/1, /*start=*/0, /*end=*/1) - referencePoint[0], points.slice(/*dim=*/1, /*start=*/1, /*end=*/2) - referencePoint[1]);
//     //--------------------------
//     double equidistant{1.};
//     //--------------------------
//     at::parallel_for(2, points.size(0), 1, [&](int64_t start, int64_t end) {
//         //--------------------------
//         for (int64_t i = start; i < end; ++i) {
//             //--------------------------
//             torch::Tensor currentDistance = torch::hypot(points[i][0] - referencePoint[0], points[i][1] - referencePoint[1]);
//             //--------------------------
//             if (torch::abs(currentDistance - distance[i]).item<double>() > tolerance) {
//                 //--------------------------
//                 equidistant = 0.;
//                 //--------------------------
//                 break;
//                 //--------------------------
//             }// end if (torch::abs(currentDistance - distance).item<double>() > tolerance)
//             //--------------------------
//         }// end for (int64_t i = start; i < end; ++i)
//         //--------------------------
//     });
//     //--------------------------
//     reward += equidistant;
//     //--------------------------
// }// end void Utils::CircleEquation::arePointsEquidistant(double& reward, const torch::Tensor& points, const double& tolerance)
//--------------------------------------------------------------
void Utils::CircleEquation::arePointsEquidistant(double& reward, const torch::Tensor& points, const double& tolerance) {
    if (points.size(0) < 3) {
        reward += 1.;
        return;
    }

    torch::Tensor referencePoint = points[0];
    double referenceDistance = torch::hypot(points[1][0] - referencePoint[0], points[1][1] - referencePoint[1]).item<double>();

    std::atomic<bool> equidistant(true);

    at::parallel_for(2, points.size(0), 1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end && equidistant.load(); ++i) {
            double currentDistance = torch::hypot(points[i][0] - referencePoint[0], points[i][1] - referencePoint[1]).item<double>();
            
            if (std::abs(currentDistance - referenceDistance) > tolerance) {
                equidistant.store(false);
            }
        }
    });

    reward += equidistant.load() ? 1. : 0.;
}

//--------------------------------------------------------------
void Utils::CircleEquation::areAngleRatiosConsistent(torch::Tensor& reward, const torch::Tensor& points, const double& tolerance){
    //--------------------------
    if (points.size(0) < 3){
        //--------------------------
        reward += 1.;
        //--------------------------
        return;
        //--------------------------
    }// end if (points.size(0) < 3)
    //--------------------------
    torch::Tensor xDiff = points[1][0] - points[0][0];
    torch::Tensor yDiff = points[1][1] - points[0][1];
    torch::Tensor ratio = xDiff / yDiff;
    //--------------------------
    at::parallel_for(2, points.size(0), 1, [&](int64_t start, int64_t end) {
        //--------------------------
        for (int64_t i = start; i < end; ++i) {
            //--------------------------
            torch::Tensor currentXDiff = points[i][0] - points[0][0];
            torch::Tensor currentYDiff = points[i][1] - points[0][1];
            torch::Tensor currentRatio = currentXDiff / currentYDiff;
            //--------------------------
            if (torch::abs(currentRatio - ratio).item<double>() <= tolerance) {
                //--------------------------
                reward[i] += 1.;
                //--------------------------
            }// end if (torch::abs(currentRatio - ratio).item<double>() <= tolerance)
            //--------------------------
        }// end for (int64_t i = start; i < end; ++i)
        //--------------------------
    });
    //--------------------------
}// end void Utils::CircleEquation::areAngleRatiosConsistent(torch::Tensor& reward, const torch::Tensor& points, const double& tolerance)
//--------------------------------------------------------------
/*
torch::Tensor Utils::CircleEquation::areAngleRatiosConsistent(const torch::Tensor& points, const double& tolerance){
    //--------------------------
    if (points.size(0) < 3){
        //--------------------------
        return torch::ones({points.size(0)}, torch::kBool);
        //--------------------------
    }// end if (points.size(0) < 3)
    //--------------------------
    torch::Tensor xDiff = points[1][0] - points[0][0];
    torch::Tensor yDiff = points[1][1] - points[0][1];
    torch::Tensor ratio = xDiff / yDiff;
    //--------------------------
    // auto consistent = torch::ones({points.size(0)}, torch::kBool);
    // //--------------------------
    // at::parallel_for(2, points.size(0), 1, [&](int64_t start, int64_t end) {
    //     //--------------------------
    //     for (int64_t i = start; i < end; ++i) {
    //         //--------------------------
    //         torch::Tensor currentXDiff = points[i][0] - points[0][0];
    //         torch::Tensor currentYDiff = points[i][1] - points[0][1];
    //         torch::Tensor currentRatio = currentXDiff / currentYDiff;
    //         //--------------------------
    //         if (torch::abs(currentRatio - ratio).item<double>() > tolerance) {
    //             //--------------------------
    //             consistent[i] = false;
    //             //--------------------------
    //         }// end if (torch::abs(currentRatio - ratio) > tolerance)
    //         //--------------------------
    //     }// end for (int64_t i = start; i < end; ++i)
    //     //--------------------------
    // });
    //--------------------------
    torch::Tensor currentXDiff = points.slice(1).select(1, 0) - points[0][0];
    torch::Tensor currentYDiff = points.slice(1).select(1, 1) - points[0][1];
    torch::Tensor currentRatio = currentXDiff / currentYDiff;
    //--------------------------
    torch::Tensor consistent = torch::abs(currentRatio - ratio) <= tolerance;
    //--------------------------
    return consistent;
    //--------------------------
}// end torch::Tensor Utils::CircleEquation::areAngleRatiosConsistent(const torch::Tensor& points, const double& tolerance)
*/
//--------------------------------------------------------------
bool Utils::CircleEquation::areAngleRatiosConsistent(const torch::Tensor& points, const double& tolerance){
    //--------------------------
    if (points.size(0) < 3){
        //--------------------------
        return true;
        //--------------------------
    }// end if (points.size(0) < 3)
    //--------------------------
    torch::Tensor xDiff = points[1][0] - points[0][0];
    torch::Tensor yDiff = points[1][1] - points[0][1];
    torch::Tensor ratio = xDiff / yDiff;
    //--------------------------
    bool consistent{true};
    //--------------------------
    at::parallel_for(2, points.size(0), 1, [&](int64_t start, int64_t end) {
        //--------------------------
        for (int64_t i = start; i < end; ++i) {
            //--------------------------
            torch::Tensor currentXDiff = points[i][0] - points[0][0];
            torch::Tensor currentYDiff = points[i][1] - points[0][1];
            torch::Tensor currentRatio = currentXDiff / currentYDiff;
            //--------------------------
            if (torch::abs(currentRatio - ratio).item<double>() > tolerance) {
                //--------------------------
                consistent = false;
                //--------------------------
                break;
                //--------------------------
            }// end if (torch::abs(currentRatio - ratio) > tolerance)
            //--------------------------
        }// end for (int64_t i = start; i < end; ++i)
        //--------------------------
    });
    //--------------------------
    return consistent;
    //--------------------------
}// end torch::Tensor Utils::CircleEquation::areAngleRatiosConsistent(const torch::Tensor& points, const double& tolerance)
//--------------------------------------------------------------
// void Utils::CircleEquation::areAngleRatiosConsistent(double& reward, const torch::Tensor& points, const double& tolerance){
//     //--------------------------
//     if (points.size(0) < 3){
//         //--------------------------
//         reward += 1.;
//         //--------------------------
//         return;
//         //--------------------------
//     }// end if (points.size(0) < 3)
//     //--------------------------
//     torch::Tensor xDiff = points[1][0] - points[0][0];
//     torch::Tensor yDiff = points[1][1] - points[0][1];
//     torch::Tensor ratio = xDiff / yDiff;
//     //--------------------------
//     double consistent{1.};
//     //--------------------------
//     at::parallel_for(2, points.size(0), 1, [&](int64_t start, int64_t end) {
//         //--------------------------
//         for (int64_t i = start; i < end; ++i) {
//             //--------------------------
//             torch::Tensor currentXDiff = points[i][0] - points[0][0];
//             torch::Tensor currentYDiff = points[i][1] - points[0][1];
//             torch::Tensor currentRatio = currentXDiff / currentYDiff;
//             //--------------------------
//             if (torch::abs(currentRatio - ratio).item<double>() > tolerance) {
//                 //--------------------------
//                 consistent = 0.;
//                 //--------------------------
//                 break;
//                 //--------------------------
//             }// end if (torch::abs(currentRatio - ratio) > tolerance)
//             //--------------------------
//         }// end for (int64_t i = start; i < end; ++i)
//         //--------------------------
//     });
//     //--------------------------
//     reward += consistent;
//     //--------------------------
// }// end void Utils::CircleEquation::areAngleRatiosConsistent(double& reward, const torch::Tensor& points, const double& tolerance)
//--------------------------------------------------------------
void Utils::CircleEquation::areAngleRatiosConsistent(double& reward, const torch::Tensor& points, const double& tolerance) {
    if (points.size(0) < 3) {
        reward += 1.;
        return;
    }

    torch::Tensor xDiff = points[1][0] - points[0][0];
    torch::Tensor yDiff = points[1][1] - points[0][1];
    // Ensure that we don't divide by zero
    if (yDiff.item<double>() == 0) {
        reward += 0.; // Set reward to 0 if the denominator is zero
        return;
    }
    torch::Tensor ratio = xDiff / yDiff;

    std::atomic<bool> consistent(true);

    at::parallel_for(2, points.size(0), 1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end && consistent.load(); ++i) {
            torch::Tensor currentXDiff = points[i][0] - points[0][0];
            torch::Tensor currentYDiff = points[i][1] - points[0][1];
            // Again, ensure we don't divide by zero
            if (currentYDiff.item<double>() == 0) {
                consistent.store(false);
                break;
            }
            torch::Tensor currentRatio = currentXDiff / currentYDiff;
            
            if (torch::abs(currentRatio - ratio).item<double>() > tolerance) {
                consistent.store(false);
                break;
            }
        }
    });

    reward += consistent.load() ? 1. : 0.;
}
//--------------------------------------------------------------
void Utils::CircleEquation::arePointsSymmetric(torch::Tensor& reward, const torch::Tensor& points){
    //--------------------------
    if (points.size(0) < 3){
        //--------------------------
        reward += 1.;
        //--------------------------
        return;
        //--------------------------
    }// end if (points.size(0) < 3)
    //--------------------------
    torch::Tensor xSum = torch::sum(points.select(1, 0));
    torch::Tensor ySum = torch::sum(points.select(1, 1));
    double numPoints = static_cast<double>(points.size(0));
    //--------------------------
    torch::Tensor centerX = xSum / numPoints;
    torch::Tensor centerY = ySum / numPoints;
    //--------------------------
    torch::Tensor reflectedX = 2.0 * centerX - points.select(1, 0);
    torch::Tensor reflectedY = 2.0 * centerY - points.select(1, 1);
    //--------------------------
    at::parallel_for(0, points.size(0), 1, [&](int64_t start, int64_t end) {
        //--------------------------
        for (int64_t i = start; i < end; ++i) {
            //--------------------------
            torch::Tensor diffX = torch::abs(reflectedX - points[i][0]);
            torch::Tensor diffY = torch::abs(reflectedY - points[i][1]);
            //--------------------------
            torch::Tensor foundReflection = torch::any(diffX < std::numeric_limits<double>::epsilon() &
                                                       diffY < std::numeric_limits<double>::epsilon());
            //--------------------------
            if (foundReflection.item<bool>()) {
                //--------------------------
                reward[i] += 1.;
                //--------------------------
            }// end if (!foundReflection.item<bool>())
            //--------------------------
        }// end for (int64_t i = start; i < end; ++i)
        //--------------------------
    });
    //--------------------------
}// end void Utils::CircleEquation::arePointsSymmetric(torch::Tensor& reward, const torch::Tensor& points)
//--------------------------------------------------------------
/*
torch::Tensor Utils::CircleEquation::arePointsSymmetric(const torch::Tensor& points){
    //--------------------------
    if (points.size(0) < 3){
        //--------------------------
        return torch::ones({points.size(0)}, torch::kBool);
        //--------------------------
    }// end if (points.size(0) < 3)
    //--------------------------
    torch::Tensor xSum = torch::sum(points.select(1, 0));
    torch::Tensor ySum = torch::sum(points.select(1, 1));
    double numPoints = static_cast<double>(points.size(0));
    //--------------------------
    torch::Tensor centerX = xSum / numPoints;
    torch::Tensor centerY = ySum / numPoints;
    //--------------------------
    torch::Tensor reflectedX = 2.0 * centerX - points.select(1, 0);
    torch::Tensor reflectedY = 2.0 * centerY - points.select(1, 1);
    //--------------------------
    auto symmetric = torch::ones({points.size(0)}, torch::kBool);
    //--------------------------
    at::parallel_for(0, points.size(0), 1, [&](int64_t start, int64_t end) {
        //--------------------------
        for (int64_t i = start; i < end; ++i) {
            //--------------------------
            torch::Tensor diffX = torch::abs(reflectedX - points[i][0]);
            torch::Tensor diffY = torch::abs(reflectedY - points[i][1]);
            //--------------------------
            torch::Tensor foundReflection = torch::any(diffX < std::numeric_limits<double>::epsilon() &
                                                       diffY < std::numeric_limits<double>::epsilon());
            //--------------------------
            if (!foundReflection.item<bool>()) {
                //--------------------------
                symmetric[i] = false;
                //--------------------------
            }// end if (!foundReflection.item<bool>())
            //--------------------------
        }// end for (int64_t i = start; i < end; ++i)
        //--------------------------
    });
    //--------------------------
    return symmetric;
    //--------------------------
}// end bool Utils::CircleEquation::arePointsSymmetric(const torch::Tensor& points)
*/
//--------------------------------------------------------------
bool Utils::CircleEquation::arePointsSymmetric(const torch::Tensor& points){
    //--------------------------
    if (points.size(0) < 3){
        //--------------------------
        return true;
        //--------------------------
    }// end if (points.size(0) < 3)
    //--------------------------
    torch::Tensor xSum = torch::sum(points.select(1, 0));
    torch::Tensor ySum = torch::sum(points.select(1, 1));
    double numPoints = static_cast<double>(points.size(0));
    //--------------------------
    torch::Tensor centerX = xSum / numPoints;
    torch::Tensor centerY = ySum / numPoints;
    //--------------------------
    torch::Tensor reflectedX = 2.0 * centerX - points.select(1, 0);
    torch::Tensor reflectedY = 2.0 * centerY - points.select(1, 1);
    //--------------------------
    bool symmetric{true};
    //--------------------------
    at::parallel_for(0, points.size(0), 1, [&](int64_t start, int64_t end) {
        //--------------------------
        for (int64_t i = start; i < end; ++i) {
            //--------------------------
            torch::Tensor diffX = torch::abs(reflectedX - points[i][0]);
            torch::Tensor diffY = torch::abs(reflectedY - points[i][1]);
            //--------------------------
            torch::Tensor foundReflection = torch::any(diffX < std::numeric_limits<double>::epsilon() &
                                                       diffY < std::numeric_limits<double>::epsilon());
            //--------------------------
            if (!foundReflection.item<bool>()) {
                //--------------------------
                symmetric = false;
                //--------------------------
            }// end if (!foundReflection.item<bool>())
            //--------------------------
        }// end for (int64_t i = start; i < end; ++i)
        //--------------------------
    });
    //--------------------------
    return symmetric;
    //--------------------------
}// end bool Utils::CircleEquation::arePointsSymmetric(const torch::Tensor& points)
//--------------------------------------------------------------
// void Utils::CircleEquation::arePointsSymmetric(double& reward, const torch::Tensor& points){
//     //--------------------------
//     if (points.size(0) < 3){
//         //--------------------------
//         reward += 1.;
//         //--------------------------
//         return;
//         //--------------------------
//     }// end if (points.size(0) < 3)
//     //--------------------------
//     torch::Tensor xSum = torch::sum(points.select(1, 0));
//     torch::Tensor ySum = torch::sum(points.select(1, 1));
//     double numPoints = static_cast<double>(points.size(0));
//     //--------------------------
//     torch::Tensor centerX = xSum / numPoints;
//     torch::Tensor centerY = ySum / numPoints;
//     //--------------------------
//     torch::Tensor reflectedX = 2.0 * centerX - points.select(1, 0);
//     torch::Tensor reflectedY = 2.0 * centerY - points.select(1, 1);
//     //--------------------------
//     double symmetric{1.};
//     //--------------------------
//     at::parallel_for(0, points.size(0), 1, [&](int64_t start, int64_t end) {
//         //--------------------------
//         for (int64_t i = start; i < end; ++i) {
//             //--------------------------
//             torch::Tensor diffX = torch::abs(reflectedX - points[i][0]);
//             torch::Tensor diffY = torch::abs(reflectedY - points[i][1]);
//             //--------------------------
//             torch::Tensor foundReflection = torch::any(diffX < std::numeric_limits<double>::epsilon() &
//                                                        diffY < std::numeric_limits<double>::epsilon());
//             //--------------------------
//             if (!foundReflection.item<bool>()) {
//                 //--------------------------
//                 symmetric = 0.;
//                 //--------------------------
//             }// end if (!foundReflection.item<bool>())
//             //--------------------------
//         }// end for (int64_t i = start; i < end; ++i)
//         //--------------------------
//     });
//     //--------------------------
//     reward += symmetric;
//     //--------------------------
// }// end void Utils::CircleEquation::arePointsSymmetric(double& reward, const torch::Tensor& points)
//--------------------------------------------------------------
// void Utils::CircleEquation::arePointsSymmetric(double& reward, const torch::Tensor& points) {
//     if (points.size(0) < 3) {
//         reward += 1.;
//         return;
//     }

//     torch::Tensor xSum = torch::sum(points.select(1, 0));
//     torch::Tensor ySum = torch::sum(points.select(1, 1));
//     double numPoints = static_cast<double>(points.size(0));

//     torch::Tensor centerX = xSum / numPoints;
//     torch::Tensor centerY = ySum / numPoints;

//     std::atomic<bool> symmetric(true);

//     at::parallel_for(0, points.size(0), 1, [&](int64_t start, int64_t end) {
//         for (int64_t i = start; i < end && symmetric.load(); ++i) {
//             torch::Tensor reflectedPointX = 2.0 * centerX - points[i][0];
//             torch::Tensor reflectedPointY = 2.0 * centerY - points[i][1];

//             bool foundReflection = false;
//             for (int64_t j = 0; j < points.size(0) and !foundReflection; ++j) {
//                 torch::Tensor diffX = torch::abs(reflectedPointX - points[j][0]);
//                 torch::Tensor diffY = torch::abs(reflectedPointY - points[j][1]);

//                 if (torch::all(diffX < std::numeric_limits<double>::epsilon() & 
//                                diffY < std::numeric_limits<double>::epsilon()).item<bool>()) {
//                     foundReflection = true;
//                 }
//             }
//             if (!foundReflection) {
//                 symmetric.store(false);
//             }
//         }
//     });

//     reward += symmetric.load() ? 1.0 : 0.0;
// }
//--------------------------------------------------------------
bool Utils::CircleEquation::isSymmetricCounterpartInSet(const m_point& reflected, 
                                                        const boost::geometry::index::rtree<m_point, boost::geometry::index::quadratic<16>>& rtree) {
    // Using a very small box around the reflected point for querying
    m_point min_corner(reflected.get<0>() - std::numeric_limits<double>::epsilon(),
                       reflected.get<1>() - std::numeric_limits<double>::epsilon());
    m_point max_corner(reflected.get<0>() + std::numeric_limits<double>::epsilon(),
                       reflected.get<1>() + std::numeric_limits<double>::epsilon());
    m_box query_box(min_corner, max_corner);

    std::vector<m_point> result;
    result.reserve(10);
    rtree.query(boost::geometry::index::intersects(query_box), std::back_inserter(result));
    return !result.empty();
}
//--------------------------------------------------------------
void Utils::CircleEquation::arePointsSymmetric(double& reward, const torch::Tensor& points) {
    if (points.size(0) < 3) {
        reward += 1.;
        return;
    }

    torch::Tensor xSum = torch::sum(points.select(1, 0));
    torch::Tensor ySum = torch::sum(points.select(1, 1));
    double numPoints = static_cast<double>(points.size(0));

    torch::Tensor centerX = xSum / numPoints;
    torch::Tensor centerY = ySum / numPoints;

    torch::Tensor reflectedX = 2.0 * centerX - points.select(1, 0);
    torch::Tensor reflectedY = 2.0 * centerY - points.select(1, 1);

    // Prepare the R-tree
    std::vector<m_point> pointVector(points.size(0));
    std::atomic<int64_t> atomicIndex(0);

    std::generate_n(std::execution::par, pointVector.begin(), points.size(0), [&]() {
        int64_t i = atomicIndex.fetch_add(1);
        return m_point(points[i][0].item<double>(), points[i][1].item<double>());
    });

    boost::geometry::index::rtree<m_point, boost::geometry::index::quadratic<16>> rtree(pointVector.begin(), pointVector.end());

    std::atomic<bool> symmetric(true);

    at::parallel_for(0, points.size(0), 1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end and symmetric.load(); ++i) {
            m_point reflectedPoint(reflectedX[i].item<double>(), reflectedY[i].item<double>());
            auto nearest_iter = rtree.qbegin(boost::geometry::index::nearest(reflectedPoint, 1));
            if (nearest_iter == rtree.qend() || 
                boost::geometry::distance(reflectedPoint, *nearest_iter) > std::numeric_limits<double>::epsilon()) {
                symmetric.store(false);
            }
        }
    });

    reward += symmetric.load() ? 1. : 0.;
}
//--------------------------------------------------------------
// torch::Tensor Utils::CircleEquation::calculateTriangleArea(const torch::Tensor& points) {
//     if (points.size(0) < 3){
//         //--------------------------
//         return torch::zeros({}, torch::kFloat64);
//         //--------------------------
//     }// end if (points.size(0) < 3)
//     //--------------------------
//     torch::Tensor area = torch::zeros({}, torch::kFloat64);
//     //--------------------------
//     at::parallel_for(0, points.size(0) - 2, 1, [&](int64_t start, int64_t end) {
//         //--------------------------
//         for (int64_t i = start; i < end; ++i) {
//             //--------------------------
//             auto x1 = points[i][0];
//             auto y1 = points[i][1];
//             auto x2 = points[i + 1][0];
//             auto y2 = points[i + 1][1];
//             auto x3 = points[i + 2][0];
//             auto y3 = points[i + 2][1];
//             //--------------------------
//             area += 0.5 * torch::abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)));
//             //--------------------------
//         }// end for (int64_t i = start; i < end; ++i)
//         //--------------------------
//     });
//     //--------------------------
//     return area;
//     //--------------------------
// }// end torch::Tensor Utils::CircleEquation::calculateTriangleArea(const torch::Tensor& points)
//--------------------------------------------------------------
torch::Tensor Utils::CircleEquation::calculateTriangleArea(const torch::Tensor& points, const long double& tolerance) {
    if (points.size(0) < 3){
        return torch::zeros({}, torch::kFloat64);
    }

    // Calculate side lengths
    boost::multiprecision::cpp_dec_float_50 a = torch::norm(points[1] - points[0]).item<double>();
    boost::multiprecision::cpp_dec_float_50 b = torch::norm(points[2] - points[1]).item<double>();
    boost::multiprecision::cpp_dec_float_50 c = torch::norm(points[0] - points[2]).item<double>();

    // Normalize by dividing each side by the longest side
    boost::multiprecision::cpp_dec_float_50 max_side = std::max({a, b, c});

    if (max_side <= std::numeric_limits<double>::epsilon()) {
        return torch::zeros({}, torch::kFloat64);
    }

    a /= max_side;
    b /= max_side;
    c /= max_side;

    boost::multiprecision::cpp_dec_float_50 s = (a + b + c) / 2.0;
    // boost::multiprecision::cpp_dec_float_50 area = boost::multiprecision::sqrt(s * (s - a) * (s - b) * (s - c));

    boost::multiprecision::cpp_dec_float_50 area_term = s * (s - a) * (s - b) * (s - c);
    // Check for negative value under the square root
    if (area_term < 0) {
        // This should ideally never happen after our triangle validation, but better safe than sorry
        return torch::tensor(0.0, torch::kFloat64);
    }

    boost::multiprecision::cpp_dec_float_50 area = sqrt(area_term);
    // Check if triangle is near-degenerate
    if (area < tolerance) {
        // Handle special case, e.g., return a small default value
        area = tolerance;
    }
  
    // Normalize the area to [0,1] range
    // We use 0.5 as the maximum possible area for an equilateral triangle with side length of 1
    area = area / 0.5;

    return torch::tensor(static_cast<double>(area), torch::kFloat64);
}
//--------------------------------------------------------------
void Utils::CircleEquation::calculateTriangleArea(double& reward, const torch::Tensor& points, const long double& tolerance){
    //-------------------------
    reward += (reward > 1E-6) ? calculateTriangleArea(points, tolerance).item<double>() : 0.;
    //--------------------------
}// end void Utils::CircleEquation::calculateTriangleArea(double& reward, const torch::Tensor& points)
//--------------------------------------------------------------
// torch::Tensor Utils::CircleEquation::calculateCircleSmoothness(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius) {
//     //--------------------------
//     torch::Tensor smoothness = torch::zeros({}, torch::kFloat64);
//     //--------------------------
//     at::parallel_for(0, points.size(0), 1, [&](int64_t start, int64_t end) {
//         //--------------------------
//         for (int64_t i = start; i < end; ++i) {
//             //--------------------------
//             auto distance = torch::hypot((points[i][0] - center.select(1, 0)), (points[i][1] - center.select(1, 1)));
//             auto deviation = torch::abs(distance - radius);
//             //--------------------------
//             smoothness += deviation.item<double>();
//             //--------------------------
//         }// end for (int64_t i = start; i < end; ++i)
//         //--------------------------
//     });
//     //--------------------------
//     return smoothness;
//     //--------------------------
// }// end torch::Tensor Utils::CircleEquation::calculateCircleSmoothness(const torch::Tensor& points, const torch::Tensor& center, torch::Tensor radius) 
//--------------------------------------------------------------
torch::Tensor Utils::CircleEquation::calculateCircleSmoothness(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius) {
    //--------------------------
    std::atomic<double> atomic_smoothness(0.0);
    //--------------------------
    at::parallel_for(0, points.size(0), 1, [&](int64_t start, int64_t end) {
        //--------------------------
        double local_smoothness = 0.0; // thread-local variable for accumulating results
        //--------------------------
        for (int64_t i = start; i < end; ++i) {
            //--------------------------
            auto distance = torch::hypot((points[i][0] - center.select(1, 0)), (points[i][1] - center.select(1, 1)));
            auto deviation = torch::abs(distance - radius);
            //--------------------------
            local_smoothness += deviation.item<double>();
            //--------------------------
        }// end for (int64_t i = start; i < end; ++i)
        //--------------------------
        atomic_smoothness.fetch_add(local_smoothness, std::memory_order_relaxed); // atomic addition after local accumulation
        //--------------------------
    });
    //--------------------------
    return torch::tensor(atomic_smoothness.load(std::memory_order_relaxed), torch::kFloat64); // converting atomic variable to tensor
    //--------------------------
}// end torch::Tensor Utils::CircleEquation::calculateCircleSmoothness(const torch::Tensor& points, const torch::Tensor& center, torch::Tensor radius) 
//--------------------------------------------------------------
void Utils::CircleEquation::calculateCircleSmoothness(double& reward, const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius){
    //--------------------------
    auto smoothness = calculateCircleSmoothness(points, center, radius).item<double>();
    //--------------------------
    (smoothness <= 1E-9) ? reward /= 1. : reward /= smoothness;
    //--------------------------
}// end void Utils::CircleEquation::calculateCircleSmoothness(double& reward, const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius)
//--------------------------------------------------------------
torch::Tensor Utils::CircleEquation::getMaxPointLimiter(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius) {
    //--------------------------
    // Compute distance_squared for each point in the batch
    torch::Tensor distance_squared = torch::sum((points - center).pow(2), /*dim=*/1);
    //--------------------------
    // Compare distance_squared with circle_radius squared and create a tensor of bools for points inside or on the circle
    torch::Tensor is_inside_circle = torch::le(distance_squared, radius.pow(2));
    //--------------------------
    // Return the mean of the reward tensor as a scalar double value
    return is_inside_circle.to(torch::kDouble);
    //--------------------------
}// end torch::Tensor Utils::CircleEquation::getMaxPointLimiter(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius)
//--------------------------------------------------------------
// void Utils::CircleEquation::getMaxPointLimiter(double& reward, const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius) {
//     //--------------------------
//     // Compute distance_squared for each point in the batch
//     torch::Tensor distance_squared = torch::sum((points - center).pow(2), /*dim=*/1);
//     //--------------------------
//     // Compare distance_squared with circle_radius squared and create a tensor of bools for points inside or on the circle
//     torch::Tensor is_inside_circle = torch::le(distance_squared, radius.pow(2));
//     //--------------------------
//     // Return the mean of the reward tensor as a scalar double value
//     reward += is_inside_circle.to(torch::kDouble).mean().item<double>();
//     //--------------------------
// }//end bool Utils::CircleEquation::getMaxPointLimiter(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius)
//--------------------------------------------------------------
// void Utils::CircleEquation::getMaxPointLimiter(double& reward, const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius) {
//     // Compute squared distance for each point in the batch from the center
//     torch::Tensor distance_squared = torch::sum((points - center).pow(2), /*dim=*/1);

//     // Compare distance_squared with circle's squared radius
//     torch::Tensor is_inside_circle = torch::le(distance_squared, radius.pow(2));

//     // Calculate the fraction of points inside or on the circle
//     double inside_fraction = is_inside_circle.to(torch::kDouble).mean().item<double>();

//     // Strategy 1: Exponential scaling
//     reward += std::pow(inside_fraction, 2); // or some other exponent based on how much emphasis you want
    
//     // Strategy 2: Threshold-Based Penalty
//     const double threshold = 0.9; // for example, at least 90% points should be inside
//     if(inside_fraction < threshold) {
//         reward -= 0.5; // or some other penalty value
//     }
// }
//--------------------------------------------------------------
// void Utils::CircleEquation::getMaxPointLimiter(double& reward, const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius) {
//     // Compute squared distance for each point in the batch from the center
//     torch::Tensor distance_squared = torch::sum((points - center).pow(2), /*dim=*/1);

//     // Compute squared radius once
//     torch::Tensor radius_squared = radius.pow(2);

//     // Compare distance_squared with circle's squared radius
//     torch::Tensor is_inside_circle = torch::le(distance_squared, radius_squared);

//     // Calculate the fraction of points inside or on the circle
//     double inside_fraction = is_inside_circle.to(torch::kDouble).mean().item<double>();

//     // Exponential scaling
//     double inside_reward = std::pow(inside_fraction, 3);  // cubing to penalize more for deviations
//     double outside_reward = 1.0 - inside_reward;

//     // Now apply a weight or scaling factor
//     double weight_inside = 2.0;  // giving more weight to points inside the circle
//     double weight_outside = -2.0;  // penalty for points outside the circle

//     reward += weight_inside * inside_reward + weight_outside * outside_reward;
// }
//--------------------------------------------------------------
// void Utils::CircleEquation::getMaxPointLimiter(double& reward, const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius) {
//     // Compute squared distance for each point in the batch from the center
//     torch::Tensor distance_squared = torch::sum((points - center).pow(2), /*dim=*/1);

//     // Compute squared radius once
//     torch::Tensor radius_squared = radius.pow(2);

//     // Compare distance_squared with circle's squared radius
//     torch::Tensor is_inside_circle = torch::le(distance_squared, radius_squared);

//     // Calculate the fraction of points inside or on the circle
//     double inside_fraction = is_inside_circle.to(torch::kDouble).mean().item<double>();

//     // Calculate reward
//     // If all points are inside the circle, maximum reward of 1 is given.
//     // If any point is outside, we subtract a penalty from the reward.
//     double penalty_factor = 2.0;  // adjust based on the severity of the punishment you want
//     reward = inside_fraction - penalty_factor * (1.0 - inside_fraction);
// }
//--------------------------------------------------------------
void Utils::CircleEquation::getMaxPointLimiter(double& reward, const torch::Tensor& point, const torch::Tensor& center, const torch::Tensor& radius) {
    
    // Constants for weighting and punishment
    constexpr double weight_inside = 1.0;  // Example value
    constexpr double punishment_outside = -0.5;  // Example value

    // Compute distance squared from the point to the circle's center
    torch::Tensor dist_squared = torch::sum((point - center).pow(2));
    
    // Check if the point lies inside or on the circle
    bool is_inside = dist_squared.item<double>() <= radius.item<double>();
    
    // Update the reward based on the point's position
    reward += (reward>0) ? is_inside ? weight_inside : -reward : 0; // or another appropriate value for outside points
}
//--------------------------------------------------------------
bool Utils::CircleEquation::PointsDistinct(const torch::Tensor& point1, const torch::Tensor& point2) {
    //--------------------------
    // Check if the coordinates of the two points are different
    return torch::all(torch::ne(point1, point2)).item<bool>();
    //--------------------------
}// end bool Utils::CircleEquation::PointsDistinct(const torch::Tensor& point1, const torch::Tensor& point2)
//--------------------------------------------------------------
void Utils::CircleEquation::PointsDistinct(double& reward, const torch::Tensor& point1, const torch::Tensor& point2){
    //--------------------------
    // Check if the coordinates of the two points are different
    if(torch::all(torch::ne(point1, point2)).item<bool>()){
        //--------------------------
        reward += 1.;
        //--------------------------
    }// end if(torch::all(torch::ne(point1, point2)).item<bool>())
    else{
        //--------------------------
        reward += 0.;
        //--------------------------
    }// end else 
    //--------------------------
}// end void Utils::CircleEquation::PointsDistinct(double& reward, const torch::Tensor& point1, const torch::Tensor& point2)
//--------------------------------------------------------------