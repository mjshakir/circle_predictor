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
// public 
//--------------------------
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
torch::Tensor Utils::CircleEquation::TriangleArea(const torch::Tensor& points){
    //--------------------------
    return calculateTriangleArea(points);
    //--------------------------
}// end torch::Tensor Utils::CircleEquation::TriangleArea(const torch::Tensor& points)
//--------------------------------------------------------------
void Utils::CircleEquation::TriangleArea(double& reward, const torch::Tensor& points){
    //--------------------------
    calculateTriangleArea(reward, points);
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
// protected
//--------------------------
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
void Utils::CircleEquation::arePointsAligned(double& reward, const torch::Tensor& points, const double& tolerance){
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
    double aligned{1.};
    //--------------------------
    at::parallel_for(1, points.size(0), 1, [&](int64_t start, int64_t end) {
        //--------------------------
        for (int64_t i = start; i < end; ++i) {
            //--------------------------
            if (torch::abs(((xDiff * (x[i] - x[0])) - (yDiff * (y[i] - y[0])))).item<double>() > tolerance) {
                //--------------------------
                aligned = 0.;
                //--------------------------
                break;
                //--------------------------
            }// end if (torch::abs(diff) > tolerance)
            //--------------------------
        }// end or (int64_t i = start; i < end; ++i)
        //--------------------------
    });
    //--------------------------
    reward += aligned;
    //--------------------------
}// end void Utils::CircleEquation::arePointsAligned(double& reward, const torch::Tensor& points, const double& tolerance)
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
void Utils::CircleEquation::arePointsCloseToCircumference(  double& reward,
                                            const torch::Tensor& points,
                                            const torch::Tensor& center,
                                            const torch::Tensor& radius,
                                            const double& tolerance){
    //--------------------------
    if (points.size(0) < 3) {
        //--------------------------
        reward += 1.;
        //--------------------------
        return;
        //--------------------------
    }// end if (points.size(0) < 3)
    //--------------------------
    torch::Tensor distances = torch::hypot(points.select(1, 0) - center.select(1, 0), points.select(1, 1) - center.select(1, 1));
    torch::Tensor diff = torch::abs(distances - radius);
    //--------------------------
    double close{1.};
    //--------------------------
    at::parallel_for(1, points.size(0), 1, [&](int64_t start, int64_t end) {
        //--------------------------
        for (int64_t i = start; i < end; ++i) {
            //--------------------------
            if (diff[i].item<double>() > tolerance) {
                //--------------------------
                close = 0.;
                //--------------------------
                break;
                //--------------------------
            }// end if (torch::abs(diff) > tolerance)
            //--------------------------
        }// end or (int64_t i = start; i < end; ++i)
        //--------------------------
    });
    //--------------------------
    reward += close;
    //--------------------------
}// end void Utils::CircleEquation::arePointsCloseToCircumference
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
void Utils::CircleEquation::arePointsEquidistant(double& reward, const torch::Tensor& points, const double& tolerance){
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
    double equidistant{1.};
    //--------------------------
    at::parallel_for(2, points.size(0), 1, [&](int64_t start, int64_t end) {
        //--------------------------
        for (int64_t i = start; i < end; ++i) {
            //--------------------------
            torch::Tensor currentDistance = torch::hypot(points[i][0] - referencePoint[0], points[i][1] - referencePoint[1]);
            //--------------------------
            if (torch::abs(currentDistance - distance[i]).item<double>() > tolerance) {
                //--------------------------
                equidistant = 0.;
                //--------------------------
                break;
                //--------------------------
            }// end if (torch::abs(currentDistance - distance).item<double>() > tolerance)
            //--------------------------
        }// end for (int64_t i = start; i < end; ++i)
        //--------------------------
    });
    //--------------------------
    reward += equidistant;
    //--------------------------
}// end void Utils::CircleEquation::arePointsEquidistant(double& reward, const torch::Tensor& points, const double& tolerance)
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
void Utils::CircleEquation::areAngleRatiosConsistent(double& reward, const torch::Tensor& points, const double& tolerance){
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
    double consistent{1.};
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
                consistent = 0.;
                //--------------------------
                break;
                //--------------------------
            }// end if (torch::abs(currentRatio - ratio) > tolerance)
            //--------------------------
        }// end for (int64_t i = start; i < end; ++i)
        //--------------------------
    });
    //--------------------------
    reward += consistent;
    //--------------------------
}// end void Utils::CircleEquation::areAngleRatiosConsistent(double& reward, const torch::Tensor& points, const double& tolerance)
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
void Utils::CircleEquation::arePointsSymmetric(double& reward, const torch::Tensor& points){
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
    double symmetric{1.};
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
                symmetric = 0.;
                //--------------------------
            }// end if (!foundReflection.item<bool>())
            //--------------------------
        }// end for (int64_t i = start; i < end; ++i)
        //--------------------------
    });
    //--------------------------
    reward += symmetric;
    //--------------------------
}// end void Utils::CircleEquation::arePointsSymmetric(double& reward, const torch::Tensor& points)
//--------------------------------------------------------------
torch::Tensor Utils::CircleEquation::calculateTriangleArea(const torch::Tensor& points) {
    if (points.size(0) < 3){
        //--------------------------
        return torch::zeros({}, torch::kFloat64);
        //--------------------------
    }// end if (points.size(0) < 3)
    //--------------------------
    torch::Tensor area = torch::zeros({}, torch::kFloat64);
    //--------------------------
    at::parallel_for(0, points.size(0) - 2, 1, [&](int64_t start, int64_t end) {
        //--------------------------
        for (int64_t i = start; i < end; ++i) {
            //--------------------------
            auto x1 = points[i][0];
            auto y1 = points[i][1];
            auto x2 = points[i + 1][0];
            auto y2 = points[i + 1][1];
            auto x3 = points[i + 2][0];
            auto y3 = points[i + 2][1];
            //--------------------------
            area += 0.5 * torch::abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)));
            //--------------------------
        }// end for (int64_t i = start; i < end; ++i)
        //--------------------------
    });
    //--------------------------
    return area;
    //--------------------------
}// end torch::Tensor Utils::CircleEquation::calculateTriangleArea(const torch::Tensor& points)
//--------------------------------------------------------------
void Utils::CircleEquation::calculateTriangleArea(double& reward, const torch::Tensor& points){
    //-------------------------
    reward += calculateTriangleArea(points).item<double>();
    //--------------------------
}// end void Utils::CircleEquation::calculateTriangleArea(double& reward, const torch::Tensor& points)
//--------------------------------------------------------------
torch::Tensor Utils::CircleEquation::calculateCircleSmoothness(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius) {
    //--------------------------
    torch::Tensor smoothness = torch::zeros({}, torch::kFloat64);
    //--------------------------
    at::parallel_for(0, points.size(0), 1, [&](int64_t start, int64_t end) {
        //--------------------------
        for (int64_t i = start; i < end; ++i) {
            //--------------------------
            auto distance = torch::hypot((points[i][0] - center.select(1, 0)), (points[i][1] - center.select(1, 1)));
            auto deviation = torch::abs(distance - radius);
            //--------------------------
            smoothness += deviation.item<double>();
            //--------------------------
        }// end for (int64_t i = start; i < end; ++i)
        //--------------------------
    });
    //--------------------------
    return smoothness;
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