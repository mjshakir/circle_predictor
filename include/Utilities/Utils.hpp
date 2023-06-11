#pragma once

//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------
class Utils{
    //--------------------------------------------------------------
    public:
        //--------------------------
        Utils(void) = delete;
        //--------------------------
        static void Aligned(torch::Tensor& reward, const torch::Tensor& points, const double& tolerance);
        //--------------------------
        static torch::Tensor Aligned(const torch::Tensor& points, const double& tolerance);
        //--------------------------
        static void CloseToCircumference(torch::Tensor& reward, const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius, const double& tolerance);
        //--------------------------
        static torch::Tensor CloseToCircumference(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius, const double& tolerance);
        //--------------------------
        static void Equidistant(torch::Tensor& reward, const torch::Tensor& points, const double& tolerance);
        //--------------------------
        static torch::Tensor Equidistant(const torch::Tensor& points, const double& tolerance);
        //--------------------------
        static void AngleRatiosConsistent(torch::Tensor& reward, const torch::Tensor& points, const double& tolerance);
        //--------------------------
        static torch::Tensor AngleRatiosConsistent(const torch::Tensor& points, const double& tolerance);
        //--------------------------
        static void Symmetric(torch::Tensor& reward, const torch::Tensor& points);
        //--------------------------
        static torch::Tensor Symmetric(const torch::Tensor& points);
        //--------------------------
        static torch::Tensor TriangleArea(const torch::Tensor& points);
        //--------------------------
        static torch::Tensor CircleSmoothness(const torch::Tensor& points, const torch::Tensor& center, torch::Tensor radius);
    //--------------------------------------------------------------
    protected:
    //--------------------------------------------------------------
        static void arePointsAligned(torch::Tensor& reward, const torch::Tensor& points, const double& tolerance); 
        //--------------------------
        static torch::Tensor arePointsAligned(const torch::Tensor& points, const double& tolerance);
        //--------------------------
        static void arePointsCloseToCircumference(torch::Tensor& reward, const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius, const double& tolerance);
        //--------------------------
        static torch::Tensor arePointsCloseToCircumference(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius, const double& tolerance);
        //--------------------------
        static void arePointsEquidistant(torch::Tensor& reward, const torch::Tensor& points, const double& tolerance);
        //--------------------------
        static torch::Tensor arePointsEquidistant(const torch::Tensor& points, const double& tolerance);
        //--------------------------
        static void areAngleRatiosConsistent(torch::Tensor& reward, const torch::Tensor& points, const double& tolerance);
        //--------------------------
        static torch::Tensor areAngleRatiosConsistent(const torch::Tensor& points, const double& tolerance);
        //--------------------------
        static void arePointsSymmetric(torch::Tensor& reward, const torch::Tensor& points);
        //--------------------------
        static torch::Tensor arePointsSymmetric(const torch::Tensor& points);
        //--------------------------
        static torch::Tensor calculateTriangleArea(const torch::Tensor& points);
        //--------------------------
        static torch::Tensor calculateCircleSmoothness(const torch::Tensor& points, const torch::Tensor& center, torch::Tensor radius);
    //--------------------------------------------------------------
};// end class Utils
//--------------------------------------------------------------