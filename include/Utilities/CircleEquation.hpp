#pragma once

//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------
namespace Utils{
    //--------------------------------------------------------------
    class CircleEquation{
        //--------------------------------------------------------------
        public:
            //--------------------------
            CircleEquation(void) = delete;
            //--------------------------
            static void Aligned(torch::Tensor& reward, const torch::Tensor& points, const double& tolerance);
            //--------------------------
            static bool Aligned(const torch::Tensor& points, const double& tolerance);
            //--------------------------
            static void Aligned(double& reward, const torch::Tensor& points, const double& tolerance);
            //--------------------------------------------------------------
            static void CloseToCircumference(torch::Tensor& reward, const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius, const double& tolerance);
            //--------------------------
            static bool CloseToCircumference(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius, const double& tolerance);
            //--------------------------
            static void CloseToCircumference(double& reward, const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius, const double& tolerance);
            //--------------------------------------------------------------
            static void Equidistant(torch::Tensor& reward, const torch::Tensor& points, const double& tolerance);
            //--------------------------
            static bool Equidistant(const torch::Tensor& points, const double& tolerance);
            //--------------------------
            static void Equidistant(double& reward, const torch::Tensor& points, const double& tolerance);
            //--------------------------------------------------------------
            static void AngleRatiosConsistent(torch::Tensor& reward, const torch::Tensor& points, const double& tolerance);
            //--------------------------
            static bool AngleRatiosConsistent(const torch::Tensor& points, const double& tolerance);
            //--------------------------
            static void AngleRatiosConsistent(double& reward, const torch::Tensor& points, const double& tolerance);
            //--------------------------------------------------------------
            static void Symmetric(torch::Tensor& reward, const torch::Tensor& points);
            //--------------------------
            static bool Symmetric(const torch::Tensor& points);
            //--------------------------
            static void Symmetric(double& reward, const torch::Tensor& points);
            //--------------------------------------------------------------
            static torch::Tensor TriangleArea(const torch::Tensor& points);
            //--------------------------
            static void TriangleArea(double& reward, const torch::Tensor& points);
            //--------------------------------------------------------------
            static torch::Tensor CircleSmoothness(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius);
            //--------------------------
            static void CircleSmoothness(double& reward, const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius);
            //--------------------------------------------------------------
        protected:
            //--------------------------------------------------------------
            static void arePointsAligned(torch::Tensor& reward, const torch::Tensor& points, const double& tolerance); 
            //--------------------------
            /*static torch::Tensor arePointsAligned(const torch::Tensor& points, const double& tolerance);*/
            //--------------------------
            static bool arePointsAligned(const torch::Tensor& points, const double& tolerance);
            //--------------------------
            static void arePointsAligned(double& reward, const torch::Tensor& points, const double& tolerance);
            //--------------------------------------------------------------
            static void arePointsCloseToCircumference(  torch::Tensor& reward,
                                                        const torch::Tensor& points,
                                                        const torch::Tensor& center,
                                                        const torch::Tensor& radius,
                                                        const double& tolerance);
            //--------------------------
            /* static torch::Tensor arePointsCloseToCircumference(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius, const double& tolerance); */
            //--------------------------
            static bool arePointsCloseToCircumference(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius, const double& tolerance);
            //--------------------------
            static void arePointsCloseToCircumference(  double& reward,
                                                        const torch::Tensor& points,
                                                        const torch::Tensor& center,
                                                        const torch::Tensor& radius,
                                                        const double& tolerance);
            //--------------------------------------------------------------
            static void arePointsEquidistant(torch::Tensor& reward, const torch::Tensor& points, const double& tolerance);
            //--------------------------
            /* static torch::Tensor arePointsEquidistant(const torch::Tensor& points, const double& tolerance); */
            //--------------------------
            static bool arePointsEquidistant(const torch::Tensor& points, const double& tolerance);
            //--------------------------
            static void arePointsEquidistant(double& reward, const torch::Tensor& points, const double& tolerance);
            //--------------------------------------------------------------
            static void areAngleRatiosConsistent(torch::Tensor& reward, const torch::Tensor& points, const double& tolerance);
            //--------------------------
            /* static torch::Tensor areAngleRatiosConsistent(const torch::Tensor& points, const double& tolerance); */
            //--------------------------
            static bool areAngleRatiosConsistent(const torch::Tensor& points, const double& tolerance);
            //--------------------------
            static void areAngleRatiosConsistent(double& reward, const torch::Tensor& points, const double& tolerance);
            //--------------------------------------------------------------
            static void arePointsSymmetric(torch::Tensor& reward, const torch::Tensor& points);
            //--------------------------
            /* static torch::Tensor arePointsSymmetric(const torch::Tensor& points); */
            //--------------------------
            static bool arePointsSymmetric(const torch::Tensor& points);
            //--------------------------
            static void arePointsSymmetric(double& reward, const torch::Tensor& points);
            //--------------------------------------------------------------
            static torch::Tensor calculateTriangleArea(const torch::Tensor& points);
            //--------------------------
            static void calculateTriangleArea(double& reward, const torch::Tensor& points);
            //--------------------------------------------------------------
            static torch::Tensor calculateCircleSmoothness(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius);
            //--------------------------
            static void calculateCircleSmoothness(double& reward, const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius);
        //--------------------------------------------------------------
    };// end class CircleEquation
    //--------------------------------------------------------------
}//end namespace CircleEquation
//--------------------------------------------------------------