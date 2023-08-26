#pragma once

//--------------------------------------------------------------
// LibTorch library
//--------------------------------------------------------------
#include <torch/torch.h>
//--------------------------------------------------------------
// Boost library
//--------------------------------------------------------------
#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
//--------------------------------------------------------------
namespace Utils{
    //--------------------------------------------------------------
    class CircleEquation{
        //--------------------------------------------------------------
        public:
            //--------------------------
            CircleEquation(void) = delete;
            //--------------------------
            ~CircleEquation(void) = delete;
            //--------------------------------------------------------------
            static torch::Tensor distance_reward(const torch::Tensor& input, const torch::Tensor& output);
            //--------------------------
            static torch::Tensor diversity_reward(const torch::Tensor& output, const torch::Tensor& input);
            //--------------------------
            static torch::Tensor consistency_reward(const torch::Tensor& output);
            //--------------------------------------------------------------
            static torch::Tensor distance_penalty(const torch::Tensor& input, const torch::Tensor& output);
            //--------------------------
            static torch::Tensor separation_penalty(const torch::Tensor& output, const double& min_distance = 1E-1);
            //--------------------------------------------------------------
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
            static torch::Tensor TriangleArea(const torch::Tensor& points, const long double& tolerance);
            //--------------------------
            static void TriangleArea(double& reward, const torch::Tensor& points, const long double& tolerance);
            //--------------------------------------------------------------
            static torch::Tensor CircleSmoothness(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius);
            //--------------------------
            static void CircleSmoothness(double& reward, const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius);            
            //--------------------------------------------------------------
            static torch::Tensor PointLimiter(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius);
            //--------------------------
            static void PointLimiter(double& reward, const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius);
            //--------------------------
            static torch::Tensor PointLimiter(const torch::Tensor& input, const torch::Tensor& output);
            //--------------------------------------------------------------
            static bool Distinct(const torch::Tensor& point1, const torch::Tensor& point2);
            //--------------------------
            static void Distinct(double& reward, const torch::Tensor& point1, const torch::Tensor& point2);
            //--------------------------------------------------------------
        protected:
            //--------------------------------------------------------------
            static torch::Tensor compute_distance_reward(const torch::Tensor& input, const torch::Tensor& output);
            //--------------------------
            static torch::Tensor compute_diversity_reward(const torch::Tensor& output, const torch::Tensor& input);
            //--------------------------
            static torch::Tensor compute_consistency_reward(const torch::Tensor& output);
            //--------------------------------------------------------------
            static torch::Tensor compute_distance_penalty(const torch::Tensor& input, const torch::Tensor& output);
            //--------------------------
            static torch::Tensor compute_point_separation_penalty(const torch::Tensor& output, const double& min_distance);
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
            static torch::Tensor calculateTriangleArea(const torch::Tensor& points, const long double& tolerance);
            //--------------------------
            static void calculateTriangleArea(double& reward, const torch::Tensor& points, const long double& tolerance);
            //--------------------------------------------------------------
            static torch::Tensor calculateCircleSmoothness(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius);
            //--------------------------
            static void calculateCircleSmoothness(double& reward, const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius);
            //--------------------------
            static torch::Tensor getMaxPointLimiter(const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius);
            //--------------------------
            static void getMaxPointLimiter(double& reward, const torch::Tensor& points, const torch::Tensor& center, const torch::Tensor& radius);
            //--------------------------
            static torch::Tensor max_point_limiter(const torch::Tensor& input, const torch::Tensor& output);
            //--------------------------
            static bool PointsDistinct(const torch::Tensor& point1, const torch::Tensor& point2);
            //--------------------------
            static void PointsDistinct(double& reward, const torch::Tensor& point1, const torch::Tensor& point2);
            //--------------------------------------------------------------
        private:
            //--------------------------------------------------------------
            typedef boost::geometry::model::point<double, 2, boost::geometry::cs::cartesian> m_point;
            typedef boost::geometry::model::box<m_point> m_box;
            //--------------------------
            static bool isSymmetricCounterpartInSet(const m_point& reflected, 
                                                    const boost::geometry::index::rtree<m_point, boost::geometry::index::quadratic<16>>& rtree);
        //--------------------------------------------------------------
    };// end class CircleEquation
    //--------------------------------------------------------------
}//end namespace CircleEquation
//--------------------------------------------------------------