//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Generate/RL/Generate.hpp"
//--------------------------------------------------------------
//short form to get the previous even: x &= ~1;
RL::Generate::Generate( const size_t& generated_points) :   m_generated_points((generated_points%2 == 0) ? generated_points : generated_points - (generated_points & 1)), 
                                                            m_generated_points_test(m_generated_points*0.2){
    //--------------------------
}// end RL::Generate::Generate(const size_t& generated_points, const size_t& column, const double& limiter)
//--------------------------------------------------------------
RL::Generate::Generate( const size_t& generated_points, 
                    const size_t& generated_points_test) : m_generated_points((generated_points%2 == 0) ? generated_points : generated_points - (generated_points & 1)),
                                                           m_generated_points_test((generated_points_test%2 == 0 and generated_points_test==0) ? 
                                                                generated_points_test : generated_points_test - (generated_points_test & 1)){
    //--------------------------
}// end RL::Generate::Generate( const size_t& generated_points, const size_t& generated_points_test, const size_t& column, const double& limiter)
//--------------------------------------------------------------
const size_t& RL::Generate::get_generated_points(void) const{
    //--------------------------
    return m_generated_points;
    //--------------------------
}// end const size_t& RL::Generate::get_generated_points(void)
//--------------------------------------------------------------
const size_t& RL::Generate::get_generated_points_test(void) const{
    //--------------------------
    return m_generated_points_test;
    //--------------------------
}// end const size_t& RL::Generate::get_generated_points(void)
//--------------------------------------------------------------