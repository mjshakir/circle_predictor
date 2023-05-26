#pragma once
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <iostream>
//-------------------
#include <functional>
#include <algorithm>
#include <execution>
//--------------------------------------------------------------
namespace RL {
    //--------------------------------------------------------------
    class Generate{
        //--------------------------------------------------------------
        public:
            //--------------------------------------------------------------
            Generate(void) = delete;
            //--------------------------
            /**
             *  @brief A constructor. Generate both X, Y center point of the circle, and the radius.  
             *  it also generate test data that 20% of the generated_points
             *  @param generated_points [in]: How many points to generate. The constructor ensures that it is an even number    @default: 60000
             */
            Generate(const size_t& generated_points = 60000);
            //--------------------------
            /**
             *  @brief A constructor. Generate both X, Y center point of the circle, and the radius. 
             *  it also generate test data that 20% of the generated_points
             *  @param generated_points      [in]: How many points to generate. The constructor ensures that it is an even number                  @default: 60000
             *  @param generated_points_test [in]: How many points to test data is generated. The constructor ensures that it is an even number    @default: 10000
             */
            Generate( const size_t& generated_points, const size_t& generated_points_test);
            //--------------------------
            /**
             *  @brief Getter: The data generated.  
             *  
             *  @return std::vector<torch::Tensor>: Return the data X,Y, Radius.
             */
            template<typename T, typename F, typename... Args>
            std::vector<T> get_input(F&& function, const Args&... args){
                //--------------------------
                return generate_value(m_generated_points, function, args...);
                //--------------------------
            }// end std::vector<T> get_input(std::function<T(const Args&...)> function, const Args&... args)
            //--------------------------
            /**
             *  @brief Getter: The test data generated.  
             *  
             *  @return std::vector<torch::Tensor>: Return the data X,Y, Radius.
             */
            template<typename T, typename F, typename... Args>
            std::vector<T> get_test_input(F&& function, const Args&... args){
                //--------------------------
                return generate_value(m_generated_points_test, function, args...);
                //--------------------------
            }// end std::vector<T> get_test_input(std::function<T(const Args&...)> function, const Args&... args)
            //--------------------------------------------------------------
        protected:
            //--------------------------------------------------------------
            template<typename T, typename F, typename... Args>
            std::vector<T> generate_value(const size_t& generated_points, F&& function, const Args&... args){
                //--------------------------
                std::vector<T> _data;
                _data.reserve(generated_points);
                //--------------------------
                std::generate_n(std::execution::par, std::inserter(_data, _data.begin()), generated_points, [&function, &args...](){return function(args...);});
                //--------------------------
                return _data;
                //--------------------------
            }// end std::vector<torch::Tensor> generate_value(const size_t& generated_points, F&& function, const Args&... args)
            //--------------------------
            const size_t& get_generated_points(void) const;
            //--------------------------
            const size_t& get_generated_points_test(void) const;
            //--------------------------------------------------------------
        private:
            //--------------------------------------------------------------
            size_t m_generated_points, m_generated_points_test;
            //--------------------------
            // std::vector<torch::Tensor> m_data, m_data_test;
        //--------------------------------------------------------------
    };// end class Generate
    //--------------------------------------------------------------
}// end namespace RL
//--------------------------------------------------------------