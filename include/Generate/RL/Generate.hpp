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
            * @brief Construct a new RL::Generate object.
            * This constructor initializes an instance of the RL::Generate class.
            * It takes the number of generated points as a parameter and assigns it to the member variable m_generated_points.
            * If the provided number is odd, it subtracts 1 from it to make it even.
            * The constructor also calculates the value of m_generated_points_test by multiplying m_generated_points by 0.2.
            * 
            * @param generated_points The number of generated points.
            * @example usage:
            * RL::Generate generator(100); // Create a generator object with 100 generated points
            */
            Generate(const size_t& generated_points = 60000UL);
            //--------------------------
            /**
            * @brief Construct a new RL::Generate object.
            * This constructor initializes an instance of the RL::Generate class.
            * It takes the number of generated points and the number of test points as parameters and assigns them to the corresponding member variables.
            * If the provided number of generated points is odd, it subtracts 1 from it to make it even.
            * If the provided number of test points is odd and not equal to 0, it subtracts 1 from it to make it even.
            * 
            * @param generated_points The number of generated points.
            * @param generated_points_test The number of test points.
            * 
            * @example usage:
            * RL::Generate generator(100, 20); // Create a generator object with 100 generated points and 20 test points
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
                std::vector<T> _data(generated_points);
                //--------------------------
                std::generate(std::execution::par, _data.begin(), _data.end(), [&function, &args...](){return function(args...);});
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
            const size_t m_generated_points, m_generated_points_test;
        //--------------------------------------------------------------
    };// end class Generate
    //--------------------------------------------------------------
}// end namespace RL
//--------------------------------------------------------------