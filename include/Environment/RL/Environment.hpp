#pragma once
//--------------------------------------------------------------
// User Defined library
//--------------------------------------------------------------
// #include "Environment/RL/EnvironmentTestLoader.hpp"
// #include "Environment/RL/RLEnvironment.hpp"
// #include "Environment/RL/RLEnvironmentLoader.hpp"
// #include "Environment/RL/RLEnvironmentShuffle.hpp"
// #include "Environment/RL/RLEnvironmentShuffleLoader.hpp"
// #include "Environment/RL/RLEnvironmentAtomic.hpp"
// #include "Environment/RL/RLEnvironmentLoaderAtomic.hpp"
// #include "Environment/RL/RLEnvironmentShuffleAtomic.hpp"
// #include "Environment/RL/RLEnvironmentShuffleAtomicLoader.hpp"
//--------------------------------------------------------------
// Standard cpp library
//--------------------------------------------------------------
#include <iostream>
#include <functional>
//--------------------------------------------------------------
namespace RL {
    //--------------------------------------------------------------
    namespace Environment {
        //--------------------------------------------------------------
        // Forward declaration 
        //----------------------------
        // template<typename T, typename COST_OUTPUT, typename... Args>
        // class RLEnvironment;
        // //----------------------------
        // template<typename T, typename COST_OUTPUT, typename... Args>
        // class RLEnvironmentLoader;
        // //----------------------------
        // template<typename T, typename COST_OUTPUT, typename... Args>
        // class RLEnvironmentShuffle;
        // //----------------------------
        // template<typename T, typename COST_OUTPUT, typename... Args>
        // class RLEnvironmentShuffleLoader;
        // //----------------------------
        // template<typename T, typename COST_OUTPUT, typename... Args>
        // class RLEnvironmentAtomic;
        // //----------------------------
        // template<typename T, typename COST_OUTPUT, typename... Args>
        // class RLEnvironmentLoaderAtomic;
        // //----------------------------
        // template<typename T, typename COST_OUTPUT, typename... Args>
        // class RLEnvironmentShuffleAtomic;
        // //----------------------------
        // template<typename T, typename COST_OUTPUT, typename... Args>
        // class RLEnvironmentShuffleAtomicLoader;
        // //----------------------------
        // template<typename T>
        // class EnvironmentTestLoader;
        //--------------------------------------------------------------
        // Define the types Environment type
        //----------------------------
        enum TYPE : uint8_t{
            //----------------------------
            NORMAL = 0,
            ATOMIC,
            SHUFFLE,
            ATOMIC_SHUFFLE
            //----------------------------
        };//end enum TYPE
        //--------------------------------------------------------------
        class Environment {
            //--------------------------------------------------------------
            public:
                //--------------------------------------------------------------
                template<typename T, typename COST_OUTPUT, typename... Args>
                static auto CreateEnvironment(  std::vector<T>&& data, 
                                                std::function<COST_OUTPUT(const Args&...)> costFunction,
                                                const TYPE& type = TYPE::NORMAL,
                                                const double& egreedy = 0.9,
                                                const double& egreedy_final = 0.02,
                                                const double& egreedy_decay = 500.);
                //----------------------------
                template<typename T, typename COST_OUTPUT, typename... Args>
                static auto CreateEnvironment(  std::vector<T>&& data, 
                                                std::function<COST_OUTPUT(const Args&...)> costFunction,
                                                const TYPE& type = TYPE::NORMAL,
                                                const size_t& batch = 2,
                                                const double& egreedy = 0.9,
                                                const double& egreedy_final = 0.02,
                                                const double& egreedy_decay = 500.);
                //----------------------------
                template<typename T>
                static auto CreateEnvironment(std::vector<T>&& data);
                //----------------------------
                template<typename T>
                static auto CreateEnvironment(std::vector<T>&& data, const size_t& batch);
                //--------------------------------------------------------------
            protected:
                //--------------------------------------------------------------
                template<typename T, typename COST_OUTPUT, typename... Args>
                static auto generate_environment(   std::vector<T>&& data, 
                                                    std::function<COST_OUTPUT(const Args&...)> costFunction,
                                                    const double& egreedy,
                                                    const double& egreedy_final,
                                                    const double& egreedy_decay ,
                                                    const TYPE& type);
                //----------------------------
                template<typename T, typename COST_OUTPUT, typename... Args>
                static auto generate_environment(   std::vector<T>&& data, 
                                                    std::function<COST_OUTPUT(const Args&...)> costFunction,
                                                    const size_t& batch,
                                                    const double& egreedy,
                                                    const double& egreedy_final,
                                                    const double& egreedy_decay,
                                                    const TYPE& type);
                //----------------------------
                template<typename T>
                static auto generate_environment(std::vector<T>&& data);
                //----------------------------
                template<typename T>
                static auto generate_environment(std::vector<T>&& data, const size_t& batch);
                //--------------------------------------------------------------
        };// end class RLEnvironment
        //--------------------------------------------------------------
    }// end namespace Environment
    //--------------------------------------------------------------
}// end namespace RL
//--------------------------------------------------------------