#pragma once
//--------------------------------------------------------------
// User Defined library
//--------------------------------------------------------------
#include "Environment/RL/EnvironmentTestLoader.hpp"
#include "Environment/RL/RLEnvironment.hpp"
#include "Environment/RL/RLEnvironmentLoader.hpp"
#include "Environment/RL/RLEnvironmentShuffle.hpp"
#include "Environment/RL/RLEnvironmentShuffleLoader.hpp"
#include "Environment/RL/RLEnvironmentAtomic.hpp"
#include "Environment/RL/RLEnvironmentLoaderAtomic.hpp"
#include "Environment/RL/RLEnvironmentShuffleAtomic.hpp"
#include "Environment/RL/RLEnvironmentShuffleAtomicLoader.hpp"
//--------------------------------------------------------------
namespace RL {
    //--------------------------------------------------------------
    namespace Environment {
        //--------------------------------------------------------------
        enum TYPE{
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
            protected:
                //--------------------------------------------------------------
                template<typename T, typename COST_FUNCTION, typename... Args>
                static auto generate_environment(   std::vector<T>&& data, 
                                                    COST_FUNCTION&& costFunction,
                                                    const double& egreedy = 0.9,
                                                    const double& egreedy_final = 0.02,
                                                    const double& egreedy_decay = 500.,
                                                    const TYPE& type = TYPE::NORMAL);
                //----------------------------
                template<typename T, typename COST_FUNCTION, typename... Args>
                static auto generate_environment(   std::vector<T>&& data, 
                                                    COST_FUNCTION&& costFunction,
                                                    const size_t batch = 2,
                                                    const double& egreedy = 0.9,
                                                    const double& egreedy_final = 0.02,
                                                    const double& egreedy_decay = 500.,
                                                    const TYPE& type= TYPE::NORMAL);
                //----------------------------
                template<typename T>
                static auto generate_environment(std::vector<T>&& data, const size_t batch = 2);
                //--------------------------------------------------------------
        };// end class RLEnvironment
        //--------------------------------------------------------------
    }// end namespace Environment
    //--------------------------------------------------------------
}// end namespace RL
//--------------------------------------------------------------