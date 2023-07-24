#pragma once

//--------------------------------------------------------------
// User defind library
//--------------------------------------------------------------
// Environment
//--------------------------
#include "Environment/RL/Environment.hpp"
//--------------------------
// Network Handling
//--------------------------
#include "Network/RL/ReinforcementNetworkHandlingDQN.hpp"
//--------------------------
// Memory Replay
//--------------------------
#include "Network/RL/ExperienceReplay.hpp"
//--------------------------------------------------------------
namespace Utils{
    //--------------------------------------------------------------
    template <typename ENVIRONMENT, typename... Args>
    struct CheckEnvironment;
    //--------------------------
    template <typename HANDLER, typename... Args>
    struct CheckHandler; 
    //--------------------------
    template <typename MEMORY, typename... Args>
    struct CheckExperienceReplay;      
    //--------------------------------------------------------------
    template <template <typename...> typename ENVIRONMENT, typename T, typename COST_OUTPUT, typename... Args>
    struct CheckEnvironment<ENVIRONMENT<T, COST_OUTPUT, Args...>> {
        //--------------------------
        static constexpr bool value = (sizeof...(Args) > 0) and
                                        (std::is_same_v<ENVIRONMENT<T, COST_OUTPUT, Args...>, RL::Environment::RLEnvironment <T, COST_OUTPUT, Args...>> or
                                        std::is_same_v<ENVIRONMENT<T, COST_OUTPUT, Args...>, RL::Environment::RLEnvironmentLoader <T, COST_OUTPUT, Args...>> or
                                        std::is_same_v<ENVIRONMENT<T, COST_OUTPUT, Args...>, RL::Environment::RLEnvironmentLoaderAtomic <T, COST_OUTPUT, Args...>> or
                                        std::is_same_v<ENVIRONMENT<T, COST_OUTPUT, Args...>, RL::Environment::RLEnvironmentShuffleLoader <T, COST_OUTPUT, Args...>> or
                                        std::is_same_v<ENVIRONMENT<T, COST_OUTPUT, Args...>, RL::Environment::RLEnvironmentShuffleAtomicLoader <T, COST_OUTPUT, Args...>>);
        //--------------------------
    };// end struct CheckEnvironment<ENVIRONMENT<T, COST_OUTPUT, Args...>>
    //--------------------------
    template <template <typename...> typename HANDLER, typename Network, typename... Args>
    struct CheckHandler<HANDLER<Network, Args...>> {
        //--------------------------
        static constexpr bool value = (sizeof...(Args) > 0) and
                                        (std::is_same_v<HANDLER<Network, Args...>, ReinforcementNetworkHandling <Network, Args...>> or
                                        std::is_same_v<HANDLER<Network, Args...>, ReinforcementNetworkHandlingDQN <Network, Args...>>);
        //--------------------------
    };// end struct CheckHandler<HANDLER<Network, Args...>>
    //--------------------------
    template <template <typename...> typename MemoryType, typename... Args>
    struct CheckExperienceReplay<MemoryType<Args...>> {
        //--------------------------
        static constexpr bool value = (sizeof...(Args) > 0) and std::is_same_v<MemoryType<Args...>, ExperienceReplay<Args...>>;
        //--------------------------
    };// end struct CheckExperienceReplay<MemoryType<Args...>>
    //--------------------------------------------------------------
}//end namespace CircleEquation
//--------------------------------------------------------------