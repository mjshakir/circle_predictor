//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Environment/RL/Environment.hpp"
//--------------------------------------------------------------
// User Defined library
//--------------------------------------------------------------
#include "Environment/RL/RLEnvironment.hpp"
#include "Environment/RL/RLEnvironmentLoader.hpp"
#include "Environment/RL/RLEnvironmentShuffle.hpp"
#include "Environment/RL/RLEnvironmentShuffleLoader.hpp"
#include "Environment/RL/RLEnvironmentAtomic.hpp"
#include "Environment/RL/RLEnvironmentLoaderAtomic.hpp"
#include "Environment/RL/RLEnvironmentShuffleAtomic.hpp"
#include "Environment/RL/RLEnvironmentShuffleAtomicLoader.hpp"
#include "Environment/RL/RLEnvironmentTest.hpp"
#include "Environment/RL/RLEnvironmentTestLoader.hpp"
//--------------------------------------------------------------
// Standard cpp library
//--------------------------------------------------------------
#include <memory>
//--------------------------------------------------------------
template<typename T, typename COST_OUTPUT, typename... Args>
auto RL::Environment::Environment::CreateEnvironment(   std::vector<T>&& data, 
                                                        std::function<COST_OUTPUT(const Args&...)> costFunction,
                                                        const TYPE& type,
                                                        const double& egreedy,
                                                        const double& egreedy_final,
                                                        const double& egreedy_decay){
    //----------------------------
    return generate_environment(std::move(data), std::move(costFunction), egreedy, egreedy_final, egreedy_decay, type);
    //----------------------------
}//end auto RL::Environment::Environment::Environment::CreateEnvironment
//--------------------------------------------------------------
template<typename T, typename COST_OUTPUT, typename... Args>
auto RL::Environment::Environment::CreateEnvironment(   std::vector<T>&& data, 
                                                        std::function<COST_OUTPUT(const Args&...)> costFunction,
                                                        const TYPE& type,
                                                        const size_t& batch,
                                                        const double& egreedy,
                                                        const double& egreedy_final,
                                                        const double& egreedy_decay){
    //----------------------------
    return generate_environment(std::move(data), std::move(costFunction), batch, egreedy, egreedy_final, egreedy_decay, type);
    //----------------------------
}// end auto RL::Environment::Environment::CreateEnvironmen
//--------------------------------------------------------------
template<typename T>
auto RL::Environment::Environment::CreateEnvironment(std::vector<T>&& data){
    //----------------------------
    return generate_environment(std::move(data));
    //----------------------------
}// end auto RL::Environment::Environment::CreateEnvironment(std::vector<T>&& data)
//--------------------------------------------------------------
template<typename T>
auto RL::Environment::Environment::CreateEnvironment(std::vector<T>&& data, const size_t& batch){
    //----------------------------
    return generate_environment(std::move(data), batch);
    //----------------------------
}//end auto RL::Environment::Environment::CreateEnvironment(std::vector<T>&& data, const size_t batch)
//--------------------------------------------------------------
template<typename T, typename COST_OUTPUT, typename... Args>
auto RL::Environment::Environment::generate_environment(std::vector<T>&& data, 
                                                        std::function<COST_OUTPUT(const Args&...)> costFunction,
                                                        const double& egreedy,
                                                        const double& egreedy_final,
                                                        const double& egreedy_decay,
                                                        const TYPE& type){
    //----------------------------
    switch(type){
        //----------------------------
        case NORMAL:
            return std::make_unique<RL::Environment::RLEnvironment>(std::move(data),
                                                                    std::move(costFunction),
                                                                    egreedy,
                                                                    egreedy_final,
                                                                    egreedy_decay);
        //----------------------------
        case ATOMIC:
            return std::make_unique<RL::Environment::RLEnvironmentAtomic>(  std::move(data),
                                                                            std::move(costFunction),
                                                                            egreedy,
                                                                            egreedy_final,
                                                                            egreedy_decay);
        //----------------------------
        case SHUFFLE:
            return std::make_unique<RL::Environment::RLEnvironmentShuffle>( std::move(data),
                                                                            std::move(costFunction),
                                                                            egreedy,
                                                                            egreedy_final,
                                                                            egreedy_decay);
        //----------------------------
        case ATOMIC_SHUFFLE:
            return std::make_unique<RL::Environment::RLEnvironmentShuffleAtomic>(   std::move(data),
                                                                                    std::move(costFunction),
                                                                                    egreedy,
                                                                                    egreedy_final,
                                                                                    egreedy_decay);
        //----------------------------
        default:
            //----------------------------
            throw std::invalid_argument("TYPE be between" + std::to_string(TYPE::NORMAL) + " and " + std::to_string(TYPE::ATOMIC_SHUFFLE));
            //----------------------------
    }//end switch(type)
    //----------------------------
}//end auto RL::Environment::Environment::generate_environment
//--------------------------------------------------------------
template<typename T, typename COST_OUTPUT, typename... Args>
auto RL::Environment:: Environment::generate_environment(   std::vector<T>&& data, 
                                                            std::function<COST_OUTPUT(const Args&...)> costFunction,
                                                            const size_t& batch,
                                                            const double& egreedy,
                                                            const double& egreedy_final,
                                                            const double& egreedy_decay,
                                                            const TYPE& type){
    //----------------------------
    switch(type){
        //----------------------------
        case NORMAL:
            return std::make_unique<RL::Environment::RLEnvironmentLoader>(  std::move(data),
                                                                            std::move(costFunction),
                                                                            batch,
                                                                            egreedy,
                                                                            egreedy_final,
                                                                            egreedy_decay);
        //----------------------------
        case ATOMIC:
            return std::make_unique<RL::Environment::RLEnvironmentLoaderAtomic>(std::move(data),
                                                                                std::move(costFunction),
                                                                                batch,
                                                                                egreedy,
                                                                                egreedy_final,
                                                                                egreedy_decay);
        //----------------------------
        case SHUFFLE:
            return std::make_unique<RL::Environment::RLEnvironmentShuffleLoader>(   std::move(data),
                                                                                    std::move(costFunction),
                                                                                    batch,
                                                                                    egreedy,
                                                                                    egreedy_final,
                                                                                    egreedy_decay);
        //----------------------------
        case ATOMIC_SHUFFLE:
            return std::make_unique<RL::Environment::RLEnvironmentShuffleAtomicLoader>( std::move(data),
                                                                                        std::move(costFunction),
                                                                                        batch,
                                                                                        egreedy,
                                                                                        egreedy_final,
                                                                                        egreedy_decay);
        //----------------------------
        default:
            //----------------------------
            throw std::invalid_argument("TYPE be between" + std::to_string(TYPE::NORMAL) + " and " + std::to_string(TYPE::ATOMIC_SHUFFLE));
            //----------------------------
    }//end switch(type)
    //----------------------------     
}//end auto RL::Environment::Environment::generate_environment
//--------------------------------------------------------------
template<typename T>
auto RL::Environment::Environment::generate_environment(std::vector<T>&& data){
    //----------------------------
    return std::make_unique<RL::Environment::RLEnvironmentTest<T>>(std::move(data));
    //----------------------------
}// end auto RL::Environment::Environment::generate_environment(std::vector<T>&& data)
//--------------------------------------------------------------
template<typename T>
auto RL::Environment::Environment::generate_environment(std::vector<T>&& data, const size_t& batch){
    //----------------------------
    return std::make_unique<RL::Environment::RLEnvironmentTestLoader<T>>(std::move(data), batch);
    //----------------------------
}// end auto RL::Environment::Environment::generate_environment(std::vector<T>&& data, const size_t batch)
//--------------------------------------------------------------