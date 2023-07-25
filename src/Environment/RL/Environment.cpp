//--------------------------------------------------------------
// Main Header 
//--------------------------------------------------------------
#include "Environment/RL/Environment.hpp"
//--------------------------------------------------------------
// Standard cpp library
//--------------------------------------------------------------
#include <memory>
//--------------------------------------------------------------
template<typename T, typename COST_FUNCTION, typename... Args>
auto RL::Environment::Environment::generate_environment( std::vector<T>&& data, 
                                        COST_FUNCTION&& costFunction,
                                        const double& egreedy,
                                        const double& egreedy_final,
                                        const double& egreedy_decay,
                                        const TYPE& type){
    //----------------------------
    switch(type){
        //----------------------------
        case NORMAL:
            return std::make_unique<RL::Environment::RLEnvironment>( std::move(data),
                                                    std::move(costFunction),
                                                    egreedy,
                                                    egreedy_final,
                                                    egreedy_decay);
        //----------------------------
        case ATOMIC:
            return std::make_unique<RL::Environment::RLEnvironmentAtomic>(   std::move(data),
                                                            std::move(costFunction),
                                                            egreedy,
                                                            egreedy_final,
                                                            egreedy_decay);
        //----------------------------
        case SHUFFLE:
            return std::make_unique<RL::Environment::RLEnvironmentShuffle>(  std::move(data),
                                                            std::move(costFunction),
                                                            egreedy,
                                                            egreedy_final,
                                                            egreedy_decay);
        //----------------------------
        case ATOMIC_SHUFFLE:
            return std::make_unique<RL::Environment::RLEnvironmentShuffleAtomic>(std::move(data),
                                                                std::move(costFunction),
                                                                egreedy,
                                                                egreedy_final,
                                                                egreedy_decay);
        //----------------------------
        default:
            //----------------------------
            std::invalid_argument("TYPE be between" + std::to_string(TYPE::NORMAL) + " and " + std::to_string(TYPE::ATOMIC_SHUFFLE));
            //----------------------------
    }//end switch(type)
    //----------------------------
    return std::make_unique<RL::Environment::RLEnvironment>( std::move(data),
                                            std::move(costFunction),
                                            egreedy,
                                            egreedy_final,
                                            egreedy_decay);
    //----------------------------
}//end auto RL::Environment::Environment::generate_environment
//--------------------------------------------------------------
template<typename T, typename COST_FUNCTION, typename... Args>
auto RL::Environment:: Environment::generate_environment( std::vector<T>&& data, 
                                        COST_FUNCTION&& costFunction,
                                        const size_t batch,
                                        const double& egreedy,
                                        const double& egreedy_final,
                                        const double& egreedy_decay,
                                        const TYPE& type){
    //----------------------------
    switch(type){
        //----------------------------
        case NORMAL:
            return std::make_unique<RL::Environment::RLEnvironmentLoader>(   std::move(data),
                                                            std::move(costFunction),
                                                            batch,
                                                            egreedy,
                                                            egreedy_final,
                                                            egreedy_decay);
        //----------------------------
        case ATOMIC:
            return std::make_unique<RL::Environment::RLEnvironmentLoaderAtomic>( std::move(data),
                                                                std::move(costFunction),
                                                                batch,
                                                                egreedy,
                                                                egreedy_final,
                                                                egreedy_decay);
        //----------------------------
        case SHUFFLE:
            return std::make_unique<RL::Environment::RLEnvironmentShuffleLoader>(std::move(data),
                                                                std::move(costFunction),
                                                                batch,
                                                                egreedy,
                                                                egreedy_final,
                                                                egreedy_decay);
        //----------------------------
        case ATOMIC_SHUFFLE:
            return std::make_unique<RL::Environment::RLEnvironmentShuffleAtomicLoader>(  std::move(data),
                                                                        std::move(costFunction),
                                                                        batch,
                                                                        egreedy,
                                                                        egreedy_final,
                                                                        egreedy_decay);
        //----------------------------
        default:
            //----------------------------
            std::invalid_argument("TYPE be between" + std::to_string(TYPE::NORMAL) + " and " + std::to_string(TYPE::ATOMIC_SHUFFLE));
            //----------------------------
    }//end switch(type)
    //----------------------------
    return std::make_unique<RL::Environment::RLEnvironmentLoader>(   std::move(data),
                                                    std::move(costFunction),
                                                    batch,
                                                    egreedy,
                                                    egreedy_final,
                                                    egreedy_decay);
    //----------------------------        
}//end auto RL::Environment::Environment::generate_environment
//--------------------------------------------------------------
template<typename T>
auto RL::Environment::Environment::generate_environment(std::vector<T>&& data, const size_t batch){
    //----------------------------
    return std::make_unique<RL::Environment::EnvironmentTestLoader<T>>(std::move(data), batch);
    //----------------------------
}// end auto RL::Environment::Environment::generate_environment(std::vector<T>&& data, const size_t batch)
//--------------------------------------------------------------