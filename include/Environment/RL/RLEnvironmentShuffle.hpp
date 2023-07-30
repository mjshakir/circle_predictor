#pragma once
//--------------------------------------------------------------
/* From: https://stackoverflow.com/questions/14803112/short-way-to-stdbind-member-function-to-object-instance-without-binding-param 
    and https://stackoverflow.com/questions/70355767/binding-a-class-method-to-a-method-of-another-class 
    and https://stackoverflow.com/questions/28746744/passing-capturing-lambda-as-function-pointer */
//--------------------------------------------------------------
// User Defined library
//--------------------------------------------------------------
#include "Environment/RL/RLEnvironment.hpp"
//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <random>
//-------------------
#include <algorithm>
#include <execution>
//--------------------------------------------------------------
namespace RL {
    //--------------------------------------------------------------
    namespace Environment {
        //--------------------------------------------------------------
        template<typename T, typename COST_OUTPUT, typename... Args>
        //--------------------------------------------------------------
        class RLEnvironmentShuffle : public RLEnvironment<T, COST_OUTPUT, Args...> {
            //--------------------------------------------------------------
            public:
                //--------------------------------------------------------------
                RLEnvironmentShuffle(void) = delete;
                //--------------------------------------------------------------
                virtual ~RLEnvironmentShuffle(void) = default;
                //--------------------------------------------------------------
                /**
                 * @brief Construct to create a training environment for reinforcement learning 
                 * 
                 * @param data          [in] : Data Vector of the template type T
                 * @param costFunction  [in] : Cost function that return the reward. needs to define the function return and parameters
                 * @param egreedy       [in] : The starting egreedy                                           @default: 0.9
                 * @param egreedy_final [in] : The egreedy number where it will change                        @default: 0.02
                 * @param egreedy_decay [in] : The egreedy exponential (e^x) decay factor                     @default: 500.
                 * @param batch         [in] : The batch number (needs to be less then half of the data size) @default: 1ul   
                 */
                explicit RLEnvironmentShuffle(  std::vector<T>&& data, 
                                                std::function<COST_OUTPUT(const Args&...)>&& costFunction,
                                                const double& egreedy = 0.9,
                                                const double& egreedy_final = 0.02,
                                                const double& egreedy_decay = 500.) :   RLEnvironment<T, COST_OUTPUT, Args...>( std::move(data), 
                                                                                                                                std::move(costFunction), 
                                                                                                                                egreedy, 
                                                                                                                                egreedy_final, 
                                                                                                                                egreedy_decay),
                                                                                        m_distribution(create_distribution(this->size())){
                    //-------------------------- 
                }// end RLEnvironmentShuffle(Dataset&& data_loader)
                //--------------------------------------------------------------
                //Define copy constructor explicitly
                RLEnvironmentShuffle(const RLEnvironmentShuffle& other) :   RLEnvironment<T, COST_OUTPUT, Args...>(other),
                                                                            m_distribution(other.m_distribution){
                    //--------------------------
                }// end RLEnvironmentShuffle(const RLEnvironmentShuffle& other)
                //--------------------------------------------------------------
                //Copy assignment operator
                RLEnvironmentShuffle& operator=(const RLEnvironmentShuffle& other) {
                    //--------------------------
                    // Check for self-assignment
                    if (this == &other) {
                        return *this;
                    }// end if (this == &other)
                    //--------------------------
                    // Perform a deep copy of the data
                    RLEnvironment<T, COST_OUTPUT, Args...>::operator=(other);
                    m_distribution  = other.m_distribution;
                    //--------------------------
                    return *this;
                    //--------------------------
                }// end RLEnvironmentShuffle& operator=(const RLEnvironmentShuffle& other)
                //--------------------------------------------------------------
                /**
                 * @brief 
                 * 
                 * @param args 
                 * @return std::tuple<torch::Tensor, COST_OUTPUT, double, bool> 
                 */
                virtual std::tuple<torch::Tensor, COST_OUTPUT, double, bool> step(const Args&... args) override{
                    //----------------------------
                    return internal_step(args...);
                    //----------------------------
                }// std::tuple<torch::Tensor, COST_OUTPUT, double, bool> step(const Args&... args)
                //--------------------------------------------------------------
                /**
                 * @brief 
                 * 
                 * @param epsilon 
                 * @param done 
                 * @param args 
                 * @return std::tuple<torch::Tensor, COST_OUTPUT> 
                 */
                virtual std::tuple<torch::Tensor, COST_OUTPUT> step(OUT double& epsilon, OUT bool& done, const Args&... args) override{
                    //----------------------------
                    return internal_step(epsilon, done, args...);
                    //----------------------------
                }// std::tuple<torch::Tensor, COST_OUTPUT> step(OUT double& epsilon, OUT bool& done, const Args&... args))
                //--------------------------------------------------------------
                /**
                 * @brief Get the first object
                 * 
                 * @return std::tuple<torch::Tensor, double> 
                 */
                virtual std::tuple<torch::Tensor, double> get_first(void) override{
                    //----------------------------
                    return get_first_internal();
                    //----------------------------
                }// end torch::Tensor get_first(void)
                //--------------------------------------------------------------
                /**
                 * @brief Get the first object
                 * 
                 * @param epsilon 
                 * @return torch::Tensor 
                 */
                virtual torch::Tensor get_first(OUT double& epsilon) override{
                    //----------------------------
                    return get_first_internal(epsilon);
                    //----------------------------
                }// end torch::Tensor get_first(void)
                //--------------------------------------------------------------
                /**
                 * @brief 
                 * 
                 */
                virtual void reset(void) override {
                    //--------------------------
                    reset_iterator();
                    //--------------------------
                }// end void reset(void)
                //--------------------------------------------------------------
            protected:
                //--------------------------------------------------------------
                /**
                 * @brief Construct to create a training environment for reinforcement learning 
                 * 
                 * @param data          [in] : Data Vector of the template type T
                 * @param costFunction  [in] : Cost function that return the reward. needs to define the function return and parameters
                 * @param egreedy       [in] : The starting egreedy                                           @default: 0.9
                 * @param egreedy_final [in] : The egreedy number where it will change                        @default: 0.02
                 * @param egreedy_decay [in] : The egreedy exponential (e^x) decay factor                     @default: 500.
                 * @param batch         [in] : The batch number (needs to be less then half of the data size) @default: 1ul   
                 */
                explicit RLEnvironmentShuffle(  std::vector<T>& data, 
                                                std::function<COST_OUTPUT(const Args&...)>& costFunction,
                                                const double& egreedy = 0.9,
                                                const double& egreedy_final = 0.02,
                                                const double& egreedy_decay = 500.) :   RLEnvironment<T, COST_OUTPUT, Args...>(std::move(data), 
                                                                                                                               std::move(costFunction), 
                                                                                                                               egreedy, 
                                                                                                                               egreedy_final, 
                                                                                                                               egreedy_decay),
                                                                                        m_distribution(create_distribution(this->size())){
                    //-------------------------- 
                }// end end RLEnvironmentShuffle(Dataset&& data_loader)
                //--------------------------------------------------------------
                virtual std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_step(const Args&... args) override{
                    //--------------------------
                    if (this->get_iterator() == this->get_data().end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (this->get_iterator() == this->get_data().end())
                    //--------------------------
                    if (this->get_iterator() == this->get_data().begin()){
                        //--------------------------
                        torch::Tensor _data;
                        double epsilon;
                        //--------------------------
                        std::tie(_data, epsilon) = get_first_internal();
                        //--------------------------
                        return {_data, torch::tensor(0), epsilon, false};
                        //--------------------------
                    }// end if (this->get_iterator() == this->get_data().begin())
                    //--------------------------
                    auto _random_position = generate_random_position();
                    //--------------------------
                    if(this->get_iterator() == this->get_data().end()-1){
                        //--------------------------
                        return {*std::next(this->get_data().begin(), _random_position), this->cost_function(args...), this->calculate_epsilon(), true};
                        //--------------------------
                    }// if(this->get_iterator() == this->get_data().end())
                    //--------------------------
                    ++this->get_iterator();
                    //--------------------------
                    m_distribution.at(_random_position) = 0;
                    //--------------------------
                    return {*std::next(this->get_data().begin(), _random_position), this->cost_function(args...), this->calculate_epsilon(), false};
                    //--------------------------
                }// end std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_step(Args... args))
                //--------------------------------------------------------------
                virtual std::tuple<torch::Tensor, COST_OUTPUT> internal_step(OUT double& epsilon, OUT bool& done, const Args&... args) override{
                    //--------------------------
                    if (this->get_iterator() == this->get_data().end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (this->get_iterator() == this->get_data().end())
                    //--------------------------
                    if (this->get_iterator() == this->get_data().begin()){
                        //--------------------------
                        done = false;
                        //--------------------------
                        torch::Tensor _data = get_first_internal(epsilon);
                        //--------------------------
                        return {_data, torch::tensor(0)};
                        //--------------------------
                    }// end if (this->get_iterator() == this->get_data().begin())
                    //--------------------------
                    auto _random_position = generate_random_position();
                    //--------------------------
                    if(this->get_iterator() == this->get_data().end()-1){
                        //--------------------------
                        epsilon = this->calculate_epsilon();
                        done    = true;
                        //--------------------------
                        return {*std::next(this->get_data().begin(), _random_position), this->cost_function(args...)};
                        //--------------------------
                    }// if(this->get_iterator() == this->get_data().end())
                    //--------------------------
                    ++this->get_iterator();
                    //--------------------------
                    m_distribution.at(_random_position) = 0;
                    //--------------------------
                    epsilon = this->calculate_epsilon();
                    done    = false;
                    //--------------------------
                    return {*std::next(this->get_data().begin(), _random_position), this->cost_function(args...)};
                    //--------------------------
                }// end std::tuple<torch::Tensor, COST_OUTPUT> internal_step(double& epsilon, bool& done, Args... args)
                //--------------------------------------------------------------
                virtual std::tuple<torch::Tensor, double> get_first_internal(void) override{
                    //--------------------------
                    if (this->get_iterator() == this->get_data().end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (this->get_iterator() == this->get_data().end() or std::next(this->get_iterator(), batch) == this->get_data().end())
                    //--------------------------
                    if (this->get_iterator() == this->get_data().begin()){
                        //--------------------------
                        auto _random_position = generate_random_position();
                        //--------------------------
                        auto epsilon = this->calculate_epsilon();
                        //--------------------------
                        ++this->get_iterator();
                        //--------------------------
                        m_distribution.at(_random_position) = 0;
                        //--------------------------
                        return {*std::next(this->get_data().begin(), _random_position), epsilon};
                        //--------------------------
                    }// end if (this->get_iterator() == this->get_data().begin())
                    //--------------------------
                    return {torch::tensor(0), 0};
                    //--------------------------
                }// end torch::Tensor get_first_internal(void)
                //--------------------------------------------------------------
                virtual torch::Tensor get_first_internal(OUT double& epsilon) override{
                    //--------------------------
                    if (this->get_iterator() == this->get_data().end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (this->get_iterator() == this->get_data().end() or std::next(this->get_iterator(), batch) == this->get_data().end())
                    //--------------------------
                    if (this->get_iterator() == this->get_data().begin()){
                        //--------------------------
                        auto _random_position = generate_random_position();
                        //--------------------------
                        epsilon = this->calculate_epsilon();
                        //--------------------------
                        ++this->get_iterator();
                        //--------------------------
                        m_distribution.at(_random_position) = 0;
                        //--------------------------
                        return *std::next(this->get_data().begin(), _random_position);
                        //--------------------------
                    }// end if (this->get_iterator() == this->get_data().begin())
                    //--------------------------
                    epsilon = 0.;
                    //--------------------------
                    return torch::tensor(0);
                    //--------------------------
                }// end torch::Tensor get_first_internal(void)
                //--------------------------------------------------------------
                std::vector<uint8_t>& get_distribution(void) {
                    //--------------------------
                    return m_distribution;
                    //--------------------------
                }// end std::vector<uint8_t>& get_distribution(void)
                //--------------------------------------------------------------
                size_t generate_random_position(void) const {
                    //--------------------------
                    thread_local std::random_device rd;
                    thread_local std::mt19937 gen(rd());
                    thread_local std::discrete_distribution<> random_distribution(m_distribution.begin(), m_distribution.end());
                    //--------------------------
                    return static_cast<size_t>(random_distribution(gen));
                    //--------------------------
                }// end size_t generate_random_position(void) const
                //--------------------------------------------------------------
                virtual void reset_iterator(void) override{
                    //--------------------------
                    this->get_iterator() = this->get_data().begin();
                    //--------------------------
                    std::fill(std::execution::par, m_distribution.begin(), m_distribution.end(), 1u);
                    //--------------------------
                }// end void rest_iterator(void)
                //--------------------------------------------------------------
            private:
                //--------------------------------------------------------------
                std::vector<uint8_t> m_distribution;
                //--------------------------
                std::vector<uint8_t> create_distribution(const size_t& size){
                    //--------------------------
                    std::vector<uint8_t> _data;
                    //--------------------------
                    _data.reserve(size);
                    //----------------------------
                    std::fill_n(std::execution::par, std::back_inserter(_data), size, 1u);
                    //--------------------------
                    return _data;
                    //--------------------------
                }// end std::vector<uint8_t> create_distribution(const size_t& size)
            //--------------------------------------------------------------
        };// end class RLEnvironmentShuffle
        //--------------------------------------------------------------
    }// end namespace Environment
    //--------------------------------------------------------------
}// end namespace RL
//--------------------------------------------------------------