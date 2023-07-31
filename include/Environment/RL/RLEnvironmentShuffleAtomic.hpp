#pragma once
//--------------------------------------------------------------
// User Defined library
//--------------------------------------------------------------
#include "Environment/RL/RLEnvironmentAtomic.hpp"
#include "Environment/RL/RLEnvironmentShuffle.hpp"
//--------------------------------------------------------------
namespace RL {
    //--------------------------------------------------------------
    namespace Environment {
        //--------------------------------------------------------------
        template<typename T, typename COST_OUTPUT, typename... Args>
        //--------------------------------------------------------------
        class RLEnvironmentShuffleAtomic : protected RLEnvironmentAtomic<T, COST_OUTPUT, Args...>, protected RLEnvironmentShuffle<T, COST_OUTPUT, Args...>{
            //--------------------------------------------------------------
            public:
                //--------------------------------------------------------------
                RLEnvironmentShuffleAtomic(void) = delete;
                //--------------------------------------------------------------
                virtual ~RLEnvironmentShuffleAtomic(void) = default;
                //--------------------------------------------------------------
                /**
                * @brief Constructs an RLEnvironmentLoader object.
                * This class represents a loader for reinforcement learning environments. 
                * It takes as input a vector of data, a cost function, 
                * and optional parameters for epsilon-greedy exploration. 
                * The data is moved into the object, and the cost function and other parameters are stored for later use.
                * 
                * @tparam data The vector of data representing the RL environment.
                * @tparam costFunction The cost function to evaluate the RL environment.
                * @param egreedy The initial value of epsilon-greedy exploration @default: 0.9.
                * @param egreedy_final The final value of epsilon-greedy exploration @default: 0.02.
                * @param egreedy_decay The decay rate of epsilon-greedy exploration @default: 500.0.
                * @param batch The size of the batch for RL training @default: 1.
                * 
                * @note The template arguments T and Args represent the data type and additional argument types required by the cost function, respectively.
                * @note The data vector is moved into the RLEnvironmentLoader object, ensuring efficient data transfer.
                * @note The RLEnvironmentLoader class is derived from RLEnvironment and initializes its base class using the given parameters.
                * 
                * @example
                * // Create a vector of data representing the RL environment
                * std::vector<int> data = {1, 2, 3, 4, 5};
                * // Define a cost function
                * auto costFunction = [](const int& state) { return state * state; };
                * // Create an RLEnvironmentLoader object
                * RLEnvironmentLoader<int, int> loader(std::move(data), costFunction, 0.9, 0.02, 500.0, 1);
                *  @throws std::out_of_range If the specified batch size is greater than or equal to half the size of the data vector.
                *          The exception message provides details about the expected range for the batch size.
                */
                explicit RLEnvironmentShuffleAtomic(std::vector<T>&& data, 
                                                    std::function<COST_OUTPUT(const Args&...)>&& costFunction,
                                                    const double& egreedy = 0.9,
                                                    const double& egreedy_final = 0.02,
                                                    const double& egreedy_decay = 500.) :   RLEnvironment<T, COST_OUTPUT, Args...>( std::move(data),
                                                                                                                                    std::move(costFunction),
                                                                                                                                    egreedy, 
                                                                                                                                    egreedy_final, 
                                                                                                                                    egreedy_decay),
                                                                                            RLEnvironmentAtomic<T, COST_OUTPUT, Args...>(   this->get_data(),
                                                                                                                                            this->get_cost_function(),
                                                                                                                                            egreedy,
                                                                                                                                            egreedy_final,
                                                                                                                                            egreedy_decay),
                                                                                            RLEnvironmentShuffle<T, COST_OUTPUT, Args...>(  this->get_data(),
                                                                                                                                            this->get_cost_function(),
                                                                                                                                            egreedy,
                                                                                                                                            egreedy_final,
                                                                                                                                            egreedy_decay){
                    //----------------------------
                }// end RLEnvironmentShuffleAtomic(Dataset&& data_loader)
                //--------------------------------------------------------------
                RLEnvironmentShuffleAtomic(const RLEnvironmentShuffleAtomic&)            = default;
                RLEnvironmentShuffleAtomic& operator=(const RLEnvironmentShuffleAtomic&) = default;
                //----------------------------
                RLEnvironmentShuffleAtomic(RLEnvironmentShuffleAtomic&&)                 = default;
                RLEnvironmentShuffleAtomic& operator=(RLEnvironmentShuffleAtomic&&)      = default;
                //--------------------------------------------------------------
                /**
                * @brief Executes an internal step in the RL environment.
                *
                * This function performs an internal step in the RL environment, processing a batch of data
                * and returning a tuple containing the resulting tensor, cost output, epsilon value, and a flag
                * indicating whether the data iteration is complete.
                *
                * @tparam args The variadic arguments for the cost function.
                *
                * @return A tuple containing the resulting tensor, cost output, epsilon value, and a flag indicating
                *         whether the data iteration is complete.
                *
                * @throws std::out_of_range If the current data iterator is at the end or if advancing the iterator by
                *                          the batch size exceeds the end of the data.
                *
                * @note The data vector should be populated before calling this function.
                * @note The cost function should be set before calling this function.
                *
                * @example
                * size_t batchSize = 4;
                * RLEnvironmentLoader loader(data, costFunction, batchSize, 0.9, 0.02, 500.0);
                * 
                * auto result = loader.step(arg1, arg2);
                *
                * // Example usage of the returned tuple elements
                * torch::Tensor tensor = std::get<0>(result);
                * COST_OUTPUT costOutput = std::get<1>(result);
                * double epsilon = std::get<2>(result);
                * bool isComplete = std::get<3>(result);
                */
                virtual std::tuple<torch::Tensor, COST_OUTPUT, double, bool> step(const Args&... args) override {
                    //----------------------------
                    return internal_step(args...);
                    //----------------------------
                }// std::tuple<torch::Tensor, COST_OUTPUT, double, bool> step(const Args&... args)
                //--------------------------------------------------------------
                /**
                * @brief Performs an step in the algorithm.
                *
                * This function processes a batch of data, updates the epsilon value and done flag, and returns a tuple
                * containing a torch::Tensor representing the processed data and a COST_OUTPUT representing the cost output.
                *
                * @tparam Args The template parameter pack representing the additional arguments used for cost function calculations.
                * @param[out] epsilon The updated epsilon value based on internal calculations.
                * @param[out] done Indicates whether the internal step is completed or not.
                * @param batch The size of the batch to process.
                * @param args Additional arguments of type Args used for cost function calculations.
                * @return A tuple containing a torch::Tensor and a COST_OUTPUT.
                *     - The torch::Tensor represents the processed data obtained by concatenating the tensors in _data.
                *     - The COST_OUTPUT represents the cost output calculated based on the input arguments.
                * @throws std::out_of_range If the end of the data iterator is reached before processing the full batch.
                *
                * @details
                * The `internal_step` function is a member function of the class that performs an internal step in the algorithm.
                * It processes a batch of data and performs different actions based on the current state of the data iterator and the
                * batch size. The function updates the epsilon value and done flag according to the processing results and returns a
                * tuple containing the processed data and the cost output.
                *
                * The function starts by checking if the end of the data iterator or the end of the batch is reached. If either condition
                * is true, it throws a std::out_of_range exception with an appropriate error message.
                *
                * Next, the function initializes an empty vector `_data` and reserves space for `batch` tensors to optimize memory allocation.
                *
                * If the data iterator is at the beginning and the end of the batch is not the end of the data, the function calculates the
                * epsilon value using the `calculate_epsilon()` member function, sets the done flag to false, and adds the current data
                * tensor to `_data`. It then iterates over the remaining data tensors, adding them to `_data`. Finally, it returns a tuple
                * containing the concatenated tensor of `_data` along with a tensor of value 0 as the cost output.
                *
                * If the end of the batch is the end of the data iterator, the function iterates over the remaining data tensors, adds them
                * to `_data`, calculates the epsilon value, sets the done flag to true, and returns a tuple containing the concatenated
                * tensor of `_data` along with the cost output obtained by invoking the `m_CostFunction` member function with the provided
                * arguments `args`.
                *
                * If neither of the above cases is true, the function assumes that there are remaining data tensors to process. It iterates
                * over the batch size, adding the next data tensors to `_data`. It then calculates the epsilon value, sets the done flag to
                * false, and returns a tuple containing the concatenated tensor of `_data` along with the cost output obtained by invoking
                * the `m_CostFunction` member function with the provided arguments `args`.
                *
                * @note Make sure that the data iterator is properly initialized before calling this function.
                */
                virtual std::tuple<torch::Tensor, COST_OUTPUT> step(OUT double& epsilon, OUT bool& done, const Args&... args) override {
                    //----------------------------
                    return internal_step(epsilon, done, args...);
                    //----------------------------
                }// std::tuple<torch::Tensor, COST_OUTPUT> step(OUT double& epsilon, OUT bool& done, const Args&... args))
                //--------------------------------------------------------------
                /**
                 * @brief Get the first internal data batch.
                 *
                 * This function retrieves the first internal data batch from the class. It returns a tuple
                 * containing a Torch tensor and a double value. The Torch tensor represents the concatenated
                 * data batch, while the double value represents the epsilon calculated for the batch.
                 *
                 * @return A tuple containing the Torch tensor and epsilon value.
                 *
                 * @throw std::out_of_range If the end of the data iterator is reached before retrieving the batch.
                 *
                 * Example usage:
                 * @code{.cpp}
                 * RLEnvironmentLoader loader(data, costFunction, 0.9, 0.02, 500.0, batchSize);;
                 * auto result = loader.get_first();
                 * torch::Tensor batch = std::get<0>(result);
                 * double epsilon = std::get<1>(result);
                 * @endcode
                 */
                virtual std::tuple<torch::Tensor, double> get_first(void) override {
                    //----------------------------
                    return get_first_internal();
                    //----------------------------
                }// end torch::Tensor get_first(void)
                //--------------------------------------------------------------
                /**
                 * @brief Get the first internal batch of data.
                 *
                 * This function retrieves the first internal batch of data from the class. It returns a Torch tensor
                 * representing the batch of data. The epsilon value calculated during the retrieval is assigned to the
                 * output parameter `epsilon`.
                 *
                 * @param epsilon [out] The epsilon value calculated during the retrieval.
                 *
                 * @return The Torch tensor representing the first internal batch of data.
                 *
                 * @throw std::out_of_range If the end of the data iterator is reached before retrieving the batch.
                 *
                 * Example usage:
                 * @code{.cpp}
                 * size_t batch_size = 10;
                 * RLEnvironmentLoader loader(data, costFunction, 0.9, 0.02, 500.0, batchSize)
                 * double epsilon;
                 * torch::Tensor batch = loader.get_first(epsilon);
                 * @endcode
                 */
                virtual torch::Tensor get_first(OUT double& epsilon) override {
                    //----------------------------
                    return get_first_internal(epsilon);
                    //----------------------------
                }// end torch::Tensor get_first(void)
                //--------------------------------------------------------------
                /**
                 * @brief 
                 * 
                 */
                virtual void reset(void) override{
                    //--------------------------
                    reset_iterator();
                    //--------------------------
                }// end void reset(void)
                //--------------------------------------------------------------
            protected:
                //--------------------------------------------------------------
                virtual std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_step(const Args&... args) override {
                    //--------------------------------------------------------------
                    std::lock_guard<std::mutex> date_lock(m_mutex);
                    //--------------------------
                    auto _data_iter = this->get_atomic_iterator();
                    //--------------------------------------------------------------
                    if (_data_iter == this->get_data().end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (_data_iter == this->get_data().end())
                    //--------------------------------------------------------------
                    if (_data_iter == this->get_data().begin()){
                        //--------------------------
                        torch::Tensor _data;
                        double epsilon;
                        //--------------------------
                        std::tie(_data, epsilon) = get_first_internal();
                        //--------------------------
                        return {_data, torch::tensor(0), epsilon, false};
                        //--------------------------
                    }// end if (_data_iter == this->get_data().begin())
                    //--------------------------------------------------------------
                    auto _random_position = this->generate_random_position();
                    //--------------------------
                    if(_data_iter == this->get_data().end()-1){
                        //--------------------------
                        return {*std::next(this->get_data().begin(), _random_position), this->cost_function(args...), this->calculate_epsilon(_data_iter), true};
                        //--------------------------
                    }// if(_data_iter == this->get_data().end())
                    //--------------------------------------------------------------
                    ++_data_iter;
                    this->set_atomic_iterator(_data_iter);
                    //--------------------------
                    this->get_distribution().at(_random_position) = 0;
                    //--------------------------
                    return {*std::next(this->get_data().begin(), _random_position), this->cost_function(args...), this->calculate_epsilon(_data_iter), false};
                    //--------------------------
                }// end std::tuple<torch::Tensor, COST_OUTPUT, double, bool> internal_step(const size_t& batch, Args... args)
                //--------------------------------------------------------------
                virtual std::tuple<torch::Tensor, COST_OUTPUT> internal_step(OUT double& epsilon, OUT bool& done, const Args&... args) override {
                    //--------------------------------------------------------------
                    std::lock_guard<std::mutex> date_lock(m_mutex);
                    //--------------------------
                    auto _data_iter = this->get_atomic_iterator();
                    //--------------------------------------------------------------
                    if (_data_iter == this->get_data().end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (_data_iter == this->get_data().end() or std::next(_data_iter, batch) == this->get_data().end())
                    //--------------------------------------------------------------
                    if (_data_iter == this->get_data().begin()){
                        //--------------------------
                        done = false;
                        //--------------------------
                        torch::Tensor _data = get_first_internal(epsilon);
                        //--------------------------
                        return {_data, torch::tensor(0)};
                        //--------------------------
                    }// end if (_data_iter == this->get_data().begin() and std::next(_data_iter, batch) != this->get_data().end()-1)
                    //--------------------------------------------------------------
                    auto _random_position = this->generate_random_position();
                    //--------------------------
                    if(_data_iter == this->get_data().end()-1){
                        //--------------------------
                        epsilon = this->calculate_epsilon(_data_iter);
                        done = true;
                        //--------------------------
                        return {*std::next(this->get_data().begin(), _random_position), this->cost_function(args...)};
                        //--------------------------
                    }// if(_data_iter == this->get_data().end()-1)
                    //--------------------------------------------------------------
                    epsilon = this->calculate_epsilon(_data_iter);
                    done = false;
                    //--------------------------
                    ++_data_iter;
                    this->set_atomic_iterator(_data_iter);
                    //--------------------------
                    this->get_distribution().at(_random_position) = 0;
                    //--------------------------
                    return {*std::next(this->get_data().begin(), _random_position), this->cost_function(args...)};
                    //--------------------------
                }// end std::tuple<torch::Tensor, COST_OUTPUT> internal_step(double& epsilon, bool& done, const size_t& batch, Args... args)
                //--------------------------------------------------------------
                virtual std::tuple<torch::Tensor, double> get_first_internal(void) override {
                    //--------------------------
                    std::lock_guard<std::mutex> date_lock(m_mutex);
                    //--------------------------
                    auto _data_iter = this->get_atomic_iterator();
                    //--------------------------
                    if (_data_iter == this->get_data().end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (_data_iter == this->get_data().end() or std::next(_data_iter, batch) == this->get_data().end())
                    //--------------------------
                    if (_data_iter == this->get_data().begin()){
                        //--------------------------
                        auto _random_position = this->generate_random_position();
                        //--------------------------
                        auto epsilon = this->calculate_epsilon(_data_iter);
                        //--------------------------
                        ++_data_iter;
                        //--------------------------
                        this->set_atomic_iterator(_data_iter);
                        //--------------------------
                        this->get_distribution().at(_random_position) = 0;
                        //--------------------------
                        return {*std::next(this->get_data().begin(), _random_position), epsilon};
                        //--------------------------
                    }// end if (_data_iter == this->get_data().begin())
                    //--------------------------
                    return {torch::tensor(0), 0};
                    //--------------------------
                }// end torch::Tensor get_first_internal(void)
                //--------------------------------------------------------------
                virtual torch::Tensor get_first_internal(OUT double& epsilon) override {
                    //--------------------------
                    std::lock_guard<std::mutex> date_lock(m_mutex);
                    //--------------------------
                    auto _data_iter = this->get_atomic_iterator();
                    //--------------------------
                    if (_data_iter == this->get_data().end()){
                        //--------------------------
                        throw std::out_of_range("End Of The Data Iterator");
                        //--------------------------
                    }// end if (_data_iter == this->get_data().end())
                    //--------------------------
                    if (_data_iter == this->get_data().begin()){
                        //--------------------------
                        auto _random_position = this->generate_random_position();
                        //--------------------------
                        epsilon = this->calculate_epsilon(_data_iter);
                        //--------------------------
                        ++_data_iter;
                        //--------------------------
                        this->set_atomic_iterator(_data_iter);
                        //--------------------------
                        this->get_distribution().at(_random_position) = 0;
                        //--------------------------
                        return *std::next(this->get_data().begin(), _random_position);
                        //--------------------------
                    }// end (_data_iter == this->get_data().begin() and std::next(_data_iter, batch) != this->get_data().end()-1)
                    //--------------------------
                    epsilon = 0.;
                    //--------------------------
                    return torch::tensor(0);
                    //--------------------------
                }// end torch::Tensor get_first_internal(OUT double& epsilon, const size_t& batch)
                //--------------------------------------------------------------
                virtual void reset_iterator(void) override{
                    //--------------------------
                    std::lock_guard<std::mutex> date_lock(m_mutex);
                    //--------------------------
                    this->set_atomic_iterator(this->get_data().begin());
                    //--------------------------
                    std::fill(std::execution::par, this->get_distribution().begin(), this->get_distribution().end(), 1u);
                    //--------------------------
                }// end void rest_iterator(void)
                //--------------------------------------------------------------
            private:
                //--------------------------------------------------------------
                std::mutex m_mutex;
            //--------------------------------------------------------------
        };// end class RLEnvironmentShuffleAtomic
        //--------------------------------------------------------------
    }// end namespace Environment
    //--------------------------------------------------------------
}// end namespace RL
//--------------------------------------------------------------