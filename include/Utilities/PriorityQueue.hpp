#pragma once

//--------------------------------------------------------------
// Standard library
//--------------------------------------------------------------
#include <vector>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <array>
#include <optional>
//--------------------------------------------------------------
namespace Utils{
    //--------------------------------------------------------------
    template<typename T, size_t StackSize = 0, typename Compare = std::less<T>>
    class PriorityQueue {
        //--------------------------------------------------------------
        public:
            //--------------------------------------------------------------
            PriorityQueue(Compare c = Compare()) : comp(c), count(0){
                //----------------------------
            }// end PriorityQueue(Compare c = Compare()) : comp(c), count(0)
            //----------------------------
            PriorityQueue(const PriorityQueue&)            = default;
            PriorityQueue& operator=(const PriorityQueue&) = default;
            //----------------------------
            PriorityQueue(PriorityQueue&&)                 = default;
            PriorityQueue& operator=(PriorityQueue&&)      = default;
            //----------------------------
            void push(const T& val) {
                //----------------------------
                insert(val);
                //----------------------------
            }// end void push(const T& val)
            //----------------------------
            template<typename... Args>
            void emplace(Args&&... args) {
                //----------------------------
                T obj(std::forward<Args>(args)...);
                //----------------------------
                insert(obj);
                //----------------------------
            }// end void emplace(Args&&... args)
            //----------------------------
            std::optional<T> pop_top() {
                //----------------------------
                return pop_top_interal();
                //----------------------------
            }// std::optional<T> pop_top()
            //----------------------------
            bool empty(void) const {
                //----------------------------
                std::lock_guard<std::mutex> lock(mtx);
                return count == 0;
                //----------------------------
            }// end bool empty(void) const
            //----------------------------
            size_t size(void) const {
                //----------------------------
                std::lock_guard<std::mutex> lock(mtx);
                return count;
                //----------------------------
            }// end  size_t size(void)
            //--------------------------------------------------------------
        protected:
            //--------------------------------------------------------------
            bool usingStack(void) const {
                //----------------------------
                return StackSize > 0 and count <= StackSize;
                //----------------------------
            }// end bool usingStack() const
            //----------------------------
            void stackPop(void) {
                //----------------------------
                std::make_heap(stackData.begin(), stackData.begin() + count, comp);
                count--;
                //----------------------------
            }// end void stackPop(void)
            //----------------------------
            void heapPop(void) {
                //----------------------------
                std::pop_heap(heapData.begin(), heapData.end(), comp);
                heapData.pop_back();
                //----------------------------
            }// end void heapPop(void)
            //----------------------------
            void insert(const T& val) {
                //----------------------------
                std::lock_guard<std::mutex> lock(mtx);
                //----------------------------
                if (usingStack()) {
                    //----------------------------
                    if (count < StackSize) {
                        //----------------------------
                        stackData[count++] = val;
                        std::push_heap(stackData.begin(), stackData.begin() + count, comp);
                        //----------------------------
                    } else {
                        //----------------------------
                        heapData.assign(stackData.begin(), stackData.end());
                        heapData.push_back(val);
                        std::make_heap(heapData.begin(), heapData.end(), comp);
                        count++;
                        //----------------------------
                    }// end else if (count < StackSize) 
                    //----------------------------
                } else {
                    //----------------------------
                    heapData.push_back(val);
                    std::push_heap(heapData.begin(), heapData.end(), comp);
                    count++;
                    //----------------------------
                }// end else if (usingStack())
                //----------------------------
            }// end void insert(const T& val)
            //----------------------------
            std::optional<T> pop_top_interal(void) {
                //----------------------------
                std::lock_guard<std::mutex> lock(mtx);
                //----------------------------
                if (empty()) {
                    return  std::nullopt; // No value to return
                }// end if (empty())
                //----------------------------
                T top;
                //----------------------------
                if (usingStack()) {
                    top = stackData.front();
                    stackPop();
                } else {
                    top = heapData.front();
                    heapPop();
                    count--;
                }// end else
                //----------------------------
                return top;
                //----------------------------
            }// end std::optional<T> pop_top_interal(void)         
            //--------------------------------------------------------------
        private:
            //--------------------------------------------------------------
            Compare comp;
            size_t count;
            std::array<T, StackSize> stackData;
            std::vector<T> heapData;
            mutable std::mutex mtx;
        //--------------------------------------------------------------
    };// end class PriorityQueue
    //--------------------------------------------------------------
}//end namespace Utils
//--------------------------------------------------------------