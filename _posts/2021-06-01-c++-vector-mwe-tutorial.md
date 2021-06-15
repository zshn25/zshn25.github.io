---
layout: post
title:  "A minimal working example of a custom Vector class in C++"
description: "C++ tutorial on how to write a custom vector class"
image: images/c-coding-6205174_640.png
date:   2021-05-13 15:21:23 -0700
categories: c++ vector tutorial
author: Zeeshan Khan Suri
published: true
comments: true
---

<div class="pb-5 d-flex flex-wrap flex-justify-end">
          <a href="https://github.com/zshn25/VectorMatrix" role="button" target="_blank">
        <img class="notebook-badge-image" src="{{ "assets/badges/github.svg" | relative_url }}" alt="View On GitHub">
    </a>
        </div>

This post describes my minimum working implementation of a Vector class in C++, similar to the one from standard library's `std::vector` but with only required functionality. The objective is to implement a generic array of dynamic size, i.e. which grows in size as new data gets added to it. We need at least 2 variables to keep track of the capacity of the array and the current index (which is also the size of the data available in the array). We also need a way to access these variables.

```cpp
class Vector {
    size_t capacity_ = 0;   // current memory capacity
    size_t curr_idx_ = 0;   // current vector size (same as numel)

    public:
        size_t size() const {return curr_idx_;} // returns current size of our vector
        size_t capacity() const {return capacity_;} // returns current capacity of our vector
};
```

## Indexing the vector

We store a pointer to the first element of the array. With this, we can access every element of the array up to it's size. The pointer to the first element will be a private member. So, we also need a way to access the vector's elements for read and write purposes.

In order to access the vector's individual elements, we need a method to index the vector. This indexing method will be used for both reading and writing the vector's elements. We define an `operator[]` which returns the element's reference as follows

```cpp
template<class T> class Vector {

    T* vector_ = nullptr;   // pointer to first data element
    size_t capacity_ = 0;   // current memory capacity
    size_t curr_idx_ = 0;   // current vector size (same as numel)

    public:
        // ... //same as above

        // Element read/write access
        T& operator[](const size_t index); // return element reference at index
};

//// Definitions

// Element read/write access
template<class T>
T& Vector<T>::operator[](const size_t index)
{
    if (index >= curr_idx_)
        throw std::invalid_argument("Index must be less than vector's size");
    
    return vector_[index];
}
```

We need to allocate a dynamic memory using `new[]`, which reserves the input amount of memory for our array. . Dynamic allocation of memory means we also need to free the memory manually upon destruction.

```cpp
template<class T>  class Vector {
    T* vector_ = nullptr;   // pointer to first data element
    size_t capacity_ = 0;   // current memory capacity
    size_t curr_idx_ = 0;   // current vector size (same as numel)

    public:
        // Constructors 
        Vector() = default; // default constructor

        // Destructor
        ~Vector() {delete[] vector_;}
};
```

## Rule of 0/3/5: Defining copy constructor and copy assignment

Since we are explicitly defining a destructor for manual memory management, we also need to define the copy constructor and the copy assignment by the [rule of 0/3](https://stackoverflow.com/a/4172724/5984672).

```cpp
template<class T>  class Vector {
    T* vector_ = nullptr;   // pointer to first data element
    size_t capacity_ = 0;   // current memory capacity
    size_t curr_idx_ = 0;   // current vector size (same as numel)

    public:
        // Constructors 
        Vector() = default; // default constructor
        Vector(const Vector<T>& another_vector);    // copy constructor
        Vector<T>& operator=(const Vector<T> &);    // copy assignment

        // Destructor
        ~Vector() {delete[] vector_;}
};
```

The copy constructor is used to copy an object of the same type. In our case, we need the copy constructor to copy another vector's elements. Note that when not defined explicitly, the compiler defines a default copy constructor which does not do what we want. So, it is necessary to define a copy constructor

```cpp
// Declaration same as before

//// Definitions

// Copy constructor
template<class T>
Vector<T>::Vector(const Vector<T>& another_vector)
{
    delete[] vector_;   // Delete before copying everything from another vector

    // Copy everything from another vector
    curr_idx_ = another_vector.size();
    capacity_ = another_vector.capacity();
    vector_ = new T[capacity_];
    for (size_t i=0; i < capacity_; ++i)
        vector_[i] = another_vector[i];
}
```

The copy assignment is similar to the copy constructor but is called when the `=` operator is used, for e.g. `Vector vector = another_vector;`. The only difference here will be that the copy assignment will return the pointer to the object while the copy constructor doesn't have to return anything.

```cpp
// Declaration same as before

//// Definitions

// Copy assignment
template<class T>
Vector<T>& Vector<T>::operator=(const Vector<T>& another_vector)
{
    delete[] vector_;   // Delete before copying everything from another vector

    // Copy everything from another vector
    curr_idx_ = another_vector.size();
    capacity_ = another_vector.capacity();
    vector_ = new T[capacity_];
    for (size_t i=0; i < capacity_; ++i)
        vector_[i] = another_vector[i];

    return *this;
}
```

## Adding and removing elements from the vector

We need a way to add elements to our vector. We can define an additional constructor (apart from the default one), which takes `capacity` as input and allocates memory of the given capacity. We can further extend this constructor to initialize our vector with a default value.

```cpp
template<class T>  class Vector {
    T* vector_ = nullptr;   // pointer to first data element
    size_t capacity_ = 0;   // current memory capacity
    size_t curr_idx_ = 0;   // current vector size (same as numel)

    public:
        // Constructors 
        Vector() = default; // default constructor
        Vector(const Vector<T>& another_vector);    // copy constructor
        Vector<T>& operator=(const Vector<T> &);    // copy assignment
        Vector(size_t capacity, T initial = T{});   // constructor based on capacity and a default value

        // Destructor
        ~Vector() {delete[] vector_;}
};

//// Definitions

template<class T>
Vector<T>::Vector(size_t capacity, T initial): capacity_{capacity},
                                 curr_idx_{capacity},
                                 vector_{new T[capacity]{}} // allocate stack and store its pointer
{
    for (size_t i=0; i < capacity; ++i)
        vector_[i] = initial;   // initialize
}
```

The above constructor can be called as follows

```cpp
Vector<int> vector(10);  // initializes a vector with capacity 10
Vector<int> vector_ones(10, 1)   // initializes with an initial value
```

At this point, a minimum working example of a static array is complete. If we also need to make our vector dynamic, we need a way to increase the vector's capacity if needed. This will be useful if we want to add elements to the array after we initialize it.

### Push back and pop methods

Since the goal of our array is to be dynamic, we also need ways to add and remove elements from it after initialization. To add and remove elements from our vector, we define `emplace_back` and `pop` methods respectively. The array also needs to increase its capacity if it needs to. For this, we define a private `reserve` method, which reserves the input amount of memory.

```cpp
template<class T> class Vector {
    public:
        // ... //same as above

        void emplace_back(const T& element);    // pass element by constant reference
        T pop();    // pops the last element

    private:
        // ... // same as above
        
        void reserve(const size_t capacity);
};

//// Definitions

// ... // same as above

template<class T>
void Vector<T>::emplace_back(const T& element)
{
    // If no cacacity, increase capacity
    if (curr_idx_ == capacity_)
    {
        if (capacity_ == 0) // handing initial when 
            reserve(8);
        else
            reserve(capacity_*2);
    }

    // Append an element to the array
    vector_[curr_idx_] = element;
    curr_idx_++;
}

template<class T>
T Vector<T>::pop()
{   
    if (curr_idx_ > 0)  // Nothing to pop otherwise
    {
        T to_return = vector_[curr_idx_-1]; // store return value before deleting
        // vector_[curr_idx_-1]->~T();         // delete from memory
        curr_idx_--;

        return to_return;
    }
    else
        throw std::out_of_range("Nothing to pop")
}


```

While appending an element to our vector, the `emplace_back` function will check if the maximum capacity of the array is reached. This is done by comparing the `curr_idx_` (size) of our vector with it's `capacity_`. If capacity is same as size, then we reserve more memory. We do this inside the `reserve` function as follows

```cpp
// Memory allocation
template<class T>
inline void Vector<T>::reserve(const size_t capacity)
{
    // Handle case when given capacity is less than equal to size. (No need to reallocate)
    if (capacity > curr_idx_)
    {
        // Reserves memory of size capacity for the vector_
        T* temp = new T[capacity];

        // Move previous elements to this memory
        for (size_t i=0; i < capacity_; ++i)
            temp[i] = vector_[i];

        delete[] vector_; // Delete old vector
        capacity_ = capacity;
        vector_ = temp;     // Copy assignment
    }
}
```

This completes our minimal working example of a vector class. In the next tutorials, we will extend this class to have iterators, a print function and mathematical operators.

Note that this implementation is just a minimal working example and is mainly for learning and educational purposes. The standard library's vector must be preferred in practice.

___

© Zeeshan Khan Suri, [<i class="fab fa-creative-commons"></i> <i class="fab fa-creative-commons-by"></i> <i class="fab fa-creative-commons-nc"></i>](http://creativecommons.org/licenses/by-nc/4.0/)

If this article was helpful to you, consider citing

```latex
@misc{suri_cpp_vector_class_2021,
      title={A minimal working example of a custom Vector class in C\+\+},
      url={https://zshn25.github.io/c++-vector-mwe-tutorial/}, 
      journal={Curiosity}, 
      author={Suri, Zeeshan Khan}, 
      year={2021}, 
      month={June}}
```

<div class="pb-5 d-flex flex-wrap flex-justify-end">
          <a href="https://github.com/zshn25/VectorMatrix" role="button" target="_blank">
        <img class="notebook-badge-image" src="{{ "assets/badges/github.svg" | relative_url }}" alt="View On GitHub">
    </a>
        </div>