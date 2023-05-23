#pragma once

#include <iostream>
#include <sstream>
#include <algorithm>
#include <type_traits>
#include <cassert>
#include <initializer_list>
#include <limits>
#include <iterator>

#ifdef EXPORT_TENSORND_TO_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#endif

inline void* aligned_malloc(size_t size, size_t align) {
    void* result;
#ifdef _MSC_VER 
    result = _aligned_malloc(size, align);
#else 
    if (posix_memalign(&result, align, size)) result = nullptr;
#endif
    std::cerr << "tensorND allocate " << result << std::endl;
    return result;
}

inline void aligned_free(void* ptr) {
#ifdef _MSC_VER 
    _aligned_free(ptr);
#else 
    free(ptr);
#endif
    std::cerr << "tensorND freeing  " << ptr << std::endl;
}

struct slice {
    int i0;
    int i1;
    slice() : i0(0), i1(std::numeric_limits<int>::max()) {} // whole dimension
    slice(int i0) : i0(i0), i1(i0 + 1) {}           // dimension collapsed
    slice(int i0, int i1) : i0(i0), i1(i1) {}   // normal
};

template<typename T, int RMAX = 8>
struct tensorND {
    constexpr static size_t cache_line_size = 64;

    T* data = nullptr;
    int ndim = 0;
    size_t capacity = 0;
    int shape[RMAX];
    int64_t strides[RMAX];

    tensorND() = default;

    // ownership of the data cannot be transfered but only stealed
    // since we choose to use raw pointer instead of shared_ptr.
    tensorND(tensorND<T, RMAX>& other) = delete;

    tensorND(tensorND<T, RMAX>&& other) {
        // steal ownership from other
        data = other.data;
        ndim = other.ndim;
        capacity = other.capacity;
        memcpy(shape, other.shape, sizeof(shape));
        memcpy(strides, other.strides, sizeof(strides));
        other.data = nullptr;
        other.capacity = 0;
    }

    template<typename ST>
    tensorND(void* _data, std::initializer_list<ST> _shape) {
        assert(_shape.size() <= RMAX);
        ndim = _shape.size();
        data = reinterpret_cast<T*>(_data);
        std::copy(_shape.begin(), _shape.end(), shape);
        // initialize strides as compact
        strides[ndim - 1] = sizeof(T);
        for (int i = ndim - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        capacity = 0;
    }

    template<typename ST>
    tensorND(void* _data, std::initializer_list<ST> _shape, std::initializer_list<ST> _strides) :
        tensorND(_data, std::begin(_shape), std::end(_shape), std::begin(_strides), std::end(_strides)) {}

    template<typename Container>
    tensorND(void* _data, const Container& _shape, const Container& _strides) :
        tensorND(_data, std::begin(_shape), std::end(_shape), std::begin(_strides), std::end(_strides)) {}

    template<typename ITSHAPE, typename ITSTRIDES>
    tensorND(void* _data, ITSHAPE itshape0, ITSHAPE itshape1, ITSTRIDES itstrides0, ITSTRIDES itstrides1) {
        auto size = std::distance(itshape0, itshape1);
        auto size2 = std::distance(itstrides0, itstrides1);
        assert(size <= RMAX);
        assert(size == size2);
        ndim = size;
        data = reinterpret_cast<T*>(_data);
        std::copy(itshape0, itshape1, shape);
        std::copy(itstrides0, itstrides1, strides);
        capacity = 0;
    }

    template<typename Container>
    tensorND(const Container& _shape, bool compact) {
        capacity = 0;
        resize(_shape, compact);
    }

    template<typename ST>
    tensorND(const std::initializer_list<ST>& _shape, bool compact) {
        capacity = 0;
        resize(_shape, compact);
    }

    ~tensorND() {
        if (data && capacity) {
            aligned_free(data);
        }
    }

    template<typename ST>
    void resize(const std::initializer_list<ST>& _shape, bool compact) {
        resize<std::initializer_list<ST>, 0>(_shape, compact);
    }

    template<typename Container, int tag = 0>
    void resize(const Container& _shape, bool compact) {
        assert(_shape.size() <= RMAX);
        ndim = _shape.size();
        std::copy(_shape.begin(), _shape.end(), shape);

        if (compact) {
            strides[ndim - 1] = sizeof(T);
            for (int i = ndim - 2; i >= 0; i--) {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        else {
            // ensure each non-last dimension starts from a cache-line boundary
            strides[ndim - 1] = sizeof(T);
            for (int i = ndim - 2; i >= 0; i--) {
                size_t next_stride_bytes = strides[i + 1] * shape[i + 1];
                size_t cache_line_tail = (next_stride_bytes % cache_line_size);
                if (cache_line_tail) {
                    // padding to multiple of cache line
                    next_stride_bytes += (cache_line_size - cache_line_tail);
                }
                strides[i] = next_stride_bytes;
            }
        }

        size_t total = strides[0] * shape[0];
        if (total > capacity) {
            if (data && capacity) aligned_free(data);
            data = reinterpret_cast<T*>(aligned_malloc(total, cache_line_size));
            capacity = total;
        }
    }



    template<typename F>
    void for_each(F f) {
        int coord[RMAX] = { 0 };
        size_t idx = 0;
        while (1) {
            f(idx, coord);

            // increase 1D linear index
            idx++;

            // increase 
            int carry = 1;
            for (int i = ndim - 1; i >= 0; i--) {
                coord[i] += carry;
                if (coord[i] < shape[i]) {
                    carry = 0;
                    break;
                }
                coord[i] -= shape[i];
            }
            if (carry) break;
        }
    }

    // TODO: data[idx] may access paddings  
    T& operator[](size_t idx) {
        return data[idx];
    }

    T& at_byte_offset(size_t offset) const {
        return *reinterpret_cast<T*>(reinterpret_cast<int8_t*>(data) + offset);
    }

    template<class IDX, size_t N>
    T& operator()(IDX(&coord)[N]) {
        assert(N >= ndim);
        size_t offset = 0;
        for (int i = 0; i < ndim; i++)
            offset += coord[i] * strides[i];
        return at_byte_offset(offset);
    }

    template<class IDX>
    T& operator()(IDX* coord) {
        size_t offset = 0;
        for (int i = 0; i < ndim; i++)
            offset += coord[i] * strides[i];
        return at_byte_offset(offset);
    }

    template<typename ... IDX>
    T& operator()(IDX ... idxs) {
        assert(sizeof...(IDX) == ndim);
        return at_byte_offset(get_element_offset<0>(idxs...));
    }

    template < typename ... IDX>
    tensorND<T> Slice(IDX ... idxs) {
        tensorND<T> ret;
        assert(sizeof...(idxs) == ndim);
        ret.data = data;
        ret.ndim = 0;
        ret.capacity = 0; // Slice do not own the data
        get_subview<0, 0>(ret, idxs...);
        return ret;
    }

    // overloaded Slice to create a clone without take ownership of the data
    tensorND<T> Slice() {
        return tensorND<T>(data, shape, shape + ndim, strides, strides + ndim);
    }

    std::string toString() {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const tensorND<T, RMAX>& t) {
        os << "tensor" << t.ndim << "D<" << typeid(T).name() << ">_shape=(" << t.shape[0];
        for (int i = 1; i < t.ndim; i++) os << "," << t.shape[i];
        os << ")_strides=(" << t.strides[0];
        for (int i = 1; i < t.ndim; i++) os << "," << t.strides[i];
        os << ")_address=" << t.data << std::endl;
        t.print_values(os);
        return os;
    }

    void print_row(std::ostream& os, int* coord, int limit = 99999) const {
        std::stringstream ss;
        ss << "[";
        const char* sep = "";
        for (int i = 0; i < ndim; i++) {
            ss << sep << coord[i];
            sep = ",";
        }
        ss << "] : ";
        sep = "";
        int k;
        for (k = 0; k < shape[ndim - 1]; k++) {
            size_t offset = strides[ndim - 1] * k;
            for (int i = 0; i < ndim - 1; i++)
                offset += strides[i] * coord[i];
            ss << sep << at_byte_offset(offset);
            sep = ",";
            if (ss.tellp() > limit)
                break;
        }
        if (k < shape[ndim - 1])
            ss << sep << "...";
        ss << std::endl;
        os << ss.str();
    }

    void print_values(std::ostream& os) const {
        int coord[RMAX] = { 0 };
        while (1) {
            print_row(os, coord, 80);
            int carry = 1;
            for (int i = ndim - 2; i >= 0; i--) {
                coord[i] += carry;
                if (coord[i] < shape[i]) {
                    carry = 0;
                    break;
                }
                coord[i] -= shape[i];
            }
            if (carry) break;
        }
    }

private:
    template<int N>
    int64_t get_element_offset() const {
        return 0;
    }

    template<int N, typename I0, typename ... IDX>
    int64_t get_element_offset(I0 i0, IDX ... idxs) const {
        return i0 * strides[N] + get_element_offset<N + 1>(idxs...);
    }

    // I0 is range
    template<int isrc, int idst, typename TV, typename ... IDX>
    void get_subview(TV& t, slice i0, IDX ... idxs) const {
        // i0 is range, so it will occupy idst'th output shape & strides
        t.data = reinterpret_cast<T*>(reinterpret_cast<int8_t*>(t.data) + i0.i0 * strides[isrc]);
        t.shape[idst] = std::min(shape[isrc], i0.i1) - i0.i0;
        t.strides[idst] = strides[isrc];
        t.ndim++;
        get_subview<isrc + 1, idst + 1>(t, idxs...);
    }

    template<int isrc, int idst, typename TV>
    void get_subview(TV& t, slice i0) const {
        t.data = reinterpret_cast<T*>(reinterpret_cast<int8_t*>(t.data) + i0.i0 * strides[isrc]);
        t.shape[idst] = std::min(shape[isrc], i0.i1) - i0.i0;
        t.strides[idst] = strides[isrc];
        t.ndim++;
    }

    // I0 is not range
    template<int isrc, int idst, typename TV, typename ... IDX>
    void get_subview(TV& t, int i0, IDX ... idxs) const {
        // i0 is int, so it will change ptr
        t.data = reinterpret_cast<T*>(reinterpret_cast<int8_t*>(t.data) + i0 * strides[isrc]);
        get_subview<isrc + 1, idst>(t, idxs...);
    }

    template<int isrc, int idst, typename TV>
    void get_subview(TV& t, int i0) const {
        // i0 is int, so it will change ptr
        t.data = reinterpret_cast<T*>(reinterpret_cast<int8_t*>(t.data) + i0 * strides[isrc]);
    }

#ifdef EXPORT_TENSORND_TO_PYBIND11
public:
    static inline void bind2py(pybind11::handle m)
    {
        pybind11::class_<tensorND<T>>(m, "tensorND", pybind11::buffer_protocol())
            .def_buffer([](tensorND<T>& t) -> pybind11::buffer_info {
            return pybind11::buffer_info(reinterpret_cast<void*>(t.data),        /* Pointer to buffer */
                sizeof(T),                          /* Size of one scalar */
                pybind11::format_descriptor<T>::format(), /* Python struct-style format descriptor */
                t.ndim,                                 /* Number of dimensions */
                { t.shape, t.shape + t.ndim },            /* Buffer dimensions */
                { t.strides, t.strides + t.ndim },
                false
            );
                })
            .def(pybind11::init([](std::vector<int64_t> a, bool b) {
                    return new tensorND<T>(a, b);
                }))
                    .def(pybind11::init([](pybind11::buffer b) {
                    // from object support buffer protocol
                    pybind11::buffer_info info = b.request();
                    if (info.format != pybind11::format_descriptor<T>::format())
                        throw std::runtime_error("Incompatible format: expected a float array!");
                    return new tensorND<float>(static_cast<T*>(info.ptr), info.shape, info.strides);
                        }))
                    .def("__repr__", &tensorND<T>::toString);
    }

    static inline tensorND<T, RMAX> from_array(pybind11::array_t<T>& arr) {
        const auto* itshape0 = arr.shape();
        const auto* itstrides0 = arr.strides();
        tensorND<T, RMAX> t(static_cast<T*>(arr.data()),
            itshape0, itshape0 + arr.ndim,
            itstrides0, itstrides0 + arr.ndim);
        return t;
    }

    pybind11::array_t<T> to_array() {
        // steal ownership of the data from t
        auto* new_t = new tensorND<T, RMAX>(std::move(*this));

        // capsule object holding reference to the C++ class
        pybind11::capsule free_when_done(new_t, [](void* f) {
            auto* foo = reinterpret_cast<tensorND<T, RMAX>*>(f);
            delete foo;
            });

        // wraps tensor data as numpy array, referencing C++ class
        // which is responsible for freeing data
        return pybind11::array_t<T>(
            { new_t->shape, new_t->shape + new_t->ndim },     // shape
            { new_t->strides, new_t->strides + new_t->ndim }, // C-style contiguous strides
            new_t->data,        // the data pointer
            free_when_done);    // numpy array references this parent
    }

#endif

};
