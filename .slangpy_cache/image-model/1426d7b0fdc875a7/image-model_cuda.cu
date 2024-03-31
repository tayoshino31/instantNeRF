#define SLANG_PRELUDE_EXPORT

#ifdef __CUDACC_RTC__
#define SLANG_CUDA_RTC 1
#else
#define SLANG_CUDA_RTC 0
#endif

#if SLANG_CUDA_RTC

#else

#include <cstdint>
#include <stdio.h>

#endif

// Define SLANG_CUDA_ENABLE_HALF to use the cuda_fp16 include to add half support. 
// For this to work NVRTC needs to have the path to the CUDA SDK.
//
// As it stands the includes paths defined for Slang are passed down to NVRTC. Similarly defines defined for the Slang compile
// are passed down. 

#ifdef SLANG_CUDA_ENABLE_HALF
// We don't want half2 operators, because it will implement comparison operators that return a bool(!). We want to generate
// those functions. Doing so means that we will have to define all the other half2 operators.
#   define __CUDA_NO_HALF2_OPERATORS__
#   include <cuda_fp16.h>
#endif

#ifdef SLANG_CUDA_ENABLE_OPTIX
#include <optix.h>
#endif

// Define slang offsetof implementation 
#ifndef SLANG_OFFSET_OF
#   define SLANG_OFFSET_OF(type, member) (size_t)((char*)&(((type *)0)->member) - (char*)0)
#endif

#ifndef SLANG_ALIGN_OF
#   define SLANG_ALIGN_OF(type) __alignof__(type)
#endif

// Must be large enough to cause overflow and therefore infinity
#ifndef SLANG_INFINITY
#   define SLANG_INFINITY   ((float)(1e+300 * 1e+300))
#endif

// For now we'll disable any asserts in this prelude
#define SLANG_PRELUDE_ASSERT(x) 

#ifndef SLANG_CUDA_WARP_SIZE 
#   define SLANG_CUDA_WARP_SIZE 32
#endif

#define SLANG_CUDA_WARP_MASK (SLANG_CUDA_WARP_SIZE - 1) // Used for masking threadIdx.x to the warp lane index
#define SLANG_CUDA_WARP_BITMASK (~int(0))

//
#define SLANG_FORCE_INLINE inline

#define SLANG_CUDA_CALL __device__ 

#define SLANG_FORCE_INLINE inline
#define SLANG_INLINE inline


// Since we are using unsigned arithmatic care is need in this comparison.
// It is *assumed* that sizeInBytes >= elemSize. Which means (sizeInBytes >= elemSize) >= 0
// Which means only a single test is needed

// Asserts for bounds checking.
// It is assumed index/count are unsigned types.
#define SLANG_BOUND_ASSERT(index, count)  SLANG_PRELUDE_ASSERT(index < count); 
#define SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes) SLANG_PRELUDE_ASSERT(index <= (sizeInBytes - elemSize) && (index & 3) == 0);

// Macros to zero index if an access is out of range
#define SLANG_BOUND_ZERO_INDEX(index, count) index = (index < count) ? index : 0; 
#define SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes) index = (index <= (sizeInBytes - elemSize)) ? index : 0; 

// The 'FIX' macro define how the index is fixed. The default is to do nothing. If SLANG_ENABLE_BOUND_ZERO_INDEX
// the fix macro will zero the index, if out of range
#ifdef  SLANG_ENABLE_BOUND_ZERO_INDEX
#   define SLANG_BOUND_FIX(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#   define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes) SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#   define SLANG_BOUND_FIX_FIXED_ARRAY(index, count) SLANG_BOUND_ZERO_INDEX(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#else
#   define SLANG_BOUND_FIX(index, count) 
#   define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes) 
#   define SLANG_BOUND_FIX_FIXED_ARRAY(index, count) 
#endif

#ifndef SLANG_BOUND_CHECK
#   define SLANG_BOUND_CHECK(index, count) SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX(index, count)
#endif

#ifndef SLANG_BOUND_CHECK_BYTE_ADDRESS
#   define SLANG_BOUND_CHECK_BYTE_ADDRESS(index, elemSize, sizeInBytes) SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes) SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#endif

#ifndef SLANG_BOUND_CHECK_FIXED_ARRAY
#   define SLANG_BOUND_CHECK_FIXED_ARRAY(index, count) SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

 // This macro handles how out-of-range surface coordinates are handled; 
 // I can equal
 // cudaBoundaryModeClamp, in which case out-of-range coordinates are clamped to the valid range
 // cudaBoundaryModeZero, in which case out-of-range reads return zero and out-of-range writes are ignored
 // cudaBoundaryModeTrap, in which case out-of-range accesses cause the kernel execution to fail. 
 
#ifndef SLANG_CUDA_BOUNDARY_MODE
#   define SLANG_CUDA_BOUNDARY_MODE cudaBoundaryModeZero

// Can be one of SLANG_CUDA_PTX_BOUNDARY_MODE. Only applies *PTX* emitted CUDA operations
// which currently is just RWTextureRW format writes
// 
// .trap         causes an execution trap on out-of-bounds addresses
// .clamp        stores data at the nearest surface location (sized appropriately)
// .zero         drops stores to out-of-bounds addresses 

#   define SLANG_PTX_BOUNDARY_MODE "zero"
#endif

struct TypeInfo
{
    size_t typeSize;
};

template <typename T, size_t SIZE>
struct FixedArray
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const { SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE); return m_data[index]; }
    SLANG_CUDA_CALL T& operator[](size_t index) { SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE); return m_data[index]; }
    
    T m_data[SIZE];
};

// An array that has no specified size, becomes a 'Array'. This stores the size so it can potentially 
// do bounds checking.  
template <typename T>
struct Array
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const { SLANG_BOUND_CHECK(index, count); return data[index]; }
    SLANG_CUDA_CALL T& operator[](size_t index) { SLANG_BOUND_CHECK(index, count); return data[index]; }
    
    T* data;
    size_t count;
};

// Typically defined in cuda.h, but we can't ship/rely on that, so just define here
typedef unsigned long long CUtexObject;                   
typedef unsigned long long CUsurfObject;                  

// On CUDA sampler state is actually bound up with the texture object. We have a SamplerState type, 
// backed as a pointer, to simplify code generation, with the downside that such a binding will take up 
// uniform space, even though it will have no effect. 
// TODO(JS): Consider ways to strip use of variables of this type so have no binding,
struct SamplerStateUnused;
typedef SamplerStateUnused* SamplerState;


// TODO(JS): Not clear yet if this can be handled on CUDA, by just ignoring.
// For now, just map to the index type. 
typedef size_t NonUniformResourceIndex;

// Code generator will generate the specific type
template <typename T, int ROWS, int COLS>
struct Matrix;

typedef int1 bool1;
typedef int2 bool2;
typedef int3 bool3;
typedef int4 bool4; 

#if SLANG_CUDA_RTC

typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#endif

typedef long long longlong;
typedef unsigned long long ulonglong;

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

union Union32 
{
    uint32_t u;
    int32_t i;
    float f;
};

union Union64
{
    uint64_t u;
    int64_t i;
    double d;
};

SLANG_FORCE_INLINE SLANG_CUDA_CALL float _slang_fmod(float x, float y)
{
    return ::fmodf(x, y);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double _slang_fmod(double x, double y)
{
    return ::fmod(x, y);
}

#if SLANG_CUDA_ENABLE_HALF

// Add the other vector half types
struct __half1 { __half x; };
struct __align__(4) __half3 { __half x, y, z; };
struct __align__(4) __half4 { __half x, y, z, w; };
#endif

#define SLANG_VECTOR_GET_ELEMENT(T) \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##1 x, int index) { return ((T*)(&x))[index]; }\
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##2 x, int index) { return ((T*)(&x))[index]; }\
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##3 x, int index) { return ((T*)(&x))[index]; }\
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##4 x, int index) { return ((T*)(&x))[index]; }
SLANG_VECTOR_GET_ELEMENT(int)
SLANG_VECTOR_GET_ELEMENT(uint)
SLANG_VECTOR_GET_ELEMENT(short)
SLANG_VECTOR_GET_ELEMENT(ushort)
SLANG_VECTOR_GET_ELEMENT(char)
SLANG_VECTOR_GET_ELEMENT(uchar)
SLANG_VECTOR_GET_ELEMENT(longlong)
SLANG_VECTOR_GET_ELEMENT(ulonglong)
SLANG_VECTOR_GET_ELEMENT(float)
SLANG_VECTOR_GET_ELEMENT(double)

#define SLANG_VECTOR_GET_ELEMENT_PTR(T) \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##1* x, int index) { return ((T*)(x)) + index; }\
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##2* x, int index) { return ((T*)(x)) + index; }\
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##3* x, int index) { return ((T*)(x)) + index; }\
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##4* x, int index) { return ((T*)(x)) + index; }
SLANG_VECTOR_GET_ELEMENT_PTR(int)
SLANG_VECTOR_GET_ELEMENT_PTR(uint)
SLANG_VECTOR_GET_ELEMENT_PTR(short)
SLANG_VECTOR_GET_ELEMENT_PTR(ushort)
SLANG_VECTOR_GET_ELEMENT_PTR(char)
SLANG_VECTOR_GET_ELEMENT_PTR(uchar)
SLANG_VECTOR_GET_ELEMENT_PTR(longlong)
SLANG_VECTOR_GET_ELEMENT_PTR(ulonglong)
SLANG_VECTOR_GET_ELEMENT_PTR(float)
SLANG_VECTOR_GET_ELEMENT_PTR(double)

#if SLANG_CUDA_ENABLE_HALF
SLANG_VECTOR_GET_ELEMENT(__half)
SLANG_VECTOR_GET_ELEMENT_PTR(__half)
#endif

#define SLANG_CUDA_VECTOR_BINARY_OP(T, n, op) \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator op(T##n thisVal, T##n other) \
    { \
        T##n result;\
        for (int i = 0; i < n; i++) \
            *_slang_vector_get_element_ptr(&result, i) = _slang_vector_get_element(thisVal,i) op _slang_vector_get_element(other,i); \
        return result;\
    }
#define SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, op) \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL bool##n operator op(T##n thisVal, T##n other) \
    { \
        bool##n result;\
        for (int i = 0; i < n; i++) \
            *_slang_vector_get_element_ptr(&result, i) = (int)(_slang_vector_get_element(thisVal,i) op _slang_vector_get_element(other,i)); \
        return result;\
    }
#define SLANG_CUDA_VECTOR_UNARY_OP(T, n, op) \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator op(T##n thisVal) \
    { \
        T##n result;\
        for (int i = 0; i < n; i++) \
            *_slang_vector_get_element_ptr(&result, i) = op _slang_vector_get_element(thisVal,i); \
        return result;\
    }

#define SLANG_CUDA_VECTOR_INT_OP(T, n) \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, +)\
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, -)\
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, *)\
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, /)\
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, %)\
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ^)\
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &)\
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, |)\
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &&)\
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ||)\
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, >>)\
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, <<)\
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >)\
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <)\
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >=)\
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <=)\
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, ==)\
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, !=)\
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, !)\
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, -)\
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, ~)

#define SLANG_CUDA_VECTOR_INT_OPS(T) \
    SLANG_CUDA_VECTOR_INT_OP(T, 2) \
    SLANG_CUDA_VECTOR_INT_OP(T, 3) \
    SLANG_CUDA_VECTOR_INT_OP(T, 4)

SLANG_CUDA_VECTOR_INT_OPS(int)
SLANG_CUDA_VECTOR_INT_OPS(uint)
SLANG_CUDA_VECTOR_INT_OPS(ushort)
SLANG_CUDA_VECTOR_INT_OPS(short)
SLANG_CUDA_VECTOR_INT_OPS(char)
SLANG_CUDA_VECTOR_INT_OPS(uchar)
SLANG_CUDA_VECTOR_INT_OPS(longlong)
SLANG_CUDA_VECTOR_INT_OPS(ulonglong)

#define SLANG_CUDA_VECTOR_FLOAT_OP(T, n) \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, +)\
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, -)\
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, *)\
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, /)\
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &&)\
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ||)\
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >)\
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <)\
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >=)\
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <=)\
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, ==)\
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, !=)\
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, -)
#define SLANG_CUDA_VECTOR_FLOAT_OPS(T) \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 2) \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 3) \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 4)

SLANG_CUDA_VECTOR_FLOAT_OPS(float)
SLANG_CUDA_VECTOR_FLOAT_OPS(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_CUDA_VECTOR_FLOAT_OPS(__half)
#endif
#define SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, n)\
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator%(const T##n& left, const T##n& right) \
    {\
        T##n result;\
        for (int i = 0; i < n; i++) \
            *_slang_vector_get_element_ptr(&result, i) = _slang_fmod(_slang_vector_get_element(left,i), _slang_vector_get_element(right,i)); \
        return result;\
    }
#define SLANG_CUDA_FLOAT_VECTOR_MOD(T) \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 2)\
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 3)\
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 4)

SLANG_CUDA_FLOAT_VECTOR_MOD(float)
SLANG_CUDA_FLOAT_VECTOR_MOD(double)

#if SLANG_CUDA_RTC
#define SLANG_MAKE_VECTOR(T) \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x, T y) { return T##2{x, y}; }\
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x, T y, T z) { return T##3{ x, y, z }; }\
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x, T y, T z, T w) { return T##4{ x, y, z, w }; }
SLANG_MAKE_VECTOR(int)
SLANG_MAKE_VECTOR(uint)
SLANG_MAKE_VECTOR(short)
SLANG_MAKE_VECTOR(ushort)
SLANG_MAKE_VECTOR(char)
SLANG_MAKE_VECTOR(uchar)
SLANG_MAKE_VECTOR(float)
SLANG_MAKE_VECTOR(double)
SLANG_MAKE_VECTOR(longlong)
SLANG_MAKE_VECTOR(ulonglong)
#endif

#if SLANG_CUDA_ENABLE_HALF
SLANG_MAKE_VECTOR(__half)
#endif

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool1 make_bool1(bool x) { return bool1{ x }; }
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool2 make_bool2(bool x, bool y) { return bool2{ x, y }; }
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool3 make_bool3(bool x, bool y, bool z) { return bool3{ x, y, z }; }
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool4 make_bool4(bool x, bool y, bool z, bool w) { return bool4{ x, y, z, w }; }

#if SLANG_CUDA_RTC
#define SLANG_MAKE_VECTOR_FROM_SCALAR(T) \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##1 make_##T##1(T x) { return T##1{x}; }\
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x) { return make_##T##2(x, x); }\
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x) { return make_##T##3(x, x, x); }\
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x) { return make_##T##4(x, x, x, x); }
#else
#define SLANG_MAKE_VECTOR_FROM_SCALAR(T) \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x) { return make_##T##2(x, x); }\
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x) { return make_##T##3(x, x, x); }\
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x) { return make_##T##4(x, x, x, x); }
#endif
SLANG_MAKE_VECTOR_FROM_SCALAR(int)
SLANG_MAKE_VECTOR_FROM_SCALAR(uint)
SLANG_MAKE_VECTOR_FROM_SCALAR(short)
SLANG_MAKE_VECTOR_FROM_SCALAR(ushort)
SLANG_MAKE_VECTOR_FROM_SCALAR(char)
SLANG_MAKE_VECTOR_FROM_SCALAR(uchar)
SLANG_MAKE_VECTOR_FROM_SCALAR(longlong)
SLANG_MAKE_VECTOR_FROM_SCALAR(ulonglong)
SLANG_MAKE_VECTOR_FROM_SCALAR(float)
SLANG_MAKE_VECTOR_FROM_SCALAR(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_MAKE_VECTOR_FROM_SCALAR(__half)
#endif

#define SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(Fn,T,N) \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##N Fn(T##N* address, T##N val) \
    {\
        T##N result; \
        for (int i = 0; i < N; i++) \
            *_slang_vector_get_element_ptr(&result, i) = Fn(_slang_vector_get_element_ptr(address, i), _slang_vector_get_element(val, i)); \
        return result; \
    }\

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 4)
#endif
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 4)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 4)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 4)

template<typename T, int n>
struct GetVectorTypeImpl {};

#define GET_VECTOR_TYPE_IMPL(T, n)\
template<>\
struct GetVectorTypeImpl<T,n>\
{\
    typedef T##n type;\
    static SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n fromScalar(T v) { return make_##T##n(v); } \
};
#define GET_VECTOR_TYPE_IMPL_N(T)\
    GET_VECTOR_TYPE_IMPL(T, 1)\
    GET_VECTOR_TYPE_IMPL(T, 2)\
    GET_VECTOR_TYPE_IMPL(T, 3)\
    GET_VECTOR_TYPE_IMPL(T, 4)

GET_VECTOR_TYPE_IMPL_N(int)
GET_VECTOR_TYPE_IMPL_N(uint)
GET_VECTOR_TYPE_IMPL_N(short)
GET_VECTOR_TYPE_IMPL_N(ushort)
GET_VECTOR_TYPE_IMPL_N(char)
GET_VECTOR_TYPE_IMPL_N(uchar)
GET_VECTOR_TYPE_IMPL_N(longlong)
GET_VECTOR_TYPE_IMPL_N(ulonglong)
GET_VECTOR_TYPE_IMPL_N(float)
GET_VECTOR_TYPE_IMPL_N(double)
#if SLANG_CUDA_ENABLE_HALF
GET_VECTOR_TYPE_IMPL_N(__half)
#endif
template<typename T, int n>
using Vector = typename GetVectorTypeImpl<T, n>::type;

template<typename T, int n, typename OtherT, int m>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, n> _slang_vector_reshape(const Vector<OtherT, m> other)
{
    Vector<T, n> result;
    for (int i = 0; i < n; i++)
    {
        OtherT otherElement = T(0);
        if (i < m)
            otherElement = _slang_vector_get_element(other, i);
        *_slang_vector_get_element_ptr(&result, i) = (T)otherElement;
    }
    return result;
}

template <typename T, int ROWS, int COLS>
struct Matrix
{
    Vector<T, COLS> rows[ROWS];
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, COLS>& operator[](size_t index) { return rows[index]; }
};


template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(T scalar)
{
    Matrix<T, ROWS, COLS> result;
    for (int i = 0; i < ROWS; i++)
        result.rows[i] = GetVectorTypeImpl<T, COLS>::fromScalar(scalar);
    return result;

}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(const Vector<T, COLS>& row0)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(const Vector<T, COLS>& row0, const Vector<T, COLS>& row1)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(const Vector<T, COLS>& row0, const Vector<T, COLS>& row1, const Vector<T, COLS>& row2)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    result.rows[2] = row2;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(const Vector<T, COLS>& row0, const Vector<T, COLS>& row1, const Vector<T, COLS>& row2, const Vector<T, COLS>& row3)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    result.rows[2] = row2;
    result.rows[3] = row3;
    return result;
}

template<typename T, int ROWS, int COLS, typename U, int otherRow, int otherCol>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(const Matrix<U, otherRow, otherCol>& other)
{
    Matrix<T, ROWS, COLS> result;
    int minRow = ROWS;
    int minCol = COLS;
    if (minRow > otherRow) minRow = otherRow;
    if (minCol > otherCol) minCol = otherCol;
    for (int i = 0; i < minRow; i++)
        for (int j = 0; j < minCol; j++)
            *_slang_vector_get_element_ptr(result.rows + i, j) = (T)_slang_vector_get_element(other.rows[i], j);
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(T v0, T v1, T v2, T v3)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;  rs.rows[0].y = v1;
    rs.rows[1].x = v2;  rs.rows[1].y = v3;
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(T v0, T v1, T v2, T v3, T v4, T v5)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 3)
    {
        rs.rows[0].x = v0;  rs.rows[0].y = v1; rs.rows[0].z = v2;
        rs.rows[1].x = v3;  rs.rows[1].y = v4; rs.rows[1].z = v5;
    }
    else
    {
        rs.rows[0].x = v0;  rs.rows[0].y = v1;
        rs.rows[1].x = v2;  rs.rows[1].y = v3;
        rs.rows[2].x = v4;  rs.rows[2].y = v5;
    }
    return rs;

}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 4)
    {
        rs.rows[0].x = v0;  rs.rows[0].y = v1; rs.rows[0].z = v2; rs.rows[0].w = v3;
        rs.rows[1].x = v4;  rs.rows[1].y = v5; rs.rows[1].z = v6; rs.rows[1].w = v7;
    }
    else
    {
        rs.rows[0].x = v0;  rs.rows[0].y = v1;
        rs.rows[1].x = v2;  rs.rows[1].y = v3;
        rs.rows[2].x = v4;  rs.rows[2].y = v5;
        rs.rows[3].x = v6;  rs.rows[3].y = v7;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;  rs.rows[0].y = v1;  rs.rows[0].z = v2;
    rs.rows[1].x = v3;  rs.rows[1].y = v4;  rs.rows[1].z = v5;
    rs.rows[2].x = v6;  rs.rows[2].y = v7;  rs.rows[2].z = v8;
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 4)
    {
        rs.rows[0].x = v0;  rs.rows[0].y = v1;  rs.rows[0].z = v2;  rs.rows[0].w = v3;
        rs.rows[1].x = v4;  rs.rows[1].y = v5;  rs.rows[1].z = v6;  rs.rows[1].w = v7;
        rs.rows[2].x = v8;  rs.rows[2].y = v9;  rs.rows[2].z = v10; rs.rows[2].w = v11;
    }
    else
    {
        rs.rows[0].x = v0;  rs.rows[0].y = v1;  rs.rows[0].z = v2;
        rs.rows[1].x = v3;  rs.rows[1].y = v4;  rs.rows[1].z = v5;
        rs.rows[2].x = v6;  rs.rows[2].y = v7;  rs.rows[2].z = v8;
        rs.rows[3].x = v9;  rs.rows[3].y = v10; rs.rows[3].z = v11;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11, T v12, T v13, T v14, T v15)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;  rs.rows[0].y = v1;  rs.rows[0].z = v2;  rs.rows[0].w = v3;
    rs.rows[1].x = v4;  rs.rows[1].y = v5;  rs.rows[1].z = v6;  rs.rows[1].w = v7;
    rs.rows[2].x = v8;  rs.rows[2].y = v9;  rs.rows[2].z = v10; rs.rows[2].w = v11;
    rs.rows[3].x = v12; rs.rows[3].y = v13; rs.rows[3].z = v14; rs.rows[3].w = v15;
    return rs;
}

#define SLANG_MATRIX_BINARY_OP(T, op) \
    template<int R, int C> \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator op(const Matrix<T, R, C>& thisVal, const Matrix<T, R, C>& other) \
    { \
        Matrix<T, R, C> result;\
        for (int i = 0; i < R; i++) \
            for (int j = 0; j < C; j++) \
                *_slang_vector_get_element_ptr(result.rows+i,j) = _slang_vector_get_element(thisVal.rows[i], j) op _slang_vector_get_element(other.rows[i], j); \
        return result;\
    }

#define SLANG_MATRIX_UNARY_OP(T, op) \
    template<int R, int C> \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator op(const Matrix<T, R, C>& thisVal) \
    { \
        Matrix<T, R, C> result;\
        for (int i = 0; i < R; i++) \
            for (int j = 0; j < C; j++) \
                *_slang_vector_get_element_ptr(result.rows+i,j) = op _slang_vector_get_element(thisVal.rows[i], j); \
        return result;\
    }
#define SLANG_INT_MATRIX_OPS(T) \
    SLANG_MATRIX_BINARY_OP(T, +)\
    SLANG_MATRIX_BINARY_OP(T, -)\
    SLANG_MATRIX_BINARY_OP(T, *)\
    SLANG_MATRIX_BINARY_OP(T, / )\
    SLANG_MATRIX_BINARY_OP(T, &)\
    SLANG_MATRIX_BINARY_OP(T, |)\
    SLANG_MATRIX_BINARY_OP(T, &&)\
    SLANG_MATRIX_BINARY_OP(T, ||)\
    SLANG_MATRIX_BINARY_OP(T, ^)\
    SLANG_MATRIX_BINARY_OP(T, %)\
    SLANG_MATRIX_UNARY_OP(T, !)\
    SLANG_MATRIX_UNARY_OP(T, ~)
#define SLANG_FLOAT_MATRIX_OPS(T) \
    SLANG_MATRIX_BINARY_OP(T, +)\
    SLANG_MATRIX_BINARY_OP(T, -)\
    SLANG_MATRIX_BINARY_OP(T, *)\
    SLANG_MATRIX_BINARY_OP(T, /)\
    SLANG_MATRIX_UNARY_OP(T, -)
SLANG_INT_MATRIX_OPS(int)
SLANG_INT_MATRIX_OPS(uint)
SLANG_INT_MATRIX_OPS(short)
SLANG_INT_MATRIX_OPS(ushort)
SLANG_INT_MATRIX_OPS(char)
SLANG_INT_MATRIX_OPS(uchar)
SLANG_INT_MATRIX_OPS(longlong)
SLANG_INT_MATRIX_OPS(ulonglong)
SLANG_FLOAT_MATRIX_OPS(float)
SLANG_FLOAT_MATRIX_OPS(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_FLOAT_MATRIX_OPS(__half)
#endif
#define SLANG_MATRIX_INT_NEG_OP(T) \
    template<int R, int C>\
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator-(Matrix<T, R, C> thisVal) \
    { \
        Matrix<T, R, C> result;\
        for (int i = 0; i < R; i++) \
            for (int j = 0; j < C; j++) \
                *_slang_vector_get_element_ptr(result.rows+i,j) = 0 - _slang_vector_get_element(thisVal.rows[i], j); \
        return result;\
    }
    SLANG_MATRIX_INT_NEG_OP(int)
    SLANG_MATRIX_INT_NEG_OP(uint)
    SLANG_MATRIX_INT_NEG_OP(short)
    SLANG_MATRIX_INT_NEG_OP(ushort)
    SLANG_MATRIX_INT_NEG_OP(char)
    SLANG_MATRIX_INT_NEG_OP(uchar)
    SLANG_MATRIX_INT_NEG_OP(longlong)
    SLANG_MATRIX_INT_NEG_OP(ulonglong)

#define SLANG_FLOAT_MATRIX_MOD(T)\
    template<int R, int C> \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator%(Matrix<T, R, C> left, Matrix<T, R, C> right) \
    {\
        Matrix<T, R, C> result;\
        for (int i = 0; i < R; i++) \
            for (int j = 0; j < C; j++) \
                *_slang_vector_get_element_ptr(result.rows+i,j) = _slang_fmod(_slang_vector_get_element(left.rows[i], j), _slang_vector_get_element(right.rows[i], j)); \
        return result;\
    }

    SLANG_FLOAT_MATRIX_MOD(float)
    SLANG_FLOAT_MATRIX_MOD(double)
#if SLANG_CUDA_ENABLE_HALF
    template<int R, int C> 
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<__half, R, C> operator%(Matrix<__half, R, C> left, Matrix<__half, R, C> right)
    {
        Matrix<__half, R, C> result;
        for (int i = 0; i < R; i++) 
            for (int j = 0; j < C; j++) 
                * _slang_vector_get_element_ptr(result.rows + i, j) = __float2half(_slang_fmod(__half2float(_slang_vector_get_element(left.rows[i], j)), __half2float(_slang_vector_get_element(right.rows[i], j))));
        return result;
    }
#endif
#undef SLANG_FLOAT_MATRIX_MOD
#undef SLANG_MATRIX_BINARY_OP
#undef SLANG_MATRIX_UNARY_OP
#undef SLANG_INT_MATRIX_OPS
#undef SLANG_FLOAT_MATRIX_OPS
#undef SLANG_MATRIX_INT_NEG_OP
#undef SLANG_FLOAT_MATRIX_MOD

#define SLANG_SELECT_IMPL(T, N)\
SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, N> _slang_select(bool##N condition, Vector<T, N> v0, Vector<T, N> v1) \
{ \
    Vector<T, N> result; \
    for (int i = 0; i < N; i++) \
    { \
        *_slang_vector_get_element_ptr(&result, i) = _slang_vector_get_element(condition, i) ? _slang_vector_get_element(v0, i) : _slang_vector_get_element(v1, i); \
    } \
    return result; \
}
#define SLANG_SELECT_T(T)\
    SLANG_SELECT_IMPL(T, 2)\
    SLANG_SELECT_IMPL(T, 3)\
    SLANG_SELECT_IMPL(T, 4)

SLANG_SELECT_T(int)
SLANG_SELECT_T(uint)
SLANG_SELECT_T(short)
SLANG_SELECT_T(ushort)
SLANG_SELECT_T(char)
SLANG_SELECT_T(uchar)
SLANG_SELECT_T(float)
SLANG_SELECT_T(double)

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_select(bool condition, T v0, T v1)
{
    return condition ? v0 : v1;
}

//
// Half support
// 

#if SLANG_CUDA_ENABLE_HALF
SLANG_SELECT_T(__half)

// Convenience functions ushort -> half

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half2 __ushort_as_half(const ushort2& i) { return __halves2half2(__ushort_as_half(i.x), __ushort_as_half(i.y)); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half3 __ushort_as_half(const ushort3& i) { return __half3{__ushort_as_half(i.x), __ushort_as_half(i.y), __ushort_as_half(i.z)}; }
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half4 __ushort_as_half(const ushort4& i) { return __half4{ __ushort_as_half(i.x), __ushort_as_half(i.y), __ushort_as_half(i.z), __ushort_as_half(i.w) }; }

// Convenience functions half -> ushort

SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort2 __half_as_ushort(const __half2& i) { return make_ushort2(__half_as_ushort(i.x), __half_as_ushort(i.y)); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort3 __half_as_ushort(const __half3& i) { return make_ushort3(__half_as_ushort(i.x), __half_as_ushort(i.y), __half_as_ushort(i.z)); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort4 __half_as_ushort(const __half4& i) { return make_ushort4(__half_as_ushort(i.x), __half_as_ushort(i.y), __half_as_ushort(i.z), __half_as_ushort(i.w)); }

// This is a little bit of a hack. Fortunately CUDA has the definitions of the templated types in 
// include/surface_indirect_functions.h
// Here we find the template definition requires a specialization of __nv_isurf_trait to allow 
// a specialization of the surface write functions. 
// This *isn't* a problem on the read functions as they don't have a return type that uses this mechanism 

template<> struct __nv_isurf_trait<__half> { typedef void type; };
template<> struct __nv_isurf_trait<__half2> { typedef void type; };
template<> struct __nv_isurf_trait<__half4> { typedef void type; };

#define SLANG_DROP_PARENS(...) __VA_ARGS__

#define SLANG_SURFACE_READ(FUNC_NAME, TYPE_ARGS, ARGS) \
template <> \
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half FUNC_NAME<__half>(cudaSurfaceObject_t surfObj, SLANG_DROP_PARENS TYPE_ARGS, cudaSurfaceBoundaryMode boundaryMode) \
{ \
    return __ushort_as_half(FUNC_NAME<ushort>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
} \
\
template <> \
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half2 FUNC_NAME<__half2>(cudaSurfaceObject_t surfObj, SLANG_DROP_PARENS TYPE_ARGS, cudaSurfaceBoundaryMode boundaryMode) \
{ \
    return __ushort_as_half(FUNC_NAME<ushort2>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
} \
\
template <> \
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half4 FUNC_NAME<__half4>(cudaSurfaceObject_t surfObj, SLANG_DROP_PARENS TYPE_ARGS, cudaSurfaceBoundaryMode boundaryMode) \
{ \
    return __ushort_as_half(FUNC_NAME<ushort4>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
}

SLANG_SURFACE_READ(surf1Dread, (int x), (x))
SLANG_SURFACE_READ(surf2Dread, (int x, int y), (x, y))
SLANG_SURFACE_READ(surf3Dread, (int x, int y, int z), (x, y, z))
SLANG_SURFACE_READ(surf1DLayeredread, (int x, int layer), (x, layer))
SLANG_SURFACE_READ(surf2DLayeredread, (int x, int y, int layer), (x, y, layer))
SLANG_SURFACE_READ(surfCubemapread, (int x, int y, int face), (x, y, face))
SLANG_SURFACE_READ(surfCubemapLayeredread, (int x, int y, int layerFace), (x, y, layerFace))

#define SLANG_SURFACE_WRITE(FUNC_NAME, TYPE_ARGS, ARGS) \
template <> \
SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half>(__half data, cudaSurfaceObject_t surfObj, SLANG_DROP_PARENS TYPE_ARGS, cudaSurfaceBoundaryMode boundaryMode) \
{ \
    FUNC_NAME<ushort>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode);  \
} \
\
template <> \
SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half2>(__half2 data, cudaSurfaceObject_t surfObj, SLANG_DROP_PARENS TYPE_ARGS, cudaSurfaceBoundaryMode boundaryMode) \
{ \
    FUNC_NAME<ushort2>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode);  \
} \
\
template <> \
SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half4>(__half4 data, cudaSurfaceObject_t surfObj, SLANG_DROP_PARENS TYPE_ARGS, cudaSurfaceBoundaryMode boundaryMode) \
{ \
    FUNC_NAME<ushort4>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode); \
}

SLANG_SURFACE_WRITE(surf1Dwrite, (int x), (x))
SLANG_SURFACE_WRITE(surf2Dwrite, (int x, int y), (x, y))
SLANG_SURFACE_WRITE(surf3Dwrite, (int x, int y, int z), (x, y, z))
SLANG_SURFACE_WRITE(surf1DLayeredwrite, (int x, int layer), (x, layer))
SLANG_SURFACE_WRITE(surf2DLayeredwrite, (int x, int y, int layer), (x, y, layer))
SLANG_SURFACE_WRITE(surfCubemapwrite, (int x, int y, int face), (x, y, face))
SLANG_SURFACE_WRITE(surfCubemapLayeredwrite, (int x, int y, int layerFace), (x, y, layerFace))

// ! Hack to test out reading !!!
// Only works converting *from* half 
 
//template <typename T> 
//SLANG_FORCE_INLINE SLANG_CUDA_CALL T surf2Dread_convert(cudaSurfaceObject_t surfObj, int x, int y, cudaSurfaceBoundaryMode boundaryMode);

#define SLANG_SURFACE_READ_HALF_CONVERT(FUNC_NAME, TYPE_ARGS, ARGS) \
\
template <typename T>  \
SLANG_FORCE_INLINE SLANG_CUDA_CALL T FUNC_NAME##_convert(cudaSurfaceObject_t surfObj, SLANG_DROP_PARENS TYPE_ARGS, cudaSurfaceBoundaryMode boundaryMode); \
\
template <> \
SLANG_FORCE_INLINE SLANG_CUDA_CALL float FUNC_NAME##_convert<float>(cudaSurfaceObject_t surfObj, SLANG_DROP_PARENS TYPE_ARGS, cudaSurfaceBoundaryMode boundaryMode)  \
{ \
    return __ushort_as_half(FUNC_NAME<uint16_t>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
} \
\
template <> \
SLANG_FORCE_INLINE SLANG_CUDA_CALL float2 FUNC_NAME##_convert<float2>(cudaSurfaceObject_t surfObj, SLANG_DROP_PARENS TYPE_ARGS, cudaSurfaceBoundaryMode boundaryMode) \
{ \
    const __half2 v = __ushort_as_half(FUNC_NAME<ushort2>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
    return float2{v.x, v.y}; \
} \
\
template <> \
SLANG_FORCE_INLINE SLANG_CUDA_CALL float4 FUNC_NAME##_convert<float4>(cudaSurfaceObject_t surfObj, SLANG_DROP_PARENS TYPE_ARGS, cudaSurfaceBoundaryMode boundaryMode) \
{ \
    const __half4 v = __ushort_as_half(FUNC_NAME<ushort4>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
    return float4{v.x, v.y, v.z, v.w}; \
}

SLANG_SURFACE_READ_HALF_CONVERT(surf1Dread, (int x), (x)) 
SLANG_SURFACE_READ_HALF_CONVERT(surf2Dread, (int x, int y), (x, y)) 
SLANG_SURFACE_READ_HALF_CONVERT(surf3Dread, (int x, int y, int z), (x, y, z))

#endif

// Support for doing format conversion when writing to a surface/RWTexture

// NOTE! For normal surface access x values are *byte* addressed.
// For the _convert versions they are *not*. They don't need to be because sust.p does not require it.

template <typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert(T, cudaSurfaceObject_t surfObj, int x, cudaSurfaceBoundaryMode boundaryMode);
template <typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert(T, cudaSurfaceObject_t surfObj, int x, int y, cudaSurfaceBoundaryMode boundaryMode);
template <typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert(T, cudaSurfaceObject_t surfObj, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode);

// https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-sust

// Float

template <>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<float>(float v, cudaSurfaceObject_t surfObj, int x, cudaSurfaceBoundaryMode boundaryMode)
{
    asm volatile ( "{sust.p.1d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1}], {%2};}\n\t" :: "l"(surfObj),"r"(x),"f"(v));     
}
 
template <>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<float>(float v, cudaSurfaceObject_t surfObj, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
    asm volatile ( "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2}], {%3};}\n\t" :: "l"(surfObj),"r"(x),"r"(y),"f"(v));
}

template <>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<float>(float v, cudaSurfaceObject_t surfObj, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
    asm volatile ( "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2,%3}], {%4};}\n\t" :: "l"(surfObj),"r"(x),"r"(y),"r"(z),"f"(v));
}

// Float2

template <>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<float2>(float2 v, cudaSurfaceObject_t surfObj, int x, cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y;
    asm volatile ( "{sust.p.1d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1}], {%2,%3};}\n\t" :: "l"(surfObj),"r"(x),"f"(vx),"f"(vy));     
}
 
template <>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<float2>(float2 v, cudaSurfaceObject_t surfObj, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y;
    asm volatile ( "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2}], {%3,%4};}\n\t" :: "l"(surfObj),"r"(x),"r"(y),"f"(vx),"f"(vy));
}

template <>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<float2>(float2 v, cudaSurfaceObject_t surfObj, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y;
    asm volatile ( "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2,%3}], {%4,%5};}\n\t" :: "l"(surfObj),"r"(x),"r"(y),"r"(z),"f"(vx),"f"(vy));
}

// Float4
template <>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<float4>(float4 v, cudaSurfaceObject_t surfObj, int x, cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y, vz = v.z, vw = v.w;
    asm volatile ( "{sust.p.1d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1}], {%2,%3,%4,%5};}\n\t" :: "l"(surfObj),"r"(x),"f"(vx),"f"(vy),"f"(vz),"f"(vw));     
}
 
template <>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<float4>(float4 v, cudaSurfaceObject_t surfObj, int x, int y, cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y, vz = v.z, vw = v.w;
    asm volatile ( "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2}], {%3,%4,%5,%6};}\n\t" :: "l"(surfObj),"r"(x),"r"(y),"f"(vx),"f"(vy),"f"(vz),"f"(vw));
}

template <>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<float4>(float4 v, cudaSurfaceObject_t surfObj, int x, int y, int z, cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y, vz = v.z, vw = v.w;
    asm volatile ( "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2,%3}], {%4,%5,%6,%7};}\n\t" :: "l"(surfObj),"r"(x),"r"(y),"r"(z),"f"(vx),"f"(vy),"f"(vz),"f"(vw));
}

// ----------------------------- F32 -----------------------------------------

// Unary 
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_ceil(float f) { return ::ceilf(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_floor(float f) { return ::floorf(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_round(float f) { return ::roundf(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sin(float f) { return ::sinf(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_cos(float f) { return ::cosf(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL void F32_sincos(float f, float* s, float* c) { ::sincosf(f, s, c); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_tan(float f) { return ::tanf(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_asin(float f) { return ::asinf(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_acos(float f) { return ::acosf(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_atan(float f) { return ::atanf(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sinh(float f) { return ::sinhf(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_cosh(float f) { return ::coshf(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_tanh(float f) { return ::tanhf(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log2(float f) { return ::log2f(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log(float f) { return ::logf(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log10(float f) { return ::log10f(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_exp2(float f) { return ::exp2f(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_exp(float f) { return ::expf(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_abs(float f) { return ::fabsf(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_trunc(float f) { return ::truncf(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sqrt(float f) { return ::sqrtf(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_rsqrt(float f) { return ::rsqrtf(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sign(float f) { return ( f == 0.0f) ? f : (( f < 0.0f) ? -1.0f : 1.0f); } 
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_frac(float f) { return f - F32_floor(f); }

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isnan(float f) { return isnan(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isfinite(float f) { return isfinite(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isinf(float f) { return isinf(f); }

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_min(float a, float b) { return ::fminf(a, b); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_max(float a, float b) { return ::fmaxf(a, b); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_pow(float a, float b) { return ::powf(a, b); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_fmod(float a, float b) { return ::fmodf(a, b); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_remainder(float a, float b) { return ::remainderf(a, b); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_atan2(float a, float b) { return float(::atan2(a, b)); }

SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_frexp(float x, int* e) { return frexpf(x, e); }

SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_modf(float x, float* ip)
{
    return ::modff(x, ip);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t F32_asuint(float f) { Union32 u; u.f = f; return u.u; }
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t F32_asint(float f) { Union32 u; u.f = f; return u.i; }

// Ternary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_fma(float a, float b, float c) { return ::fmaf(a, b, c); }


// ----------------------------- F64 -----------------------------------------

// Unary 
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_ceil(double f) { return ::ceil(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_floor(double f) { return ::floor(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_round(double f) { return ::round(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sin(double f) { return ::sin(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_cos(double f) { return ::cos(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_sincos(double f, double* s, double* c) { ::sincos(f, s, c); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_tan(double f) { return ::tan(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_asin(double f) { return ::asin(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_acos(double f) { return ::acos(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_atan(double f) { return ::atan(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sinh(double f) { return ::sinh(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_cosh(double f) { return ::cosh(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_tanh(double f) { return ::tanh(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log2(double f) { return ::log2(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log(double f) { return ::log(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log10(float f) { return ::log10(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_exp2(double f) { return ::exp2(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_exp(double f) { return ::exp(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_abs(double f) { return ::fabs(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_trunc(double f) { return ::trunc(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sqrt(double f) { return ::sqrt(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_rsqrt(double f) { return ::rsqrt(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sign(double f) { return (f == 0.0) ? f : ((f < 0.0) ? -1.0 : 1.0); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_frac(double f) { return f - F64_floor(f); }

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isnan(double f) { return isnan(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isfinite(double f) { return isfinite(f); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isinf(double f) { return isinf(f); }

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_min(double a, double b) { return ::fmin(a, b); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_max(double a, double b) { return ::fmax(a, b); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_pow(double a, double b) { return ::pow(a, b); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_fmod(double a, double b) { return ::fmod(a, b); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_remainder(double a, double b) { return ::remainder(a, b); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_atan2(double a, double b) { return ::atan2(a, b); }

SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_frexp(double x, int* e) { return ::frexp(x, e); }

SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_modf(double x, double* ip)
{
    return ::modf(x, ip);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_asuint(double d, uint32_t* low, uint32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = uint32_t(u.u);
    *hi = uint32_t(u.u >> 32);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_asint(double d, int32_t* low, int32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = int32_t(u.u);
    *hi = int32_t(u.u >> 32);
}

// Ternary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_fma(double a, double b, double c) { return ::fma(a, b, c); }

// ----------------------------- I32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_abs(int32_t f) { return (f < 0) ? -f : f; }

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_min(int32_t a, int32_t b) { return a < b ? a : b; }
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_max(int32_t a, int32_t b) { return a > b ? a : b; }

SLANG_FORCE_INLINE SLANG_CUDA_CALL float I32_asfloat(int32_t x) { Union32 u; u.i = x; return u.f; }
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I32_asuint(int32_t x) { return uint32_t(x); }
SLANG_FORCE_INLINE SLANG_CUDA_CALL double I32_asdouble(int32_t low, int32_t hi )
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | uint32_t(low);
    return u.d;
}

// ----------------------------- U32 -----------------------------------------

// Unary 
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_abs(uint32_t f) { return f; }

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_min(uint32_t a, uint32_t b) { return a < b ? a : b; }
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_max(uint32_t a, uint32_t b) { return a > b ? a : b; }

SLANG_FORCE_INLINE SLANG_CUDA_CALL float U32_asfloat(uint32_t x) { Union32 u; u.u = x; return u.f; }
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_asint(int32_t x) { return uint32_t(x); }

SLANG_FORCE_INLINE SLANG_CUDA_CALL double U32_asdouble(uint32_t low, uint32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | low;
    return u.d;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_countbits(uint32_t v)
{
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html#group__CUDA__MATH__INTRINSIC__INT_1g43c9c7d2b9ebf202ff1ef5769989be46
    return __popc(v);
}


// ----------------------------- I64 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_abs(int64_t f) { return (f < 0) ? -f : f; }

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_min(int64_t a, int64_t b) { return a < b ? a : b; }
SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_max(int64_t a, int64_t b) { return a > b ? a : b; }

// ----------------------------- U64 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_abs(uint64_t f) { return f; }

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_min(uint64_t a, uint64_t b) { return a < b ? a : b; }
SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_max(uint64_t a, uint64_t b) { return a > b ? a : b; }

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U64_countbits(uint64_t v)
{
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html#group__CUDA__MATH__INTRINSIC__INT_1g43c9c7d2b9ebf202ff1ef5769989be46
    return __popcll(v);
}


// ----------------------------- ResourceType -----------------------------------------


// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-structuredbuffer-getdimensions
// Missing  Load(_In_  int  Location, _Out_ uint Status);

template <typename T>
struct StructuredBuffer
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, count);
#endif
        return data[index];
    }

    SLANG_CUDA_CALL const T& Load(size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, count);
#endif
        return data[index];
    }

#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outNumStructs, uint32_t* outStride) { *outNumStructs = uint32_t(count); *outStride = uint32_t(sizeof(T)); }
#endif

    T* data;
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
    size_t count;
#endif
};

template <typename T>
struct RWStructuredBuffer : StructuredBuffer<T>
{
    SLANG_CUDA_CALL T& operator[](size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, this->count);
#endif
        return this->data[index];
    }
};

// Missing  Load(_In_  int  Location, _Out_ uint Status);
struct ByteAddressBuffer
{
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }
    SLANG_CUDA_CALL uint32_t Load(size_t index) const 
    { 
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2]; 
    }
    SLANG_CUDA_CALL uint2 Load2(size_t index) const 
    { 
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes); 
        const size_t dataIdx = index >> 2; 
        return uint2{data[dataIdx], data[dataIdx + 1]}; 
    }
    SLANG_CUDA_CALL uint3 Load3(size_t index) const 
    { 
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2; 
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]}; 
    }
    SLANG_CUDA_CALL uint4 Load4(size_t index) const 
    { 
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2; 
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]}; 
    }
    template<typename T>
    SLANG_CUDA_CALL T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        T data;
        memcpy(&data, ((const char*)this->data) + index, sizeof(T));
        return data;
    }
    
    const uint32_t* data;
    size_t sizeInBytes;  //< Must be multiple of 4
};

// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-rwbyteaddressbuffer
// Missing support for Atomic operations 
// Missing support for Load with status
struct RWByteAddressBuffer
{
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }
    
    SLANG_CUDA_CALL uint32_t Load(size_t index) const 
    { 
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2]; 
    }
    SLANG_CUDA_CALL uint2 Load2(size_t index) const 
    { 
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2; 
        return uint2{data[dataIdx], data[dataIdx + 1]}; 
    }
    SLANG_CUDA_CALL uint3 Load3(size_t index) const 
    { 
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2; 
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]}; 
    }
    SLANG_CUDA_CALL uint4 Load4(size_t index) const 
    { 
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2; 
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]}; 
    }
    template<typename T>
    SLANG_CUDA_CALL T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        T data;
        memcpy(&data, ((const char*)this->data) + index, sizeof(T));
        return data;
    }
    
    SLANG_CUDA_CALL void Store(size_t index, uint32_t v) const 
    { 
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        data[index >> 2] = v; 
    }
    SLANG_CUDA_CALL void Store2(size_t index, uint2 v) const 
    { 
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2; 
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
    }
    SLANG_CUDA_CALL void Store3(size_t index, uint3 v) const 
    { 
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2; 
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
    }
    SLANG_CUDA_CALL void Store4(size_t index, uint4 v) const 
    { 
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2; 
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
        data[dataIdx + 3] = v.w;
    }
    template<typename T>
    SLANG_CUDA_CALL void Store(size_t index, T const& value) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        memcpy((char*)data + index, &value, sizeof(T));
    }
    
        /// Can be used in stdlib to gain access
    template <typename T>
    SLANG_CUDA_CALL T* _getPtrAt(size_t index)
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        return (T*)(((char*)data) + index);
    }
    
    uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4 
};


// ---------------------- Wave --------------------------------------

// TODO(JS): It appears that cuda does not have a simple way to get a lane index. 
// 
// Another approach could be... 
// laneId = ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) & SLANG_CUDA_WARP_MASK
// If that is really true another way to do this, would be for code generator to add this function 
// with the [numthreads] baked in. 
// 
// For now I'll just assume you have a launch that makes the following correct if the kernel uses WaveGetLaneIndex()
#ifndef SLANG_USE_ASM_LANE_ID
 __forceinline__ __device__ uint32_t _getLaneId()
{
    // If the launch is (or I guess some multiple of the warp size) 
    // we try this mechanism, which is apparently faster. 
    return threadIdx.x & SLANG_CUDA_WARP_MASK;
}
#else
__forceinline__ __device__ uint32_t _getLaneId()
{
    // https://stackoverflow.com/questions/44337309/whats-the-most-efficient-way-to-calculate-the-warp-id-lane-id-in-a-1-d-grid#
    // This mechanism is not the fastest way to do it, and that is why the other mechanism 
    // is the default. But the other mechanism relies on a launch that makes the assumption 
    // true.
    unsigned ret; 
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}
#endif

typedef int WarpMask;

// It appears that the __activemask() cannot always be used because 
// threads need to be converged. 
// 
// For CUDA the article claims mask has to be used carefully
// https://devblogs.nvidia.com/using-cuda-warp-level-primitives/
// With the Warp intrinsics there is no mask, and it's just the 'active lanes'. 
// __activemask() though does not require there is convergence, so that doesn't work.
// 
// '__ballot_sync' produces a convergance. 
// 
// From the CUDA docs:
// ```For __all_sync, __any_sync, and __ballot_sync, a mask must be passed that specifies the threads 
// participating in the call. A bit, representing the thread's lane ID, must be set for each participating thread 
// to ensure they are properly converged before the intrinsic is executed by the hardware. All active threads named 
// in mask must execute the same intrinsic with the same mask, or the result is undefined.```
//
// Currently there isn't a mechanism to correctly get the mask without it being passed through.
// Doing so will most likely require some changes to slang code generation to track masks, for now then we use
// _getActiveMask. 

// Return mask of all the lanes less than the current lane
__forceinline__ __device__ WarpMask _getLaneLtMask()
{
    return (int(1) << _getLaneId()) - 1;
}    

// TODO(JS): 
// THIS IS NOT CORRECT! That determining the appropriate active mask requires appropriate
// mask tracking.
__forceinline__ __device__ WarpMask _getActiveMask()
{
    return __ballot_sync(__activemask(), true);
}

// Return a mask suitable for the 'MultiPrefix' style functions
__forceinline__ __device__ WarpMask _getMultiPrefixMask(int mask)
{
    return mask;
}

// Note! Note will return true if mask is 0, but thats okay, because there must be one
// lane active to execute anything
__inline__ __device__ bool _waveIsSingleLane(WarpMask mask)
{
    return (mask & (mask - 1)) == 0;
}

// Returns the power of 2 size of run of set bits. Returns 0 if not a suitable run.
// Examples:
// 0b00000000'00000000'00000000'11111111 -> 8
// 0b11111111'11111111'11111111'11111111 -> 32
// 0b00000000'00000000'00000000'00011111 -> 0 (since 5 is not a power of 2)
// 0b00000000'00000000'00000000'11110000 -> 0 (since the run of bits does not start at the LSB)
// 0b00000000'00000000'00000000'00100111 -> 0 (since it is not a single contiguous run)
__inline__ __device__ int _waveCalcPow2Offset(WarpMask mask)
{
    // This should be the most common case, so fast path it
    if (mask == SLANG_CUDA_WARP_BITMASK)
    {
        return SLANG_CUDA_WARP_SIZE;
    }
    // Is it a contiguous run of bits?
    if ((mask & (mask + 1)) == 0)
    {
        // const int offsetSize = __ffs(mask + 1) - 1;
        const int offset = 32 - __clz(mask);
        // Is it a power of 2 size
        if ((offset & (offset - 1)) == 0)
        {
            return offset;
        }
    }
    return 0;
}

__inline__ __device__ bool _waveIsFirstLane()
{
    const WarpMask mask = __activemask();
    // We special case bit 0, as that most warps are expected to be fully active. 
    
    // mask & -mask, isolates the lowest set bit.
    //return (mask & 1 ) || ((mask & -mask) == (1 << _getLaneId()));
    
    // This mechanism is most similar to what was in an nVidia post, so assume it is prefered. 
    return (mask & 1 ) || ((__ffs(mask) - 1) == _getLaneId());
}

template <typename T>
struct WaveOpOr
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a | b; }
};

template <typename T>
struct WaveOpAnd
{
    __inline__ __device__ static T getInitial(T a) { return ~T(0); }
    __inline__ __device__ static T doOp(T a, T b) { return a & b; }
};

template <typename T>
struct WaveOpXor
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a ^ b; }
    __inline__ __device__ static T doInverse(T a, T b) { return a ^ b; }
};

template <typename T>
struct WaveOpAdd
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a + b; }
    __inline__ __device__ static T doInverse(T a, T b) { return a - b; }
};

template <typename T>
struct WaveOpMul
{
    __inline__ __device__ static T getInitial(T a) { return T(1); }
    __inline__ __device__ static T doOp(T a, T b) { return a * b; }
    // Using this inverse for int is probably undesirable - because in general it requires T to have more precision
    // There is also a performance aspect to it, where divides are generally significantly slower
    __inline__ __device__ static T doInverse(T a, T b) { return a / b; }
};

template <typename T>
struct WaveOpMax
{
    __inline__ __device__ static T getInitial(T a) { return a; }
    __inline__ __device__ static T doOp(T a, T b) { return a > b ? a : b; }
};

template <typename T>
struct WaveOpMin
{
    __inline__  __device__ static T getInitial(T a) { return a; }
    __inline__ __device__ static T doOp(T a, T b) { return a < b ? a : b; }
};

template <typename T>
struct ElementTypeTrait;

// Scalar
template <> struct ElementTypeTrait<int> { typedef int Type; };
template <> struct ElementTypeTrait<uint> { typedef uint Type; };
template <> struct ElementTypeTrait<float> { typedef float Type; };
template <> struct ElementTypeTrait<double> { typedef double Type; };
template <> struct ElementTypeTrait<uint64_t> { typedef uint64_t Type; };
template <> struct ElementTypeTrait<int64_t> { typedef int64_t Type; };

// Vector
template <> struct ElementTypeTrait<int1> { typedef int Type; };
template <> struct ElementTypeTrait<int2> { typedef int Type; };
template <> struct ElementTypeTrait<int3> { typedef int Type; };
template <> struct ElementTypeTrait<int4> { typedef int Type; };

template <> struct ElementTypeTrait<uint1> { typedef uint Type; };
template <> struct ElementTypeTrait<uint2> { typedef uint Type; };
template <> struct ElementTypeTrait<uint3> { typedef uint Type; };
template <> struct ElementTypeTrait<uint4> { typedef uint Type; };

template <> struct ElementTypeTrait<float1> { typedef float Type; };
template <> struct ElementTypeTrait<float2> { typedef float Type; };
template <> struct ElementTypeTrait<float3> { typedef float Type; };
template <> struct ElementTypeTrait<float4> { typedef float Type; };

template <> struct ElementTypeTrait<double1> { typedef double Type; };
template <> struct ElementTypeTrait<double2> { typedef double Type; };
template <> struct ElementTypeTrait<double3> { typedef double Type; };
template <> struct ElementTypeTrait<double4> { typedef double Type; };

// Matrix
template <typename T, int ROWS, int COLS> 
struct ElementTypeTrait<Matrix<T, ROWS, COLS> >  
{ 
    typedef T Type; 
};

// Scalar 
template <typename INTF, typename T>
__device__ T _waveReduceScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);
    if (offsetSize > 0)
    {
        // Fast path O(log2(activeLanes)) 
        for (int offset = offsetSize >> 1; offset > 0; offset >>= 1)
        {
            val = INTF::doOp(val, __shfl_xor_sync(mask, val, offset));
        }
    }
    else if (!_waveIsSingleLane(mask))
    {
        T result = INTF::getInitial(val);
        int remaining = mask;
        while (remaining)
        {
            const int laneBit = remaining & -remaining;
            // Get the sourceLane 
            const int srcLane = __ffs(laneBit) - 1;
            // Broadcast (can also broadcast to self) 
            result = INTF::doOp(result, __shfl_sync(mask, val, srcLane));
            remaining &= ~laneBit;
        }
        return result;
    }
    return val;
}


// Multiple values
template <typename INTF, typename T, size_t COUNT>
__device__ void _waveReduceMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);
    if (offsetSize > 0)
    {
        // Fast path O(log2(activeLanes)) 
        for (int offset = offsetSize >> 1; offset > 0; offset >>= 1)
        {
            for (size_t i = 0; i < COUNT; ++i)
            {
                val[i] = INTF::doOp(val[i], __shfl_xor_sync(mask, val[i], offset));
            }
        }
    }
    else if (!_waveIsSingleLane(mask))
    {
        // Copy the original
        T originalVal[COUNT];
        for (size_t i = 0; i < COUNT; ++i)
        {
            const T v = val[i];
            originalVal[i] = v;
            val[i] = INTF::getInitial(v);
        }
        
        int remaining = mask;
        while (remaining)
        {
            const int laneBit = remaining & -remaining;
            // Get the sourceLane 
            const int srcLane = __ffs(laneBit) - 1;
            // Broadcast (can also broadcast to self) 
            for (size_t i = 0; i < COUNT; ++i)
            {
                val[i] = INTF::doOp(val[i], __shfl_sync(mask, originalVal[i], srcLane));
            }
            remaining &= ~laneBit;
        }
    }
}

template <typename INTF, typename T>
__device__ void _waveReduceMultiple(WarpMask mask, T* val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;    
    _waveReduceMultiple<INTF, ElemType, sizeof(T) / sizeof(ElemType)>(mask, (ElemType*)val);
}

template <typename T>
__inline__ __device__  T _waveOr(WarpMask mask, T val) { return _waveReduceScalar<WaveOpOr<T>, T>(mask, val); }

template <typename T>
__inline__ __device__ T _waveAnd(WarpMask mask, T val) { return _waveReduceScalar<WaveOpAnd<T>, T>(mask, val); }

template <typename T>
__inline__ __device__ T _waveXor(WarpMask mask, T val) { return _waveReduceScalar<WaveOpXor<T>, T>(mask, val); }

template <typename T>
__inline__ __device__ T _waveProduct(WarpMask mask, T val) { return _waveReduceScalar<WaveOpMul<T>, T>(mask, val); }

template <typename T>
__inline__ __device__ T _waveSum(WarpMask mask, T val) { return _waveReduceScalar<WaveOpAdd<T>, T>(mask, val); }

template <typename T>
__inline__ __device__ T _waveMin(WarpMask mask, T val) { return _waveReduceScalar<WaveOpMin<T>, T>(mask, val); }

template <typename T>
__inline__ __device__ T _waveMax(WarpMask mask, T val) { return _waveReduceScalar<WaveOpMax<T>, T>(mask, val); }

// Fast-path specializations when CUDA warp reduce operators are available
#if __CUDA_ARCH__ >= 800 // 8.x or higher
template<>
__inline__ __device__ unsigned _waveOr<unsigned>(WarpMask mask, unsigned val) { return __reduce_or_sync(mask, val); }

template<>
__inline__ __device__ unsigned _waveAnd<unsigned>(WarpMask mask, unsigned val) { return __reduce_and_sync(mask, val); }

template<>
__inline__ __device__ unsigned _waveXor<unsigned>(WarpMask mask, unsigned val) { return __reduce_xor_sync(mask, val); }

template<>
__inline__ __device__ unsigned _waveSum<unsigned>(WarpMask mask, unsigned val) { return __reduce_add_sync(mask, val); }

template<>
__inline__ __device__ int _waveSum<int>(WarpMask mask, int val) { return __reduce_add_sync(mask, val); }

template<>
__inline__ __device__ unsigned _waveMin<unsigned>(WarpMask mask, unsigned val) { return __reduce_min_sync(mask, val); }

template<>
__inline__ __device__ int _waveMin<int>(WarpMask mask, int val) { return __reduce_min_sync(mask, val); }

template<>
__inline__ __device__ unsigned _waveMax<unsigned>(WarpMask mask, unsigned val) { return __reduce_max_sync(mask, val); }

template<>
__inline__ __device__ int _waveMax<int>(WarpMask mask, int val) { return __reduce_max_sync(mask, val); }
#endif


// Multiple

template <typename T>
__inline__ __device__  T _waveOrMultiple(WarpMask mask, T val) { typedef typename ElementTypeTrait<T>::Type ElemType; _waveReduceMultiple<WaveOpOr<ElemType> >(mask, &val); return val; }

template <typename T>
__inline__ __device__  T _waveAndMultiple(WarpMask mask, T val) { typedef typename ElementTypeTrait<T>::Type ElemType; _waveReduceMultiple<WaveOpAnd<ElemType> >(mask, &val); return val; }

template <typename T>
__inline__ __device__  T _waveXorMultiple(WarpMask mask, T val) { typedef typename ElementTypeTrait<T>::Type ElemType; _waveReduceMultiple<WaveOpXor<ElemType> >(mask, &val); return val; }

template <typename T>
__inline__ __device__  T _waveProductMultiple(WarpMask mask, T val) { typedef typename ElementTypeTrait<T>::Type ElemType; _waveReduceMultiple<WaveOpMul<ElemType> >(mask, &val); return val; }

template <typename T>
__inline__ __device__  T _waveSumMultiple(WarpMask mask, T val) { typedef typename ElementTypeTrait<T>::Type ElemType; _waveReduceMultiple<WaveOpAdd<ElemType> >(mask, &val); return val; }

template <typename T>
__inline__ __device__  T _waveMinMultiple(WarpMask mask, T val) { typedef typename ElementTypeTrait<T>::Type ElemType; _waveReduceMultiple<WaveOpMin<ElemType> >(mask, &val); return val; }

template <typename T>
__inline__ __device__  T _waveMaxMultiple(WarpMask mask, T val) { typedef typename ElementTypeTrait<T>::Type ElemType; _waveReduceMultiple<WaveOpMax<ElemType> >(mask, &val); return val; }


template <typename T>
__inline__ __device__ bool _waveAllEqual(WarpMask mask, T val) 
{
    int pred;
    __match_all_sync(mask, val, &pred);
    return pred != 0;
}

template <typename T>
__inline__ __device__ bool _waveAllEqualMultiple(WarpMask mask, T inVal) 
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    int pred;
    const ElemType* src = (const ElemType*)&inVal;
    for (size_t i = 0; i < count; ++i)
    {
        __match_all_sync(mask, src[i], &pred);
        if (pred == 0)
        {
            return false;
        }
    }
    return true;
}

template <typename T>
__inline__ __device__ T _waveReadFirst(WarpMask mask, T val) 
{
    const int lowestLaneId = __ffs(mask) - 1;
    return __shfl_sync(mask, val, lowestLaneId);   
}

template <typename T>
__inline__ __device__ T _waveReadFirstMultiple(WarpMask mask, T inVal) 
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    T outVal;
    const ElemType* src = (const ElemType*)&inVal;
    ElemType* dst = (ElemType*)&outVal;
    const int lowestLaneId = __ffs(mask) - 1;
    for (size_t i = 0; i < count; ++i)
    {
        dst[i] = __shfl_sync(mask, src[i], lowestLaneId);   
    }
    return outVal;
}

template <typename T>
__inline__ __device__ T _waveShuffleMultiple(WarpMask mask, T inVal, int lane)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    T outVal;
    const ElemType* src = (const ElemType*)&inVal;
    ElemType* dst = (ElemType*)&outVal;
    for (size_t i = 0; i < count; ++i)
    {
        dst[i] = __shfl_sync(mask, src[i], lane);   
    }
    return outVal;
}

// Scalar 

// Invertable means that when we get to the end of the reduce, we can remove val (to make exclusive), using 
// the inverse of the op.
template <typename INTF, typename T>
__device__ T _wavePrefixInvertableScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);
    
    const int laneId = _getLaneId();
    T result;
    if (offsetSize > 0)
    {    
        // Sum is calculated inclusive of this lanes value
        result = val;
        for (int i = 1; i < offsetSize; i += i) 
        {
            const T readVal = __shfl_up_sync(mask, result, i, offsetSize);
            if (laneId >= i)
            {
                result = INTF::doOp(result, readVal);
            }
        }
        // Remove val from the result, by applyin inverse
        result = INTF::doInverse(result, val);
    }
    else 
    {
        result = INTF::getInitial(val);
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane 
                const int srcLane = __ffs(laneBit) - 1;
                // Broadcast (can also broadcast to self) 
                const T readValue = __shfl_sync(mask, val, srcLane);
                // Only accumulate if srcLane is less than this lane
                if (srcLane < laneId)
                {
                    result = INTF::doOp(result, readValue);
                }
                remaining &= ~laneBit;
            }
        }   
    }
    return result;
}
 

// This implementation separately tracks the value to be propogated, and the value
// that is the final result 
template <typename INTF, typename T>
__device__ T _wavePrefixScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);
    
    const int laneId = _getLaneId();
    T result = INTF::getInitial(val);           
    if (offsetSize > 0)
    {    
        // For transmitted value we will do it inclusively with this lanes value
        // For the result we do not include the lanes value. This means an extra multiply for each iteration
        // but means we don't need to have a divide at the end and also removes overflow issues in that scenario.
        for (int i = 1; i < offsetSize; i += i) 
        {
            const T readVal = __shfl_up_sync(mask, val, i, offsetSize);
            if (laneId >= i)
            {
                result = INTF::doOp(result, readVal);
                val = INTF::doOp(val, readVal);
            }
        }
    }
    else 
    {
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane 
                const int srcLane = __ffs(laneBit) - 1;
                // Broadcast (can also broadcast to self) 
                const T readValue = __shfl_sync(mask, val, srcLane);
                // Only accumulate if srcLane is less than this lane
                if (srcLane < laneId)
                {
                    result = INTF::doOp(result, readValue);
                }
                remaining &= ~laneBit;
            }
        }
    }
    return result;
}


template <typename INTF, typename T, size_t COUNT>
__device__ T _waveOpCopy(T* dst, const T* src)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        dst[j] = src[j];
    }
}    


template <typename INTF, typename T, size_t COUNT>
__device__ T _waveOpDoInverse(T* inOut, const T* val)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        inOut[j] = INTF::doInverse(inOut[j], val[j]);
    }
}    

template <typename INTF, typename T, size_t COUNT>
__device__ T _waveOpSetInitial(T* out, const T* val)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        out[j] = INTF::getInitial(val[j]);
    }
} 

template <typename INTF, typename T, size_t COUNT>
__device__ T _wavePrefixInvertableMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);
    
    const int laneId = _getLaneId();
    T originalVal[COUNT];
    _waveOpCopy<INTF, T, COUNT>(originalVal, val);
    
    if (offsetSize > 0)
    {    
        // Sum is calculated inclusive of this lanes value
        for (int i = 1; i < offsetSize; i += i) 
        {
            // TODO(JS): Note that here I don't split the laneId outside so it's only tested once.
            // This may be better but it would also mean that there would be shfl between lanes 
            // that are on different (albeit identical) instructions. So this seems more likely to 
            // work as expected with everything in lock step.
            for (size_t j = 0; j < COUNT; ++j)
            {
                const T readVal = __shfl_up_sync(mask, val[j], i, offsetSize);
                if (laneId >= i)
                {
                    val[j] = INTF::doOp(val[j], readVal);
                }
            }
        }
        // Remove originalVal from the result, by applyin inverse
        _waveOpDoInverse<INTF, T, COUNT>(val, originalVal);
    }
    else 
    {
        _waveOpSetInitial<INTF, T, COUNT>(val, val);
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane 
                const int srcLane = __ffs(laneBit) - 1;
                
                for (size_t j = 0; j < COUNT; ++j)
                {
                    // Broadcast (can also broadcast to self) 
                    const T readValue = __shfl_sync(mask, originalVal[j], srcLane);
                    // Only accumulate if srcLane is less than this lane
                    if (srcLane < laneId)
                    {
                        val[j] = INTF::doOp(val[j], readValue);
                    }
                    remaining &= ~laneBit;
                }
            }
        }   
    }
}
 
template <typename INTF, typename T, size_t COUNT>
__device__ T _wavePrefixMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);
    
    const int laneId = _getLaneId();
    
    T work[COUNT];
    _waveOpCopy<INTF, T, COUNT>(work, val);
    _waveOpSetInitial<INTF, T, COUNT>(val, val);
    
    if (offsetSize > 0)
    {    
        // For transmitted value we will do it inclusively with this lanes value
        // For the result we do not include the lanes value. This means an extra op for each iteration
        // but means we don't need to have a divide at the end and also removes overflow issues in that scenario.
        for (int i = 1; i < offsetSize; i += i) 
        {
            for (size_t j = 0; j < COUNT; ++j)
            {
                const T readVal = __shfl_up_sync(mask, work[j], i, offsetSize);
                if (laneId >= i)
                {
                    work[j] = INTF::doOp(work[j], readVal);
                    val[j] = INTF::doOp(val[j], readVal);     
                }
            }
        }
    }
    else 
    {
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane 
                const int srcLane = __ffs(laneBit) - 1;
                
                for (size_t j = 0; j < COUNT; ++j)
                {
                    // Broadcast (can also broadcast to self) 
                    const T readValue = __shfl_sync(mask, work[j], srcLane);
                    // Only accumulate if srcLane is less than this lane
                    if (srcLane < laneId)
                    {
                        val[j] = INTF::doOp(val[j], readValue);
                    }
                }
                remaining &= ~laneBit;
            }
        }
    }
}

template <typename T>
__inline__ __device__ T _wavePrefixProduct(WarpMask mask, T val) { return _wavePrefixScalar<WaveOpMul<T>, T>(mask, val); }

template <typename T>
__inline__ __device__ T _wavePrefixSum(WarpMask mask, T val) { return _wavePrefixInvertableScalar<WaveOpAdd<T>, T>(mask, val); }    

template <typename T>
__inline__ __device__ T _wavePrefixXor(WarpMask mask, T val) { return _wavePrefixInvertableScalar<WaveOpXor<T>, T>(mask, val); }    
    
template <typename T>
__inline__ __device__ T _wavePrefixOr(WarpMask mask, T val) { return _wavePrefixScalar<WaveOpOr<T>, T>(mask, val); }      
    
template <typename T>
__inline__ __device__ T _wavePrefixAnd(WarpMask mask, T val) { return _wavePrefixScalar<WaveOpAnd<T>, T>(mask, val); }      
    
    
template <typename T>
__inline__ __device__ T _wavePrefixProductMultiple(WarpMask mask, T val)  
{ 
    typedef typename ElementTypeTrait<T>::Type ElemType;    
    _wavePrefixInvertableMultiple<WaveOpMul<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(mask, (ElemType*)&val);    
    return val;
}

template <typename T>
__inline__ __device__ T _wavePrefixSumMultiple(WarpMask mask, T val) 
{ 
    typedef typename ElementTypeTrait<T>::Type ElemType;    
    _wavePrefixInvertableMultiple<WaveOpAdd<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(mask, (ElemType*)&val);    
    return val;
}

template <typename T>
__inline__ __device__ T _wavePrefixXorMultiple(WarpMask mask, T val)  
{ 
    typedef typename ElementTypeTrait<T>::Type ElemType;    
    _wavePrefixInvertableMultiple<WaveOpXor<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(mask, (ElemType*)&val);    
    return val;
}

template <typename T>
__inline__ __device__ T _wavePrefixOrMultiple(WarpMask mask, T val) 
{ 
    typedef typename ElementTypeTrait<T>::Type ElemType;    
    _wavePrefixMultiple<WaveOpOr<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(mask, (ElemType*)&val);    
    return val;
}

template <typename T>
__inline__ __device__ T _wavePrefixAndMultiple(WarpMask mask, T val)  
{ 
    typedef typename ElementTypeTrait<T>::Type ElemType;    
    _wavePrefixMultiple<WaveOpAnd<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(mask, (ElemType*)&val);    
    return val;
}

template <typename T>
__inline__ __device__ uint4 _waveMatchScalar(WarpMask mask, T val) 
{
    int pred;
    return make_uint4(__match_all_sync(mask, val, &pred), 0, 0, 0);
}

template <typename T>
__inline__ __device__ uint4 _waveMatchMultiple(WarpMask mask, const T& inVal) 
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    int pred;
    const ElemType* src = (const ElemType*)&inVal;
    uint matchBits = 0xffffffff;
    for (size_t i = 0; i < count && matchBits; ++i)
    {
        matchBits = matchBits & __match_all_sync(mask, src[i], &pred);
    }
    return make_uint4(matchBits, 0, 0, 0);
}

__device__ uint getAt(dim3 a,  int b)
{
    SLANG_PRELUDE_ASSERT(b >= 0 && b < 3);
    return (&a.x)[b];
}
__device__ uint3 operator*(uint3 a, dim3 b)
{
    uint3 r;
    r.x = a.x * b.x;
    r.y = a.y * b.y;
    r.z = a.z * b.z;
    return r;
}

template<typename TResult, typename TInput>
__inline__ __device__ TResult slang_bit_cast(TInput val)
{
    return *(TResult*)(&val);
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */


/* Type that defines the uniform entry point params. The actual content of this type is dependent on the entry point parameters, and can be
found via reflection or defined such that it matches the shader appropriately.
*/
struct UniformEntryPointParams;
struct UniformState;

// ---------------------- OptiX Ray Payload --------------------------------------
#ifdef SLANG_CUDA_ENABLE_OPTIX
struct RayDesc
{
    float3 Origin;
    float  TMin;
    float3 Direction;
    float  TMax;
};

static __forceinline__ __device__
void *unpackOptiXRayPayloadPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__
void  packOptiXRayPayloadPointer(void* ptr, uint32_t& i0, uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ void *getOptiXRayPayloadPtr()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return unpackOptiXRayPayloadPointer(u0, u1);
}

template<typename T>
__forceinline__ __device__ void *traceOptiXRay(
    OptixTraversableHandle AccelerationStructure,
    uint32_t RayFlags,
    uint32_t InstanceInclusionMask,
    uint32_t RayContributionToHitGroupIndex,
    uint32_t MultiplierForGeometryContributionToHitGroupIndex,
    uint32_t MissShaderIndex,
    RayDesc Ray,
    T *Payload
) {
    uint32_t r0, r1;
    packOptiXRayPayloadPointer((void*)Payload, r0, r1);
    optixTrace(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        Ray.TMax,
        0.f, /* Time for motion blur, currently unsupported in slang */
        InstanceInclusionMask,
        RayFlags,
        RayContributionToHitGroupIndex,
        MultiplierForGeometryContributionToHitGroupIndex,
        MissShaderIndex,
        r0, r1
    );
}

#endif

static const int kSlangTorchTensorMaxDim = 5;

// TensorView
struct TensorView
{
    uint8_t* data;
    uint32_t strides[kSlangTorchTensorMaxDim];
    uint32_t sizes[kSlangTorchTensorMaxDim];
    uint32_t dimensionCount;

    template<typename T>
    __device__ T* data_ptr()
    {
        return reinterpret_cast<T*>(data);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint32_t index)
    {
        uint64_t offset = strides[0] * index;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint2 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint3 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y + strides[2] * index.z;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint4 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y + strides[2] * index.z + strides[3] * index.w;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T, unsigned int N>
    __device__ T* data_ptr_at(uint index[N])
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T& load(uint32_t x)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y);
    }
    template<typename T>
    __device__ T& load(uint2 index)
    {
        return *reinterpret_cast<T*>(data + strides[0] * index.x + strides[1] * index.y);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y, uint32_t z)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y + strides[2] * z);
    }
    template<typename T>
    __device__ T& load(uint3 index)
    {
        return *reinterpret_cast<T*>(data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y, uint32_t z, uint32_t w)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y + strides[2] * z + strides[3] * w);
    }
    template<typename T>
    __device__ T& load(uint4 index)
    {
        return *reinterpret_cast<T*>(data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z + strides[3] * index.w);
    }
    template<typename T>
    __device__ T& load(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4)
    {
        return *reinterpret_cast<T*>(data + strides[0] * i0 + strides[1] * i1 + strides[2] * i2 + strides[3] * i3 + strides[4] * i4);
    }

    // Generic version of load
    template<typename T, unsigned int N>
    __device__ T& load(uint index[N])
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        return *reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ void store(uint32_t x, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y) = val;
    }
    template<typename T>
    __device__ void store(uint2 index, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * index.x + strides[1] * index.y) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, uint32_t z, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y + strides[2] * z) = val;
    }
    template<typename T>
    __device__ void store(uint3 index, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, uint32_t z, uint32_t w, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * x + strides[1] * y + strides[2] * z + strides[3] * w) = val;
    }
    template<typename T>
    __device__ void store(uint4 index, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z + strides[3] * index.w) = val;
    }
    template<typename T>
    __device__ void store(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * i0 + strides[1] * i1 + strides[2] * i2 + strides[3] * i3 + strides[4] * i4) = val;
    }

    // Generic version
    template<typename T, unsigned int N>
    __device__ void store(uint index[N], T val)
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        *reinterpret_cast<T*>(data + offset) = val;
    }
};

#include "cuda_matmul_prelude.cuh"
__device__ __shared__ FixedArray<float, 4096>  shared_output_buffer_0;

__device__ __shared__ FixedArray<float, 2048>  shared_weights_buffer_0;

__device__ __shared__ FixedArray<float, 4096>  shared_inputs_buffer_0;

struct AtomicAdd_0
{
    TensorView diff_0;
};

__device__ float AtomicAdd_load_forward_0(AtomicAdd_0 this_0, uint4  i_0)
{
    float _S1 = ((this_0.diff_0).load<float>((i_0)));
    return _S1;
}

__device__ void AtomicAdd_load_backward_0(AtomicAdd_0 this_1, uint4  i_1, float dOut_0)
{
    float oldVal_0;
    *((&oldVal_0)) = atomicAdd((this_1.diff_0).data_ptr_at<float>((i_1)), (dOut_0));
    return;
}

__device__ void AtomicAdd_storeOnce_forward_0(AtomicAdd_0 this_2, uint4  i_2, float dx_0)
{
    (this_2.diff_0).store<float>((i_2), (dx_0));
    return;
}

__device__ float AtomicAdd_storeOnce_backward_0(AtomicAdd_0 this_3, uint4  i_3)
{
    float _S2 = ((this_3.diff_0).load<float>((i_3)));
    return _S2;
}

struct s_diff_Feature_0
{
    FixedArray<float, 16>  vals_0;
};

__device__ s_diff_Feature_0 Feature_x24_syn_dadd_0(s_diff_Feature_0 SLANG_anonymous_0_0, s_diff_Feature_0 SLANG_anonymous_1_0)
{
    s_diff_Feature_0 result_0;
    float _S3 = SLANG_anonymous_0_0.vals_0[int(1)] + SLANG_anonymous_1_0.vals_0[int(1)];
    float _S4 = SLANG_anonymous_0_0.vals_0[int(2)] + SLANG_anonymous_1_0.vals_0[int(2)];
    float _S5 = SLANG_anonymous_0_0.vals_0[int(3)] + SLANG_anonymous_1_0.vals_0[int(3)];
    float _S6 = SLANG_anonymous_0_0.vals_0[int(4)] + SLANG_anonymous_1_0.vals_0[int(4)];
    float _S7 = SLANG_anonymous_0_0.vals_0[int(5)] + SLANG_anonymous_1_0.vals_0[int(5)];
    float _S8 = SLANG_anonymous_0_0.vals_0[int(6)] + SLANG_anonymous_1_0.vals_0[int(6)];
    float _S9 = SLANG_anonymous_0_0.vals_0[int(7)] + SLANG_anonymous_1_0.vals_0[int(7)];
    float _S10 = SLANG_anonymous_0_0.vals_0[int(8)] + SLANG_anonymous_1_0.vals_0[int(8)];
    float _S11 = SLANG_anonymous_0_0.vals_0[int(9)] + SLANG_anonymous_1_0.vals_0[int(9)];
    float _S12 = SLANG_anonymous_0_0.vals_0[int(10)] + SLANG_anonymous_1_0.vals_0[int(10)];
    float _S13 = SLANG_anonymous_0_0.vals_0[int(11)] + SLANG_anonymous_1_0.vals_0[int(11)];
    float _S14 = SLANG_anonymous_0_0.vals_0[int(12)] + SLANG_anonymous_1_0.vals_0[int(12)];
    float _S15 = SLANG_anonymous_0_0.vals_0[int(13)] + SLANG_anonymous_1_0.vals_0[int(13)];
    float _S16 = SLANG_anonymous_0_0.vals_0[int(14)] + SLANG_anonymous_1_0.vals_0[int(14)];
    float _S17 = SLANG_anonymous_0_0.vals_0[int(15)] + SLANG_anonymous_1_0.vals_0[int(15)];
    *(&(&result_0)->vals_0[int(0)]) = SLANG_anonymous_0_0.vals_0[int(0)] + SLANG_anonymous_1_0.vals_0[int(0)];
    *(&(&result_0)->vals_0[int(1)]) = _S3;
    *(&(&result_0)->vals_0[int(2)]) = _S4;
    *(&(&result_0)->vals_0[int(3)]) = _S5;
    *(&(&result_0)->vals_0[int(4)]) = _S6;
    *(&(&result_0)->vals_0[int(5)]) = _S7;
    *(&(&result_0)->vals_0[int(6)]) = _S8;
    *(&(&result_0)->vals_0[int(7)]) = _S9;
    *(&(&result_0)->vals_0[int(8)]) = _S10;
    *(&(&result_0)->vals_0[int(9)]) = _S11;
    *(&(&result_0)->vals_0[int(10)]) = _S12;
    *(&(&result_0)->vals_0[int(11)]) = _S13;
    *(&(&result_0)->vals_0[int(12)]) = _S14;
    *(&(&result_0)->vals_0[int(13)]) = _S15;
    *(&(&result_0)->vals_0[int(14)]) = _S16;
    *(&(&result_0)->vals_0[int(15)]) = _S17;
    return result_0;
}

__device__ s_diff_Feature_0 Feature_x24_syn_dzero_0()
{
    s_diff_Feature_0 result_1;
    *(&(&result_1)->vals_0[int(0)]) = 0.0f;
    *(&(&result_1)->vals_0[int(1)]) = 0.0f;
    *(&(&result_1)->vals_0[int(2)]) = 0.0f;
    *(&(&result_1)->vals_0[int(3)]) = 0.0f;
    *(&(&result_1)->vals_0[int(4)]) = 0.0f;
    *(&(&result_1)->vals_0[int(5)]) = 0.0f;
    *(&(&result_1)->vals_0[int(6)]) = 0.0f;
    *(&(&result_1)->vals_0[int(7)]) = 0.0f;
    *(&(&result_1)->vals_0[int(8)]) = 0.0f;
    *(&(&result_1)->vals_0[int(9)]) = 0.0f;
    *(&(&result_1)->vals_0[int(10)]) = 0.0f;
    *(&(&result_1)->vals_0[int(11)]) = 0.0f;
    *(&(&result_1)->vals_0[int(12)]) = 0.0f;
    *(&(&result_1)->vals_0[int(13)]) = 0.0f;
    *(&(&result_1)->vals_0[int(14)]) = 0.0f;
    *(&(&result_1)->vals_0[int(15)]) = 0.0f;
    return result_1;
}

struct DiffTensorView_0
{
    TensorView primal_0;
    AtomicAdd_0 diff_1;
};

__device__ uint DiffTensorView_size_0(DiffTensorView_0 this_4, uint i_4)
{
    uint _S18 = ((this_4.primal_0).sizes[(i_4)]);
    return _S18;
}

struct DiffPair_float_0
{
    float primal_1;
    float differential_0;
};

__device__ DiffPair_float_0 DiffTensorView_load_forward_0(DiffTensorView_0 this_5, uint4  x_0)
{
    float _S19 = ((this_5.primal_0).load<float>((x_0)));
    DiffPair_float_0 _S20 = { _S19, AtomicAdd_load_forward_0(this_5.diff_1, x_0) };
    return _S20;
}

__device__ void DiffTensorView_load_backward_0(DiffTensorView_0 this_6, uint4  x_1, float dOut_1)
{
    AtomicAdd_load_backward_0(this_6.diff_1, x_1, dOut_1);
    return;
}

__device__ float DiffTensorView_load_0(DiffTensorView_0 this_7, uint4  i_5)
{
    float _S21 = ((this_7.primal_0).load<float>((i_5)));
    return _S21;
}

__device__ float DiffTensorView_load_1(DiffTensorView_0 this_8, uint2  i_6)
{
    float _S22 = ((this_8.primal_0).load<float>((i_6)));
    return _S22;
}

struct Feature_0
{
    FixedArray<float, 16>  vals_1;
};

__device__ Feature_0 getInFeature_0(DiffTensorView_0 input_0, uint3  idx_0)
{
    Feature_0 output_0;
    uint _S23 = idx_0.x;
    uint _S24 = idx_0.y;
    uint _S25 = idx_0.z;
    uint4  _S26 = make_uint4 (_S23, _S24, _S25, 15U);
    int i_7 = int(0);
    for(;;)
    {
        if(i_7 < int(15))
        {
        }
        else
        {
            break;
        }
        *(&(&output_0)->vals_1[i_7]) = DiffTensorView_load_0(input_0, make_uint4 (_S23, _S24, _S25, uint(i_7)));
        i_7 = i_7 + int(1);
    }
    *(&(&output_0)->vals_1[15U]) = DiffTensorView_load_0(input_0, _S26);
    return output_0;
}

struct Linear_0
{
    DiffTensorView_0 weights_0;
    DiffTensorView_0 bias_0;
};

__device__ void Linear_loadArray_0(Linear_0 this_9, float * memptr_0, FixedArray<float, 16>  * input_1)
{
    int _S27 = int(uint(int(((threadIdx)).x) % int(32))) * int(16);
    *(&(*input_1)[int(0)]) = *(&memptr_0[_S27]);
    *(&(*input_1)[int(1)]) = *(&memptr_0[_S27 + int(1)]);
    *(&(*input_1)[int(2)]) = *(&memptr_0[_S27 + int(2)]);
    *(&(*input_1)[int(3)]) = *(&memptr_0[_S27 + int(3)]);
    *(&(*input_1)[int(4)]) = *(&memptr_0[_S27 + int(4)]);
    *(&(*input_1)[int(5)]) = *(&memptr_0[_S27 + int(5)]);
    *(&(*input_1)[int(6)]) = *(&memptr_0[_S27 + int(6)]);
    *(&(*input_1)[int(7)]) = *(&memptr_0[_S27 + int(7)]);
    *(&(*input_1)[int(8)]) = *(&memptr_0[_S27 + int(8)]);
    *(&(*input_1)[int(9)]) = *(&memptr_0[_S27 + int(9)]);
    *(&(*input_1)[int(10)]) = *(&memptr_0[_S27 + int(10)]);
    *(&(*input_1)[int(11)]) = *(&memptr_0[_S27 + int(11)]);
    *(&(*input_1)[int(12)]) = *(&memptr_0[_S27 + int(12)]);
    *(&(*input_1)[int(13)]) = *(&memptr_0[_S27 + int(13)]);
    *(&(*input_1)[int(14)]) = *(&memptr_0[_S27 + int(14)]);
    *(&(*input_1)[int(15)]) = *(&memptr_0[_S27 + int(15)]);
    return;
}

__device__ uint Linear_calcOffset_0(Linear_0 this_10)
{
    uint3  threadIdx_0 = ((threadIdx));
    return uint((int(threadIdx_0.x) / int(32) + int(threadIdx_0.y) * (int(((blockDim)).x) / int(32))) * int(512));
}

__device__ uint Linear_calcOffset_1(Linear_0 this_11)
{
    uint3  threadIdx_1 = ((threadIdx));
    return uint((int(threadIdx_1.x) / int(32) + int(threadIdx_1.y) * (int(((blockDim)).x) / int(32))) * int(256));
}

__device__ float * Linear_outBufferForCurrentWarp_0(Linear_0 this_12)
{
    return &(*(&shared_output_buffer_0))[Linear_calcOffset_0(this_12)];
}

__device__ void Linear_moveOutputsToLocalArray_0(Linear_0 this_13, FixedArray<float, 16>  * outputs_0, float * bias_1)
{
    float * outPtr_0 = Linear_outBufferForCurrentWarp_0(this_13);
    Linear_loadArray_0(this_13, outPtr_0, outputs_0);
    int i_8 = int(0);
    for(;;)
    {
        if(i_8 < int(16))
        {
        }
        else
        {
            break;
        }
        *(&(*outputs_0)[i_8]) = *(&(*outputs_0)[i_8]) + *(&bias_1[i_8]);
        i_8 = i_8 + int(1);
    }
    return;
}

__device__ void _inline_matmul_0(float * input_2, float * weights_1, float * output_1)
{
    wmma_inline_matmul< (int(16)), (int(32)), (int(16)), (int(16)), (int(16)), (int(8)) >((input_2), (weights_1), (output_1));
    return;
}

__device__ void _inline_matmul_1(float * input_3, float * weights_2, float * output_2)
{
    wmma_inline_matmul< (int(32)), (int(16)), (int(16)), (int(16)), (int(16)), (int(8)) >((input_3), (weights_2), (output_2));
    return;
}

__device__ float * Linear_wtBufferForCurrentWarp_0(Linear_0 this_14)
{
    return &(*(&shared_weights_buffer_0))[Linear_calcOffset_1(this_14)];
}

__device__ float * Linear_moveWeightsToSharedMem_0(Linear_0 this_15)
{
    float * wtPtr_0 = Linear_wtBufferForCurrentWarp_0(this_15);
    int threadIdInWarp_0 = int(((threadIdx)).x) % int(32);
    bool _S28 = threadIdInWarp_0 >= int(16);
    uint _S29 = uint(threadIdInWarp_0);
    uint2  _S30 = make_uint2 (_S29, 0U);
    uint2  _S31 = make_uint2 (_S29, 1U);
    uint2  _S32 = make_uint2 (_S29, 2U);
    uint2  _S33 = make_uint2 (_S29, 3U);
    uint2  _S34 = make_uint2 (_S29, 4U);
    uint2  _S35 = make_uint2 (_S29, 5U);
    uint2  _S36 = make_uint2 (_S29, 6U);
    uint2  _S37 = make_uint2 (_S29, 7U);
    uint2  _S38 = make_uint2 (_S29, 8U);
    uint2  _S39 = make_uint2 (_S29, 9U);
    uint2  _S40 = make_uint2 (_S29, 10U);
    uint2  _S41 = make_uint2 (_S29, 11U);
    uint2  _S42 = make_uint2 (_S29, 12U);
    uint2  _S43 = make_uint2 (_S29, 13U);
    uint2  _S44 = make_uint2 (_S29, 14U);
    uint2  _S45 = make_uint2 (_S29, 15U);
    for(;;)
    {
        for(;;)
        {
            for(;;)
            {
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S28)
                            {
                                break;
                            }
                            *(&wtPtr_0[threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_0, _S30);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S28)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(16) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_0, _S31);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S28)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(32) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_0, _S32);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S28)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(48) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_0, _S33);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S28)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(64) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_0, _S34);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S28)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(80) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_0, _S35);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S28)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(96) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_0, _S36);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S28)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(112) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_0, _S37);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S28)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(128) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_0, _S38);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S28)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(144) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_0, _S39);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S28)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(160) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_0, _S40);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S28)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(176) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_0, _S41);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S28)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(192) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_0, _S42);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S28)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(208) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_0, _S43);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S28)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(224) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_0, _S44);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S28)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(240) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_0, _S45);
                            break;
                        }
                        break;
                    }
                    break;
                }
                break;
            }
            break;
        }
        break;
    }
    return wtPtr_0;
}

__device__ float * Linear_moveWeightsToSharedMem_1(Linear_0 this_16)
{
    float * wtPtr_1 = Linear_wtBufferForCurrentWarp_0(this_16);
    int threadIdInWarp_1 = int(((threadIdx)).x) % int(32);
    bool _S46 = threadIdInWarp_1 >= int(16);
    int _S47 = threadIdInWarp_1 * int(16);
    uint _S48 = uint(threadIdInWarp_1);
    uint2  _S49 = make_uint2 (_S48, 0U);
    uint2  _S50 = make_uint2 (_S48, 1U);
    uint2  _S51 = make_uint2 (_S48, 2U);
    uint2  _S52 = make_uint2 (_S48, 3U);
    uint2  _S53 = make_uint2 (_S48, 4U);
    uint2  _S54 = make_uint2 (_S48, 5U);
    uint2  _S55 = make_uint2 (_S48, 6U);
    uint2  _S56 = make_uint2 (_S48, 7U);
    uint2  _S57 = make_uint2 (_S48, 8U);
    uint2  _S58 = make_uint2 (_S48, 9U);
    uint2  _S59 = make_uint2 (_S48, 10U);
    uint2  _S60 = make_uint2 (_S48, 11U);
    uint2  _S61 = make_uint2 (_S48, 12U);
    uint2  _S62 = make_uint2 (_S48, 13U);
    uint2  _S63 = make_uint2 (_S48, 14U);
    uint2  _S64 = make_uint2 (_S48, 15U);
    for(;;)
    {
        for(;;)
        {
            for(;;)
            {
                for(;;)
                {
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S46)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S47]) = DiffTensorView_load_1(this_16.weights_0, _S49);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S46)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S47 + int(1)]) = DiffTensorView_load_1(this_16.weights_0, _S50);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S46)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S47 + int(2)]) = DiffTensorView_load_1(this_16.weights_0, _S51);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S46)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S47 + int(3)]) = DiffTensorView_load_1(this_16.weights_0, _S52);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S46)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S47 + int(4)]) = DiffTensorView_load_1(this_16.weights_0, _S53);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S46)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S47 + int(5)]) = DiffTensorView_load_1(this_16.weights_0, _S54);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S46)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S47 + int(6)]) = DiffTensorView_load_1(this_16.weights_0, _S55);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S46)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S47 + int(7)]) = DiffTensorView_load_1(this_16.weights_0, _S56);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S46)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S47 + int(8)]) = DiffTensorView_load_1(this_16.weights_0, _S57);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S46)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S47 + int(9)]) = DiffTensorView_load_1(this_16.weights_0, _S58);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S46)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S47 + int(10)]) = DiffTensorView_load_1(this_16.weights_0, _S59);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S46)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S47 + int(11)]) = DiffTensorView_load_1(this_16.weights_0, _S60);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S46)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S47 + int(12)]) = DiffTensorView_load_1(this_16.weights_0, _S61);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S46)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S47 + int(13)]) = DiffTensorView_load_1(this_16.weights_0, _S62);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S46)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S47 + int(14)]) = DiffTensorView_load_1(this_16.weights_0, _S63);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S46)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S47 + int(15)]) = DiffTensorView_load_1(this_16.weights_0, _S64);
                            break;
                        }
                        break;
                    }
                    break;
                }
                break;
            }
            break;
        }
        break;
    }
    return wtPtr_1;
}

__device__ float * Linear_storeArray_0(Linear_0 this_17, float * memptr_1, FixedArray<float, 16>  input_4)
{
    int _S65 = int(uint(int(((threadIdx)).x) % int(32)));
    *(&memptr_1[_S65]) = input_4[int(0)];
    *(&memptr_1[int(32) + _S65]) = input_4[int(1)];
    *(&memptr_1[int(64) + _S65]) = input_4[int(2)];
    *(&memptr_1[int(96) + _S65]) = input_4[int(3)];
    *(&memptr_1[int(128) + _S65]) = input_4[int(4)];
    *(&memptr_1[int(160) + _S65]) = input_4[int(5)];
    *(&memptr_1[int(192) + _S65]) = input_4[int(6)];
    *(&memptr_1[int(224) + _S65]) = input_4[int(7)];
    *(&memptr_1[int(256) + _S65]) = input_4[int(8)];
    *(&memptr_1[int(288) + _S65]) = input_4[int(9)];
    *(&memptr_1[int(320) + _S65]) = input_4[int(10)];
    *(&memptr_1[int(352) + _S65]) = input_4[int(11)];
    *(&memptr_1[int(384) + _S65]) = input_4[int(12)];
    *(&memptr_1[int(416) + _S65]) = input_4[int(13)];
    *(&memptr_1[int(448) + _S65]) = input_4[int(14)];
    *(&memptr_1[int(480) + _S65]) = input_4[int(15)];
    return memptr_1;
}

__device__ float * Linear_storeArray_1(Linear_0 this_18, float * memptr_2, FixedArray<float, 16>  input_5)
{
    int _S66 = int(uint(int(((threadIdx)).x) % int(32))) * int(16);
    *(&memptr_2[_S66]) = input_5[int(0)];
    *(&memptr_2[_S66 + int(1)]) = input_5[int(1)];
    *(&memptr_2[_S66 + int(2)]) = input_5[int(2)];
    *(&memptr_2[_S66 + int(3)]) = input_5[int(3)];
    *(&memptr_2[_S66 + int(4)]) = input_5[int(4)];
    *(&memptr_2[_S66 + int(5)]) = input_5[int(5)];
    *(&memptr_2[_S66 + int(6)]) = input_5[int(6)];
    *(&memptr_2[_S66 + int(7)]) = input_5[int(7)];
    *(&memptr_2[_S66 + int(8)]) = input_5[int(8)];
    *(&memptr_2[_S66 + int(9)]) = input_5[int(9)];
    *(&memptr_2[_S66 + int(10)]) = input_5[int(10)];
    *(&memptr_2[_S66 + int(11)]) = input_5[int(11)];
    *(&memptr_2[_S66 + int(12)]) = input_5[int(12)];
    *(&memptr_2[_S66 + int(13)]) = input_5[int(13)];
    *(&memptr_2[_S66 + int(14)]) = input_5[int(14)];
    *(&memptr_2[_S66 + int(15)]) = input_5[int(15)];
    return memptr_2;
}

__device__ float * Linear_inpBufferForCurrentWarp_0(Linear_0 this_19)
{
    return &(*(&shared_inputs_buffer_0))[Linear_calcOffset_0(this_19)];
}

__device__ float * Linear_moveInputsToSharedMem_0(Linear_0 this_20, FixedArray<float, 16>  input_6)
{
    float * inPtr_0 = Linear_inpBufferForCurrentWarp_0(this_20);
    float * _S67 = Linear_storeArray_1(this_20, inPtr_0, input_6);
    return _S67;
}

__device__ float * Linear_moveDOutputsToSharedMem_0(Linear_0 this_21, FixedArray<float, 16>  d_output_0)
{
    float * outPtr_1 = Linear_outBufferForCurrentWarp_0(this_21);
    float * _S68 = Linear_storeArray_0(this_21, outPtr_1, d_output_0);
    return _S68;
}

__device__ float * Linear_moveDInputsToSharedMem_0(Linear_0 this_22, FixedArray<float, 16>  input_7)
{
    float * inPtr_1 = Linear_inpBufferForCurrentWarp_0(this_22);
    float * _S69 = Linear_storeArray_0(this_22, inPtr_1, input_7);
    return _S69;
}

__device__ uint WaveGetActiveMask_0(uint _S70)
{
    return _S70;
}

__device__ float WaveActiveSum_0(float expr_0, uint _S71)
{
    float _S72 = (_waveSum((WaveGetActiveMask_0(_S71)), (expr_0)));
    return _S72;
}

__device__ bool WaveIsFirstLane_0(uint _S73)
{
    uint _S74 = WaveGetActiveMask_0(_S73);
    bool _S75 = ((((_S74) & -(_S74)) == (WarpMask(1) << _getLaneId())));
    return _S75;
}

struct DiffPair_Feature_0
{
    Feature_0 primal_1;
    s_diff_Feature_0 differential_0;
};

__device__ void Linear_eval_bwd_0(Linear_0 this_23, DiffPair_Feature_0 * in_feature_pair_0, s_diff_Feature_0 d_output_1, uint _S76)
{
    uint _S77 = 0U;
    float * dOutPtr_0 = Linear_moveInputsToSharedMem_0(this_23, d_output_1.vals_0);
    float * wtPtr_2 = Linear_moveWeightsToSharedMem_1(this_23);
    float * dInPtr_0 = Linear_outBufferForCurrentWarp_0(this_23);
    _inline_matmul_1(dOutPtr_0, wtPtr_2, dInPtr_0);
    s_diff_Feature_0 d_input_feature_0;
    Linear_loadArray_0(this_23, dInPtr_0, &(&d_input_feature_0)->vals_0);
    DiffPair_Feature_0 _S78 = *in_feature_pair_0;
    in_feature_pair_0->primal_1 = (*in_feature_pair_0).primal_1;
    in_feature_pair_0->differential_0 = d_input_feature_0;
    float * inPtr_2 = Linear_moveDInputsToSharedMem_0(this_23, _S78.primal_1.vals_1);
    float * outPtr_2 = Linear_moveDOutputsToSharedMem_0(this_23, d_output_1.vals_0);
    float * wtPtr_3 = Linear_wtBufferForCurrentWarp_0(this_23);
    _inline_matmul_0(outPtr_2, inPtr_2, wtPtr_3);
    int threadIdInWarp_2 = int(((threadIdx)).x) % int(32);
    bool _S79 = threadIdInWarp_2 >= int(16);
    uint _S80 = uint(threadIdInWarp_2);
    int _S81 = threadIdInWarp_2 * int(16);
    uint2  _S82 = make_uint2 (0U, _S80);
    uint2  _S83 = make_uint2 (1U, _S80);
    uint2  _S84 = make_uint2 (2U, _S80);
    uint2  _S85 = make_uint2 (3U, _S80);
    uint2  _S86 = make_uint2 (4U, _S80);
    uint2  _S87 = make_uint2 (5U, _S80);
    uint2  _S88 = make_uint2 (6U, _S80);
    uint2  _S89 = make_uint2 (7U, _S80);
    uint2  _S90 = make_uint2 (8U, _S80);
    uint2  _S91 = make_uint2 (9U, _S80);
    uint2  _S92 = make_uint2 (10U, _S80);
    uint2  _S93 = make_uint2 (11U, _S80);
    uint2  _S94 = make_uint2 (12U, _S80);
    uint2  _S95 = make_uint2 (13U, _S80);
    uint2  _S96 = make_uint2 (14U, _S80);
    uint2  _S97 = make_uint2 (15U, _S80);
    for(;;)
    {
        uint _S98 = 0U;
        for(;;)
        {
            for(;;)
            {
                for(;;)
                {
                    uint _S99 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S100 = __ballot_sync(_S76, _S79);
                            if(_S79)
                            {
                                uint _S101 = __ballot_sync(_S76, false);
                                uint _S102 = __ballot_sync(_S76, true);
                                break;
                            }
                            else
                            {
                                uint _S103 = __ballot_sync(_S76, true);
                            }
                            float oldVal_1;
                            *((&oldVal_1)) = atomicAdd((this_23.weights_0.diff_1.diff_0).data_ptr_at<float>((_S82)), (*(&wtPtr_3[_S81])));
                            uint _S104 = __ballot_sync(_S76, true);
                            break;
                        }
                        uint _S105 = __ballot_sync(_S76, true);
                        _S99 = _S105;
                        break;
                    }
                    uint _S106 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S107 = __ballot_sync(_S99, _S79);
                            if(_S79)
                            {
                                uint _S108 = __ballot_sync(_S99, false);
                                uint _S109 = __ballot_sync(_S99, true);
                                break;
                            }
                            else
                            {
                                uint _S110 = __ballot_sync(_S99, true);
                            }
                            float oldVal_2;
                            *((&oldVal_2)) = atomicAdd((this_23.weights_0.diff_1.diff_0).data_ptr_at<float>((_S83)), (*(&wtPtr_3[_S81 + int(1)])));
                            uint _S111 = __ballot_sync(_S99, true);
                            break;
                        }
                        uint _S112 = __ballot_sync(_S99, true);
                        _S106 = _S112;
                        break;
                    }
                    uint _S113 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S114 = __ballot_sync(_S106, _S79);
                            if(_S79)
                            {
                                uint _S115 = __ballot_sync(_S106, false);
                                uint _S116 = __ballot_sync(_S106, true);
                                break;
                            }
                            else
                            {
                                uint _S117 = __ballot_sync(_S106, true);
                            }
                            float oldVal_3;
                            *((&oldVal_3)) = atomicAdd((this_23.weights_0.diff_1.diff_0).data_ptr_at<float>((_S84)), (*(&wtPtr_3[_S81 + int(2)])));
                            uint _S118 = __ballot_sync(_S106, true);
                            break;
                        }
                        uint _S119 = __ballot_sync(_S106, true);
                        _S113 = _S119;
                        break;
                    }
                    uint _S120 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S121 = __ballot_sync(_S113, _S79);
                            if(_S79)
                            {
                                uint _S122 = __ballot_sync(_S113, false);
                                uint _S123 = __ballot_sync(_S113, true);
                                break;
                            }
                            else
                            {
                                uint _S124 = __ballot_sync(_S113, true);
                            }
                            float oldVal_4;
                            *((&oldVal_4)) = atomicAdd((this_23.weights_0.diff_1.diff_0).data_ptr_at<float>((_S85)), (*(&wtPtr_3[_S81 + int(3)])));
                            uint _S125 = __ballot_sync(_S113, true);
                            break;
                        }
                        uint _S126 = __ballot_sync(_S113, true);
                        _S120 = _S126;
                        break;
                    }
                    uint _S127 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S128 = __ballot_sync(_S120, _S79);
                            if(_S79)
                            {
                                uint _S129 = __ballot_sync(_S120, false);
                                uint _S130 = __ballot_sync(_S120, true);
                                break;
                            }
                            else
                            {
                                uint _S131 = __ballot_sync(_S120, true);
                            }
                            float oldVal_5;
                            *((&oldVal_5)) = atomicAdd((this_23.weights_0.diff_1.diff_0).data_ptr_at<float>((_S86)), (*(&wtPtr_3[_S81 + int(4)])));
                            uint _S132 = __ballot_sync(_S120, true);
                            break;
                        }
                        uint _S133 = __ballot_sync(_S120, true);
                        _S127 = _S133;
                        break;
                    }
                    uint _S134 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S135 = __ballot_sync(_S127, _S79);
                            if(_S79)
                            {
                                uint _S136 = __ballot_sync(_S127, false);
                                uint _S137 = __ballot_sync(_S127, true);
                                break;
                            }
                            else
                            {
                                uint _S138 = __ballot_sync(_S127, true);
                            }
                            float oldVal_6;
                            *((&oldVal_6)) = atomicAdd((this_23.weights_0.diff_1.diff_0).data_ptr_at<float>((_S87)), (*(&wtPtr_3[_S81 + int(5)])));
                            uint _S139 = __ballot_sync(_S127, true);
                            break;
                        }
                        uint _S140 = __ballot_sync(_S127, true);
                        _S134 = _S140;
                        break;
                    }
                    uint _S141 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S142 = __ballot_sync(_S134, _S79);
                            if(_S79)
                            {
                                uint _S143 = __ballot_sync(_S134, false);
                                uint _S144 = __ballot_sync(_S134, true);
                                break;
                            }
                            else
                            {
                                uint _S145 = __ballot_sync(_S134, true);
                            }
                            float oldVal_7;
                            *((&oldVal_7)) = atomicAdd((this_23.weights_0.diff_1.diff_0).data_ptr_at<float>((_S88)), (*(&wtPtr_3[_S81 + int(6)])));
                            uint _S146 = __ballot_sync(_S134, true);
                            break;
                        }
                        uint _S147 = __ballot_sync(_S134, true);
                        _S141 = _S147;
                        break;
                    }
                    uint _S148 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S149 = __ballot_sync(_S141, _S79);
                            if(_S79)
                            {
                                uint _S150 = __ballot_sync(_S141, false);
                                uint _S151 = __ballot_sync(_S141, true);
                                break;
                            }
                            else
                            {
                                uint _S152 = __ballot_sync(_S141, true);
                            }
                            float oldVal_8;
                            *((&oldVal_8)) = atomicAdd((this_23.weights_0.diff_1.diff_0).data_ptr_at<float>((_S89)), (*(&wtPtr_3[_S81 + int(7)])));
                            uint _S153 = __ballot_sync(_S141, true);
                            break;
                        }
                        uint _S154 = __ballot_sync(_S141, true);
                        _S148 = _S154;
                        break;
                    }
                    uint _S155 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S156 = __ballot_sync(_S148, _S79);
                            if(_S79)
                            {
                                uint _S157 = __ballot_sync(_S148, false);
                                uint _S158 = __ballot_sync(_S148, true);
                                break;
                            }
                            else
                            {
                                uint _S159 = __ballot_sync(_S148, true);
                            }
                            float oldVal_9;
                            *((&oldVal_9)) = atomicAdd((this_23.weights_0.diff_1.diff_0).data_ptr_at<float>((_S90)), (*(&wtPtr_3[_S81 + int(8)])));
                            uint _S160 = __ballot_sync(_S148, true);
                            break;
                        }
                        uint _S161 = __ballot_sync(_S148, true);
                        _S155 = _S161;
                        break;
                    }
                    uint _S162 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S163 = __ballot_sync(_S155, _S79);
                            if(_S79)
                            {
                                uint _S164 = __ballot_sync(_S155, false);
                                uint _S165 = __ballot_sync(_S155, true);
                                break;
                            }
                            else
                            {
                                uint _S166 = __ballot_sync(_S155, true);
                            }
                            float oldVal_10;
                            *((&oldVal_10)) = atomicAdd((this_23.weights_0.diff_1.diff_0).data_ptr_at<float>((_S91)), (*(&wtPtr_3[_S81 + int(9)])));
                            uint _S167 = __ballot_sync(_S155, true);
                            break;
                        }
                        uint _S168 = __ballot_sync(_S155, true);
                        _S162 = _S168;
                        break;
                    }
                    uint _S169 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S170 = __ballot_sync(_S162, _S79);
                            if(_S79)
                            {
                                uint _S171 = __ballot_sync(_S162, false);
                                uint _S172 = __ballot_sync(_S162, true);
                                break;
                            }
                            else
                            {
                                uint _S173 = __ballot_sync(_S162, true);
                            }
                            float oldVal_11;
                            *((&oldVal_11)) = atomicAdd((this_23.weights_0.diff_1.diff_0).data_ptr_at<float>((_S92)), (*(&wtPtr_3[_S81 + int(10)])));
                            uint _S174 = __ballot_sync(_S162, true);
                            break;
                        }
                        uint _S175 = __ballot_sync(_S162, true);
                        _S169 = _S175;
                        break;
                    }
                    uint _S176 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S177 = __ballot_sync(_S169, _S79);
                            if(_S79)
                            {
                                uint _S178 = __ballot_sync(_S169, false);
                                uint _S179 = __ballot_sync(_S169, true);
                                break;
                            }
                            else
                            {
                                uint _S180 = __ballot_sync(_S169, true);
                            }
                            float oldVal_12;
                            *((&oldVal_12)) = atomicAdd((this_23.weights_0.diff_1.diff_0).data_ptr_at<float>((_S93)), (*(&wtPtr_3[_S81 + int(11)])));
                            uint _S181 = __ballot_sync(_S169, true);
                            break;
                        }
                        uint _S182 = __ballot_sync(_S169, true);
                        _S176 = _S182;
                        break;
                    }
                    uint _S183 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S184 = __ballot_sync(_S176, _S79);
                            if(_S79)
                            {
                                uint _S185 = __ballot_sync(_S176, false);
                                uint _S186 = __ballot_sync(_S176, true);
                                break;
                            }
                            else
                            {
                                uint _S187 = __ballot_sync(_S176, true);
                            }
                            float oldVal_13;
                            *((&oldVal_13)) = atomicAdd((this_23.weights_0.diff_1.diff_0).data_ptr_at<float>((_S94)), (*(&wtPtr_3[_S81 + int(12)])));
                            uint _S188 = __ballot_sync(_S176, true);
                            break;
                        }
                        uint _S189 = __ballot_sync(_S176, true);
                        _S183 = _S189;
                        break;
                    }
                    uint _S190 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S191 = __ballot_sync(_S183, _S79);
                            if(_S79)
                            {
                                uint _S192 = __ballot_sync(_S183, false);
                                uint _S193 = __ballot_sync(_S183, true);
                                break;
                            }
                            else
                            {
                                uint _S194 = __ballot_sync(_S183, true);
                            }
                            float oldVal_14;
                            *((&oldVal_14)) = atomicAdd((this_23.weights_0.diff_1.diff_0).data_ptr_at<float>((_S95)), (*(&wtPtr_3[_S81 + int(13)])));
                            uint _S195 = __ballot_sync(_S183, true);
                            break;
                        }
                        uint _S196 = __ballot_sync(_S183, true);
                        _S190 = _S196;
                        break;
                    }
                    uint _S197 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S198 = __ballot_sync(_S190, _S79);
                            if(_S79)
                            {
                                uint _S199 = __ballot_sync(_S190, false);
                                uint _S200 = __ballot_sync(_S190, true);
                                break;
                            }
                            else
                            {
                                uint _S201 = __ballot_sync(_S190, true);
                            }
                            float oldVal_15;
                            *((&oldVal_15)) = atomicAdd((this_23.weights_0.diff_1.diff_0).data_ptr_at<float>((_S96)), (*(&wtPtr_3[_S81 + int(14)])));
                            uint _S202 = __ballot_sync(_S190, true);
                            break;
                        }
                        uint _S203 = __ballot_sync(_S190, true);
                        _S197 = _S203;
                        break;
                    }
                    uint _S204 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S205 = __ballot_sync(_S197, _S79);
                            if(_S79)
                            {
                                uint _S206 = __ballot_sync(_S197, false);
                                uint _S207 = __ballot_sync(_S197, true);
                                break;
                            }
                            else
                            {
                                uint _S208 = __ballot_sync(_S197, true);
                            }
                            float oldVal_16;
                            *((&oldVal_16)) = atomicAdd((this_23.weights_0.diff_1.diff_0).data_ptr_at<float>((_S97)), (*(&wtPtr_3[_S81 + int(15)])));
                            uint _S209 = __ballot_sync(_S197, true);
                            break;
                        }
                        uint _S210 = __ballot_sync(_S197, true);
                        _S204 = _S210;
                        break;
                    }
                    uint _S211 = __ballot_sync(_S204, false);
                    uint _S212 = __ballot_sync(_S204, false);
                    uint _S213 = __ballot_sync(_S76, true);
                    break;
                }
                uint _S214 = __ballot_sync(_S76, true);
                break;
            }
            uint _S215 = __ballot_sync(_S76, true);
            _S98 = _S215;
            break;
        }
        uint _S216 = __ballot_sync(_S98, false);
        uint _S217 = __ballot_sync(_S98, false);
        uint _S218 = __ballot_sync(_S76, true);
        _S77 = _S218;
        break;
    }
    float _S219 = WaveActiveSum_0(d_output_1.vals_0[int(0)], _S77);
    bool _S220 = WaveIsFirstLane_0(_S77);
    uint _S221 = __ballot_sync(_S77, _S220);
    uint _S222;
    if(_S220)
    {
        float oldVal_17;
        *((&oldVal_17)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((0U)), (_S219));
        uint _S223 = __ballot_sync(_S77, true);
        _S222 = _S223;
    }
    else
    {
        uint _S224 = __ballot_sync(_S77, true);
        _S222 = _S224;
    }
    float _S225 = WaveActiveSum_0(d_output_1.vals_0[int(1)], _S222);
    bool _S226 = WaveIsFirstLane_0(_S222);
    uint _S227 = __ballot_sync(_S222, _S226);
    if(_S226)
    {
        float oldVal_18;
        *((&oldVal_18)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((1U)), (_S225));
        uint _S228 = __ballot_sync(_S222, true);
        _S222 = _S228;
    }
    else
    {
        uint _S229 = __ballot_sync(_S222, true);
        _S222 = _S229;
    }
    float _S230 = WaveActiveSum_0(d_output_1.vals_0[int(2)], _S222);
    bool _S231 = WaveIsFirstLane_0(_S222);
    uint _S232 = __ballot_sync(_S222, _S231);
    if(_S231)
    {
        float oldVal_19;
        *((&oldVal_19)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((2U)), (_S230));
        uint _S233 = __ballot_sync(_S222, true);
        _S222 = _S233;
    }
    else
    {
        uint _S234 = __ballot_sync(_S222, true);
        _S222 = _S234;
    }
    float _S235 = WaveActiveSum_0(d_output_1.vals_0[int(3)], _S222);
    bool _S236 = WaveIsFirstLane_0(_S222);
    uint _S237 = __ballot_sync(_S222, _S236);
    if(_S236)
    {
        float oldVal_20;
        *((&oldVal_20)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((3U)), (_S235));
        uint _S238 = __ballot_sync(_S222, true);
        _S222 = _S238;
    }
    else
    {
        uint _S239 = __ballot_sync(_S222, true);
        _S222 = _S239;
    }
    float _S240 = WaveActiveSum_0(d_output_1.vals_0[int(4)], _S222);
    bool _S241 = WaveIsFirstLane_0(_S222);
    uint _S242 = __ballot_sync(_S222, _S241);
    if(_S241)
    {
        float oldVal_21;
        *((&oldVal_21)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((4U)), (_S240));
        uint _S243 = __ballot_sync(_S222, true);
        _S222 = _S243;
    }
    else
    {
        uint _S244 = __ballot_sync(_S222, true);
        _S222 = _S244;
    }
    float _S245 = WaveActiveSum_0(d_output_1.vals_0[int(5)], _S222);
    bool _S246 = WaveIsFirstLane_0(_S222);
    uint _S247 = __ballot_sync(_S222, _S246);
    if(_S246)
    {
        float oldVal_22;
        *((&oldVal_22)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((5U)), (_S245));
        uint _S248 = __ballot_sync(_S222, true);
        _S222 = _S248;
    }
    else
    {
        uint _S249 = __ballot_sync(_S222, true);
        _S222 = _S249;
    }
    float _S250 = WaveActiveSum_0(d_output_1.vals_0[int(6)], _S222);
    bool _S251 = WaveIsFirstLane_0(_S222);
    uint _S252 = __ballot_sync(_S222, _S251);
    if(_S251)
    {
        float oldVal_23;
        *((&oldVal_23)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((6U)), (_S250));
        uint _S253 = __ballot_sync(_S222, true);
        _S222 = _S253;
    }
    else
    {
        uint _S254 = __ballot_sync(_S222, true);
        _S222 = _S254;
    }
    float _S255 = WaveActiveSum_0(d_output_1.vals_0[int(7)], _S222);
    bool _S256 = WaveIsFirstLane_0(_S222);
    uint _S257 = __ballot_sync(_S222, _S256);
    if(_S256)
    {
        float oldVal_24;
        *((&oldVal_24)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((7U)), (_S255));
        uint _S258 = __ballot_sync(_S222, true);
        _S222 = _S258;
    }
    else
    {
        uint _S259 = __ballot_sync(_S222, true);
        _S222 = _S259;
    }
    float _S260 = WaveActiveSum_0(d_output_1.vals_0[int(8)], _S222);
    bool _S261 = WaveIsFirstLane_0(_S222);
    uint _S262 = __ballot_sync(_S222, _S261);
    if(_S261)
    {
        float oldVal_25;
        *((&oldVal_25)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((8U)), (_S260));
        uint _S263 = __ballot_sync(_S222, true);
        _S222 = _S263;
    }
    else
    {
        uint _S264 = __ballot_sync(_S222, true);
        _S222 = _S264;
    }
    float _S265 = WaveActiveSum_0(d_output_1.vals_0[int(9)], _S222);
    bool _S266 = WaveIsFirstLane_0(_S222);
    uint _S267 = __ballot_sync(_S222, _S266);
    if(_S266)
    {
        float oldVal_26;
        *((&oldVal_26)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((9U)), (_S265));
        uint _S268 = __ballot_sync(_S222, true);
        _S222 = _S268;
    }
    else
    {
        uint _S269 = __ballot_sync(_S222, true);
        _S222 = _S269;
    }
    float _S270 = WaveActiveSum_0(d_output_1.vals_0[int(10)], _S222);
    bool _S271 = WaveIsFirstLane_0(_S222);
    uint _S272 = __ballot_sync(_S222, _S271);
    if(_S271)
    {
        float oldVal_27;
        *((&oldVal_27)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((10U)), (_S270));
        uint _S273 = __ballot_sync(_S222, true);
        _S222 = _S273;
    }
    else
    {
        uint _S274 = __ballot_sync(_S222, true);
        _S222 = _S274;
    }
    float _S275 = WaveActiveSum_0(d_output_1.vals_0[int(11)], _S222);
    bool _S276 = WaveIsFirstLane_0(_S222);
    uint _S277 = __ballot_sync(_S222, _S276);
    if(_S276)
    {
        float oldVal_28;
        *((&oldVal_28)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((11U)), (_S275));
        uint _S278 = __ballot_sync(_S222, true);
        _S222 = _S278;
    }
    else
    {
        uint _S279 = __ballot_sync(_S222, true);
        _S222 = _S279;
    }
    float _S280 = WaveActiveSum_0(d_output_1.vals_0[int(12)], _S222);
    bool _S281 = WaveIsFirstLane_0(_S222);
    uint _S282 = __ballot_sync(_S222, _S281);
    if(_S281)
    {
        float oldVal_29;
        *((&oldVal_29)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((12U)), (_S280));
        uint _S283 = __ballot_sync(_S222, true);
        _S222 = _S283;
    }
    else
    {
        uint _S284 = __ballot_sync(_S222, true);
        _S222 = _S284;
    }
    float _S285 = WaveActiveSum_0(d_output_1.vals_0[int(13)], _S222);
    bool _S286 = WaveIsFirstLane_0(_S222);
    uint _S287 = __ballot_sync(_S222, _S286);
    if(_S286)
    {
        float oldVal_30;
        *((&oldVal_30)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((13U)), (_S285));
        uint _S288 = __ballot_sync(_S222, true);
        _S222 = _S288;
    }
    else
    {
        uint _S289 = __ballot_sync(_S222, true);
        _S222 = _S289;
    }
    float _S290 = WaveActiveSum_0(d_output_1.vals_0[int(14)], _S222);
    bool _S291 = WaveIsFirstLane_0(_S222);
    uint _S292 = __ballot_sync(_S222, _S291);
    if(_S291)
    {
        float oldVal_31;
        *((&oldVal_31)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((14U)), (_S290));
        uint _S293 = __ballot_sync(_S222, true);
        _S222 = _S293;
    }
    else
    {
        uint _S294 = __ballot_sync(_S222, true);
        _S222 = _S294;
    }
    float _S295 = WaveActiveSum_0(d_output_1.vals_0[int(15)], _S222);
    bool _S296 = WaveIsFirstLane_0(_S222);
    uint _S297 = __ballot_sync(_S222, _S296);
    if(_S296)
    {
        float oldVal_32;
        *((&oldVal_32)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((15U)), (_S295));
    }
    return;
}

__device__ Feature_0 Linear_eval_0(Linear_0 this_24, Feature_0 in_feature_0)
{
    float * inPtr_3 = Linear_moveInputsToSharedMem_0(this_24, in_feature_0.vals_1);
    float * wtPtr_4 = Linear_moveWeightsToSharedMem_0(this_24);
    float * outPtr_3 = Linear_outBufferForCurrentWarp_0(this_24);
    _inline_matmul_1(inPtr_3, wtPtr_4, outPtr_3);
    Feature_0 out_feature_0;
    float * _S298 = ((this_24.bias_0.primal_0).data_ptr<float>());
    Linear_moveOutputsToLocalArray_0(this_24, &(&out_feature_0)->vals_1, _S298);
    return out_feature_0;
}

__device__ void _d_max_0(DiffPair_float_0 * dpx_0, DiffPair_float_0 * dpy_0, float dOut_2)
{
    DiffPair_float_0 _S299 = *dpx_0;
    float _S300;
    if((*dpx_0).primal_1 > (*dpy_0).primal_1)
    {
        _S300 = dOut_2;
    }
    else
    {
        _S300 = 0.0f;
    }
    dpx_0->primal_1 = _S299.primal_1;
    dpx_0->differential_0 = _S300;
    DiffPair_float_0 _S301 = *dpy_0;
    if((*dpy_0).primal_1 > _S299.primal_1)
    {
        _S300 = dOut_2;
    }
    else
    {
        _S300 = 0.0f;
    }
    dpy_0->primal_1 = _S301.primal_1;
    dpy_0->differential_0 = _S300;
    return;
}

__device__ DiffPair_float_0 _d_max_1(DiffPair_float_0 dpx_1, DiffPair_float_0 dpy_1)
{
    float _S302 = (F32_max((dpx_1.primal_1), (dpy_1.primal_1)));
    float _S303;
    if(dpx_1.primal_1 > dpy_1.primal_1)
    {
        _S303 = dpx_1.differential_0;
    }
    else
    {
        _S303 = dpy_1.differential_0;
    }
    DiffPair_float_0 _S304 = { _S302, _S303 };
    return _S304;
}

__device__ float s_primal_ctx_max_0(float _S305, float _S306)
{
    return (F32_max((_S305), (_S306)));
}

__device__ void s_bwd_prop_max_0(DiffPair_float_0 * _S307, DiffPair_float_0 * _S308, float _S309)
{
    _d_max_0(_S307, _S308, _S309);
    return;
}

struct MLP_0
{
    FixedArray<Linear_0, 3>  layers_0;
};

__device__ Feature_0 MLP_eval_0(MLP_0 this_25, Feature_0 in_feature_1)
{
    Feature_0 out_feature_1;
    Feature_0 _S310 = Linear_eval_0(this_25.layers_0[int(0)], in_feature_1);
    out_feature_1 = _S310;
    *(&(&out_feature_1)->vals_1[int(0)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(0)]))));
    *(&(&out_feature_1)->vals_1[int(1)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(1)]))));
    *(&(&out_feature_1)->vals_1[int(2)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(2)]))));
    *(&(&out_feature_1)->vals_1[int(3)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(3)]))));
    *(&(&out_feature_1)->vals_1[int(4)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(4)]))));
    *(&(&out_feature_1)->vals_1[int(5)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(5)]))));
    *(&(&out_feature_1)->vals_1[int(6)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(6)]))));
    *(&(&out_feature_1)->vals_1[int(7)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(7)]))));
    *(&(&out_feature_1)->vals_1[int(8)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(8)]))));
    *(&(&out_feature_1)->vals_1[int(9)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(9)]))));
    *(&(&out_feature_1)->vals_1[int(10)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(10)]))));
    *(&(&out_feature_1)->vals_1[int(11)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(11)]))));
    *(&(&out_feature_1)->vals_1[int(12)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(12)]))));
    *(&(&out_feature_1)->vals_1[int(13)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(13)]))));
    *(&(&out_feature_1)->vals_1[int(14)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(14)]))));
    *(&(&out_feature_1)->vals_1[int(15)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(15)]))));
    Feature_0 _S311 = Linear_eval_0(this_25.layers_0[int(1)], out_feature_1);
    out_feature_1 = _S311;
    *(&(&out_feature_1)->vals_1[int(0)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(0)]))));
    *(&(&out_feature_1)->vals_1[int(1)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(1)]))));
    *(&(&out_feature_1)->vals_1[int(2)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(2)]))));
    *(&(&out_feature_1)->vals_1[int(3)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(3)]))));
    *(&(&out_feature_1)->vals_1[int(4)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(4)]))));
    *(&(&out_feature_1)->vals_1[int(5)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(5)]))));
    *(&(&out_feature_1)->vals_1[int(6)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(6)]))));
    *(&(&out_feature_1)->vals_1[int(7)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(7)]))));
    *(&(&out_feature_1)->vals_1[int(8)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(8)]))));
    *(&(&out_feature_1)->vals_1[int(9)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(9)]))));
    *(&(&out_feature_1)->vals_1[int(10)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(10)]))));
    *(&(&out_feature_1)->vals_1[int(11)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(11)]))));
    *(&(&out_feature_1)->vals_1[int(12)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(12)]))));
    *(&(&out_feature_1)->vals_1[int(13)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(13)]))));
    *(&(&out_feature_1)->vals_1[int(14)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(14)]))));
    *(&(&out_feature_1)->vals_1[int(15)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(15)]))));
    Feature_0 _S312 = Linear_eval_0(this_25.layers_0[int(2)], out_feature_1);
    out_feature_1 = _S312;
    *(&(&out_feature_1)->vals_1[int(0)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(0)]))));
    *(&(&out_feature_1)->vals_1[int(1)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(1)]))));
    *(&(&out_feature_1)->vals_1[int(2)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(2)]))));
    *(&(&out_feature_1)->vals_1[int(3)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(3)]))));
    *(&(&out_feature_1)->vals_1[int(4)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(4)]))));
    *(&(&out_feature_1)->vals_1[int(5)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(5)]))));
    *(&(&out_feature_1)->vals_1[int(6)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(6)]))));
    *(&(&out_feature_1)->vals_1[int(7)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(7)]))));
    *(&(&out_feature_1)->vals_1[int(8)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(8)]))));
    *(&(&out_feature_1)->vals_1[int(9)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(9)]))));
    *(&(&out_feature_1)->vals_1[int(10)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(10)]))));
    *(&(&out_feature_1)->vals_1[int(11)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(11)]))));
    *(&(&out_feature_1)->vals_1[int(12)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(12)]))));
    *(&(&out_feature_1)->vals_1[int(13)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(13)]))));
    *(&(&out_feature_1)->vals_1[int(14)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(14)]))));
    *(&(&out_feature_1)->vals_1[int(15)]) = (F32_max((0.0f), (*(&(&out_feature_1)->vals_1[int(15)]))));
    return out_feature_1;
}

__device__ void DiffTensorView_storeOnce_forward_0(DiffTensorView_0 this_26, uint4  x_2, DiffPair_float_0 dpval_0)
{
    (this_26.primal_0).store<float>((x_2), (dpval_0.primal_1));
    AtomicAdd_storeOnce_forward_0(this_26.diff_1, x_2, dpval_0.differential_0);
    return;
}

__device__ void DiffTensorView_storeOnce_backward_0(DiffTensorView_0 this_27, uint4  x_3, DiffPair_float_0 * dpval_1)
{
    float _S313 = AtomicAdd_storeOnce_backward_0(this_27.diff_1, x_3);
    dpval_1->primal_1 = (*dpval_1).primal_1;
    dpval_1->differential_0 = _S313;
    return;
}

__device__ void DiffTensorView_storeOnce_0(DiffTensorView_0 this_28, uint4  x_4, float val_0)
{
    (this_28.primal_0).store<float>((x_4), (val_0));
    return;
}

struct s_bwd_prop_getInFeature_Intermediates_0
{
    int _S314;
};

struct s_bwd_prop_MLP_eval_Intermediates_0
{
    Feature_0 _S315;
    Feature_0 _S316;
    Feature_0 _S317;
};

struct s_bwd_prop_renderImage_Intermediates_0
{
    s_bwd_prop_getInFeature_Intermediates_0 _S318;
    Feature_0 _S319;
    s_bwd_prop_MLP_eval_Intermediates_0 _S320;
    Feature_0 _S321;
    int _S322;
    uint _S323;
    uint _S324;
    uint _S325;
};

__device__ float s_primal_ctx_DiffTensorView_load_0(DiffTensorView_0 _S326, uint4  _S327)
{
    return DiffTensorView_load_0(_S326, _S327);
}

__device__ Feature_0 s_bwd_primal_getInFeature_0(DiffTensorView_0 input_8, uint3  idx_1, s_bwd_prop_getInFeature_Intermediates_0 * _s_diff_ctx_0)
{
    _s_diff_ctx_0->_S314 = int(0);
    FixedArray<float, 16>  _S328 = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    uint _S329 = idx_1.x;
    uint _S330 = idx_1.y;
    uint _S331 = idx_1.z;
    uint4  _S332 = make_uint4 (_S329, _S330, _S331, 15U);
    bool _bflag_0 = true;
    int i_9 = int(0);
    Feature_0 output_3;
    (&output_3)->vals_1 = _S328;
    int _pc_0 = int(0);
    for(;;)
    {
        _s_diff_ctx_0->_S314 = _pc_0;
        if(_bflag_0)
        {
        }
        else
        {
            break;
        }
        int _S333;
        if(i_9 < int(15))
        {
            float _S334 = s_primal_ctx_DiffTensorView_load_0(input_8, make_uint4 (_S329, _S330, _S331, uint(i_9)));
            Feature_0 _S335 = output_3;
            *(&(&_S335)->vals_1[i_9]) = _S334;
            _S333 = int(1);
            output_3 = _S335;
        }
        else
        {
            _S333 = int(0);
        }
        if(_S333 != int(1))
        {
            _bflag_0 = false;
        }
        if(_bflag_0)
        {
            i_9 = i_9 + int(1);
        }
        _pc_0 = _pc_0 + int(1);
    }
    *(&(&output_3)->vals_1[15U]) = s_primal_ctx_DiffTensorView_load_0(input_8, _S332);
    return output_3;
}

__device__ Feature_0 s_primal_ctx_Linear_eval_0(Linear_0 _S336, Feature_0 _S337)
{
    Feature_0 _S338 = Linear_eval_0(_S336, _S337);
    return _S338;
}

__device__ Feature_0 s_bwd_primal_MLP_eval_0(MLP_0 this_29, Feature_0 dpin_feature_0, s_bwd_prop_MLP_eval_Intermediates_0 * _s_diff_ctx_1)
{
    FixedArray<float, 16>  _S339 = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    Feature_0 _S340 = { _S339 };
    _s_diff_ctx_1->_S315 = _S340;
    _s_diff_ctx_1->_S316 = _S340;
    _s_diff_ctx_1->_S317 = _S340;
    Feature_0 _S341 = s_primal_ctx_Linear_eval_0(this_29.layers_0[int(0)], dpin_feature_0);
    _s_diff_ctx_1->_S315 = _S341;
    float _S342 = s_primal_ctx_max_0(0.0f, _S341.vals_1[int(0)]);
    Feature_0 _S343 = _S341;
    *(&(&_S343)->vals_1[int(0)]) = _S342;
    *(&(&_S343)->vals_1[int(1)]) = s_primal_ctx_max_0(0.0f, _S341.vals_1[int(1)]);
    *(&(&_S343)->vals_1[int(2)]) = s_primal_ctx_max_0(0.0f, _S341.vals_1[int(2)]);
    *(&(&_S343)->vals_1[int(3)]) = s_primal_ctx_max_0(0.0f, _S341.vals_1[int(3)]);
    *(&(&_S343)->vals_1[int(4)]) = s_primal_ctx_max_0(0.0f, _S341.vals_1[int(4)]);
    *(&(&_S343)->vals_1[int(5)]) = s_primal_ctx_max_0(0.0f, _S341.vals_1[int(5)]);
    *(&(&_S343)->vals_1[int(6)]) = s_primal_ctx_max_0(0.0f, _S341.vals_1[int(6)]);
    *(&(&_S343)->vals_1[int(7)]) = s_primal_ctx_max_0(0.0f, _S341.vals_1[int(7)]);
    *(&(&_S343)->vals_1[int(8)]) = s_primal_ctx_max_0(0.0f, _S341.vals_1[int(8)]);
    *(&(&_S343)->vals_1[int(9)]) = s_primal_ctx_max_0(0.0f, _S341.vals_1[int(9)]);
    *(&(&_S343)->vals_1[int(10)]) = s_primal_ctx_max_0(0.0f, _S341.vals_1[int(10)]);
    *(&(&_S343)->vals_1[int(11)]) = s_primal_ctx_max_0(0.0f, _S341.vals_1[int(11)]);
    *(&(&_S343)->vals_1[int(12)]) = s_primal_ctx_max_0(0.0f, _S341.vals_1[int(12)]);
    *(&(&_S343)->vals_1[int(13)]) = s_primal_ctx_max_0(0.0f, _S341.vals_1[int(13)]);
    *(&(&_S343)->vals_1[int(14)]) = s_primal_ctx_max_0(0.0f, _S341.vals_1[int(14)]);
    *(&(&_S343)->vals_1[int(15)]) = s_primal_ctx_max_0(0.0f, _S341.vals_1[int(15)]);
    Feature_0 _S344 = s_primal_ctx_Linear_eval_0(this_29.layers_0[int(1)], _S343);
    _s_diff_ctx_1->_S316 = _S344;
    float _S345 = s_primal_ctx_max_0(0.0f, _S344.vals_1[int(0)]);
    _S343 = _S344;
    *(&(&_S343)->vals_1[int(0)]) = _S345;
    *(&(&_S343)->vals_1[int(1)]) = s_primal_ctx_max_0(0.0f, _S344.vals_1[int(1)]);
    *(&(&_S343)->vals_1[int(2)]) = s_primal_ctx_max_0(0.0f, _S344.vals_1[int(2)]);
    *(&(&_S343)->vals_1[int(3)]) = s_primal_ctx_max_0(0.0f, _S344.vals_1[int(3)]);
    *(&(&_S343)->vals_1[int(4)]) = s_primal_ctx_max_0(0.0f, _S344.vals_1[int(4)]);
    *(&(&_S343)->vals_1[int(5)]) = s_primal_ctx_max_0(0.0f, _S344.vals_1[int(5)]);
    *(&(&_S343)->vals_1[int(6)]) = s_primal_ctx_max_0(0.0f, _S344.vals_1[int(6)]);
    *(&(&_S343)->vals_1[int(7)]) = s_primal_ctx_max_0(0.0f, _S344.vals_1[int(7)]);
    *(&(&_S343)->vals_1[int(8)]) = s_primal_ctx_max_0(0.0f, _S344.vals_1[int(8)]);
    *(&(&_S343)->vals_1[int(9)]) = s_primal_ctx_max_0(0.0f, _S344.vals_1[int(9)]);
    *(&(&_S343)->vals_1[int(10)]) = s_primal_ctx_max_0(0.0f, _S344.vals_1[int(10)]);
    *(&(&_S343)->vals_1[int(11)]) = s_primal_ctx_max_0(0.0f, _S344.vals_1[int(11)]);
    *(&(&_S343)->vals_1[int(12)]) = s_primal_ctx_max_0(0.0f, _S344.vals_1[int(12)]);
    *(&(&_S343)->vals_1[int(13)]) = s_primal_ctx_max_0(0.0f, _S344.vals_1[int(13)]);
    *(&(&_S343)->vals_1[int(14)]) = s_primal_ctx_max_0(0.0f, _S344.vals_1[int(14)]);
    *(&(&_S343)->vals_1[int(15)]) = s_primal_ctx_max_0(0.0f, _S344.vals_1[int(15)]);
    Feature_0 _S346 = s_primal_ctx_Linear_eval_0(this_29.layers_0[int(2)], _S343);
    _s_diff_ctx_1->_S317 = _S346;
    float _S347 = s_primal_ctx_max_0(0.0f, _S346.vals_1[int(0)]);
    _S343 = _S346;
    *(&(&_S343)->vals_1[int(0)]) = _S347;
    *(&(&_S343)->vals_1[int(1)]) = s_primal_ctx_max_0(0.0f, _S346.vals_1[int(1)]);
    *(&(&_S343)->vals_1[int(2)]) = s_primal_ctx_max_0(0.0f, _S346.vals_1[int(2)]);
    *(&(&_S343)->vals_1[int(3)]) = s_primal_ctx_max_0(0.0f, _S346.vals_1[int(3)]);
    *(&(&_S343)->vals_1[int(4)]) = s_primal_ctx_max_0(0.0f, _S346.vals_1[int(4)]);
    *(&(&_S343)->vals_1[int(5)]) = s_primal_ctx_max_0(0.0f, _S346.vals_1[int(5)]);
    *(&(&_S343)->vals_1[int(6)]) = s_primal_ctx_max_0(0.0f, _S346.vals_1[int(6)]);
    *(&(&_S343)->vals_1[int(7)]) = s_primal_ctx_max_0(0.0f, _S346.vals_1[int(7)]);
    *(&(&_S343)->vals_1[int(8)]) = s_primal_ctx_max_0(0.0f, _S346.vals_1[int(8)]);
    *(&(&_S343)->vals_1[int(9)]) = s_primal_ctx_max_0(0.0f, _S346.vals_1[int(9)]);
    *(&(&_S343)->vals_1[int(10)]) = s_primal_ctx_max_0(0.0f, _S346.vals_1[int(10)]);
    *(&(&_S343)->vals_1[int(11)]) = s_primal_ctx_max_0(0.0f, _S346.vals_1[int(11)]);
    *(&(&_S343)->vals_1[int(12)]) = s_primal_ctx_max_0(0.0f, _S346.vals_1[int(12)]);
    *(&(&_S343)->vals_1[int(13)]) = s_primal_ctx_max_0(0.0f, _S346.vals_1[int(13)]);
    *(&(&_S343)->vals_1[int(14)]) = s_primal_ctx_max_0(0.0f, _S346.vals_1[int(14)]);
    *(&(&_S343)->vals_1[int(15)]) = s_primal_ctx_max_0(0.0f, _S346.vals_1[int(15)]);
    return _S343;
}

__device__ void s_primal_ctx_DiffTensorView_storeOnce_0(DiffTensorView_0 _S348, uint4  _S349, float _S350)
{
    DiffTensorView_storeOnce_0(_S348, _S349, _S350);
    return;
}

__device__ void s_bwd_primal_renderImage_0(MLP_0 mlp_0, DiffTensorView_0 featureGrid_0, DiffTensorView_0 imageOutput_0, s_bwd_prop_renderImage_Intermediates_0 * _s_diff_ctx_2)
{
    s_bwd_prop_getInFeature_Intermediates_0 _S351 = { int(0) };
    FixedArray<float, 16>  _S352 = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    Feature_0 _S353 = { _S352 };
    s_bwd_prop_MLP_eval_Intermediates_0 _S354 = { _S353, _S353, _S353 };
    _s_diff_ctx_2->_S318 = _S351;
    _s_diff_ctx_2->_S319 = _S353;
    _s_diff_ctx_2->_S320 = _S354;
    _s_diff_ctx_2->_S321 = _S353;
    _s_diff_ctx_2->_S322 = int(0);
    _s_diff_ctx_2->_S323 = 0U;
    _s_diff_ctx_2->_S324 = 0U;
    _s_diff_ctx_2->_S325 = 0U;
    (&_s_diff_ctx_2->_S319)->vals_1 = _S352;
    (&_s_diff_ctx_2->_S321)->vals_1 = _S352;
    _s_diff_ctx_2->_S322 = int(0);
    uint3  dispatchIdx_0 = ((threadIdx)) + ((blockIdx)) * ((blockDim));
    uint _S355 = dispatchIdx_0.x;
    uint _S356 = DiffTensorView_size_0(imageOutput_0, 0U);
    _s_diff_ctx_2->_S323 = _S356;
    bool _S357 = _S355 >= _S356;
    uint _S358 = dispatchIdx_0.y;
    uint _S359 = DiffTensorView_size_0(imageOutput_0, 1U);
    _s_diff_ctx_2->_S324 = _S359;
    bool _S360 = _S357 || _S358 >= _S359;
    uint _S361 = dispatchIdx_0.z;
    uint _S362 = DiffTensorView_size_0(imageOutput_0, 2U);
    _s_diff_ctx_2->_S325 = _S362;
    if(!(_S360 || _S361 >= _S362))
    {
        Feature_0 _S363 = s_bwd_primal_getInFeature_0(featureGrid_0, make_uint3 (_S355, _S358, _S361), &_s_diff_ctx_2->_S318);
        _s_diff_ctx_2->_S319 = _S363;
        Feature_0 _S364 = s_bwd_primal_MLP_eval_0(mlp_0, _S363, &_s_diff_ctx_2->_S320);
        _s_diff_ctx_2->_S321 = _S364;
        bool _bflag_1 = true;
        int i_10 = int(0);
        int _pc_1 = int(0);
        for(;;)
        {
            _s_diff_ctx_2->_S322 = _pc_1;
            if(_bflag_1)
            {
            }
            else
            {
                break;
            }
            s_primal_ctx_DiffTensorView_storeOnce_0(imageOutput_0, make_uint4 (_S355, _S358, _S361, uint(i_10)), _S364.vals_1[i_10]);
            int i_11 = i_10 + int(1);
            if(i_11 < int(4))
            {
            }
            else
            {
                _bflag_1 = false;
            }
            if(_bflag_1)
            {
                i_10 = i_11;
            }
            _pc_1 = _pc_1 + int(1);
        }
    }
    return;
}

__device__ void s_bwd_prop_Linear_eval_0(Linear_0 _S365, DiffPair_Feature_0 * _S366, s_diff_Feature_0 _S367, uint _S368)
{
    Linear_eval_bwd_0(_S365, _S366, _S367, _S368);
    return;
}

__device__ void s_bwd_prop_MLP_eval_0(MLP_0 this_30, DiffPair_Feature_0 * dpin_feature_1, s_diff_Feature_0 _s_dOut_0, s_bwd_prop_MLP_eval_Intermediates_0 _s_diff_ctx_3, uint _S369)
{
    float _S370 = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S315.vals_1[int(0)]);
    Feature_0 _S371 = _s_diff_ctx_3._S315;
    *(&(&_S371)->vals_1[int(0)]) = _S370;
    *(&(&_S371)->vals_1[int(1)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S315.vals_1[int(1)]);
    *(&(&_S371)->vals_1[int(2)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S315.vals_1[int(2)]);
    *(&(&_S371)->vals_1[int(3)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S315.vals_1[int(3)]);
    *(&(&_S371)->vals_1[int(4)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S315.vals_1[int(4)]);
    *(&(&_S371)->vals_1[int(5)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S315.vals_1[int(5)]);
    *(&(&_S371)->vals_1[int(6)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S315.vals_1[int(6)]);
    *(&(&_S371)->vals_1[int(7)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S315.vals_1[int(7)]);
    *(&(&_S371)->vals_1[int(8)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S315.vals_1[int(8)]);
    *(&(&_S371)->vals_1[int(9)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S315.vals_1[int(9)]);
    *(&(&_S371)->vals_1[int(10)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S315.vals_1[int(10)]);
    *(&(&_S371)->vals_1[int(11)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S315.vals_1[int(11)]);
    *(&(&_S371)->vals_1[int(12)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S315.vals_1[int(12)]);
    *(&(&_S371)->vals_1[int(13)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S315.vals_1[int(13)]);
    *(&(&_S371)->vals_1[int(14)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S315.vals_1[int(14)]);
    *(&(&_S371)->vals_1[int(15)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S315.vals_1[int(15)]);
    float _S372 = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S316.vals_1[int(0)]);
    Feature_0 _S373 = _s_diff_ctx_3._S316;
    *(&(&_S373)->vals_1[int(0)]) = _S372;
    *(&(&_S373)->vals_1[int(1)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S316.vals_1[int(1)]);
    *(&(&_S373)->vals_1[int(2)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S316.vals_1[int(2)]);
    *(&(&_S373)->vals_1[int(3)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S316.vals_1[int(3)]);
    *(&(&_S373)->vals_1[int(4)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S316.vals_1[int(4)]);
    *(&(&_S373)->vals_1[int(5)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S316.vals_1[int(5)]);
    *(&(&_S373)->vals_1[int(6)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S316.vals_1[int(6)]);
    *(&(&_S373)->vals_1[int(7)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S316.vals_1[int(7)]);
    *(&(&_S373)->vals_1[int(8)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S316.vals_1[int(8)]);
    *(&(&_S373)->vals_1[int(9)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S316.vals_1[int(9)]);
    *(&(&_S373)->vals_1[int(10)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S316.vals_1[int(10)]);
    *(&(&_S373)->vals_1[int(11)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S316.vals_1[int(11)]);
    *(&(&_S373)->vals_1[int(12)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S316.vals_1[int(12)]);
    *(&(&_S373)->vals_1[int(13)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S316.vals_1[int(13)]);
    *(&(&_S373)->vals_1[int(14)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S316.vals_1[int(14)]);
    *(&(&_S373)->vals_1[int(15)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S316.vals_1[int(15)]);
    s_diff_Feature_0 _S374 = _s_dOut_0;
    *(&(&_S374)->vals_0[int(15)]) = 0.0f;
    DiffPair_float_0 _S375;
    (&_S375)->primal_1 = 0.0f;
    (&_S375)->differential_0 = 0.0f;
    DiffPair_float_0 _S376;
    (&_S376)->primal_1 = _s_diff_ctx_3._S317.vals_1[int(15)];
    (&_S376)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S375, &_S376, _s_dOut_0.vals_0[int(15)]);
    *(&(&_S374)->vals_0[int(14)]) = 0.0f;
    DiffPair_float_0 _S377;
    (&_S377)->primal_1 = 0.0f;
    (&_S377)->differential_0 = 0.0f;
    DiffPair_float_0 _S378;
    (&_S378)->primal_1 = _s_diff_ctx_3._S317.vals_1[int(14)];
    (&_S378)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S377, &_S378, _s_dOut_0.vals_0[int(14)]);
    *(&(&_S374)->vals_0[int(13)]) = 0.0f;
    DiffPair_float_0 _S379;
    (&_S379)->primal_1 = 0.0f;
    (&_S379)->differential_0 = 0.0f;
    DiffPair_float_0 _S380;
    (&_S380)->primal_1 = _s_diff_ctx_3._S317.vals_1[int(13)];
    (&_S380)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S379, &_S380, _s_dOut_0.vals_0[int(13)]);
    *(&(&_S374)->vals_0[int(12)]) = 0.0f;
    DiffPair_float_0 _S381;
    (&_S381)->primal_1 = 0.0f;
    (&_S381)->differential_0 = 0.0f;
    DiffPair_float_0 _S382;
    (&_S382)->primal_1 = _s_diff_ctx_3._S317.vals_1[int(12)];
    (&_S382)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S381, &_S382, _s_dOut_0.vals_0[int(12)]);
    *(&(&_S374)->vals_0[int(11)]) = 0.0f;
    DiffPair_float_0 _S383;
    (&_S383)->primal_1 = 0.0f;
    (&_S383)->differential_0 = 0.0f;
    DiffPair_float_0 _S384;
    (&_S384)->primal_1 = _s_diff_ctx_3._S317.vals_1[int(11)];
    (&_S384)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S383, &_S384, _s_dOut_0.vals_0[int(11)]);
    *(&(&_S374)->vals_0[int(10)]) = 0.0f;
    DiffPair_float_0 _S385;
    (&_S385)->primal_1 = 0.0f;
    (&_S385)->differential_0 = 0.0f;
    DiffPair_float_0 _S386;
    (&_S386)->primal_1 = _s_diff_ctx_3._S317.vals_1[int(10)];
    (&_S386)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S385, &_S386, _s_dOut_0.vals_0[int(10)]);
    *(&(&_S374)->vals_0[int(9)]) = 0.0f;
    DiffPair_float_0 _S387;
    (&_S387)->primal_1 = 0.0f;
    (&_S387)->differential_0 = 0.0f;
    DiffPair_float_0 _S388;
    (&_S388)->primal_1 = _s_diff_ctx_3._S317.vals_1[int(9)];
    (&_S388)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S387, &_S388, _s_dOut_0.vals_0[int(9)]);
    *(&(&_S374)->vals_0[int(8)]) = 0.0f;
    DiffPair_float_0 _S389;
    (&_S389)->primal_1 = 0.0f;
    (&_S389)->differential_0 = 0.0f;
    DiffPair_float_0 _S390;
    (&_S390)->primal_1 = _s_diff_ctx_3._S317.vals_1[int(8)];
    (&_S390)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S389, &_S390, _s_dOut_0.vals_0[int(8)]);
    *(&(&_S374)->vals_0[int(7)]) = 0.0f;
    DiffPair_float_0 _S391;
    (&_S391)->primal_1 = 0.0f;
    (&_S391)->differential_0 = 0.0f;
    DiffPair_float_0 _S392;
    (&_S392)->primal_1 = _s_diff_ctx_3._S317.vals_1[int(7)];
    (&_S392)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S391, &_S392, _s_dOut_0.vals_0[int(7)]);
    *(&(&_S374)->vals_0[int(6)]) = 0.0f;
    DiffPair_float_0 _S393;
    (&_S393)->primal_1 = 0.0f;
    (&_S393)->differential_0 = 0.0f;
    DiffPair_float_0 _S394;
    (&_S394)->primal_1 = _s_diff_ctx_3._S317.vals_1[int(6)];
    (&_S394)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S393, &_S394, _s_dOut_0.vals_0[int(6)]);
    *(&(&_S374)->vals_0[int(5)]) = 0.0f;
    DiffPair_float_0 _S395;
    (&_S395)->primal_1 = 0.0f;
    (&_S395)->differential_0 = 0.0f;
    DiffPair_float_0 _S396;
    (&_S396)->primal_1 = _s_diff_ctx_3._S317.vals_1[int(5)];
    (&_S396)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S395, &_S396, _s_dOut_0.vals_0[int(5)]);
    *(&(&_S374)->vals_0[int(4)]) = 0.0f;
    DiffPair_float_0 _S397;
    (&_S397)->primal_1 = 0.0f;
    (&_S397)->differential_0 = 0.0f;
    DiffPair_float_0 _S398;
    (&_S398)->primal_1 = _s_diff_ctx_3._S317.vals_1[int(4)];
    (&_S398)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S397, &_S398, _s_dOut_0.vals_0[int(4)]);
    *(&(&_S374)->vals_0[int(3)]) = 0.0f;
    DiffPair_float_0 _S399;
    (&_S399)->primal_1 = 0.0f;
    (&_S399)->differential_0 = 0.0f;
    DiffPair_float_0 _S400;
    (&_S400)->primal_1 = _s_diff_ctx_3._S317.vals_1[int(3)];
    (&_S400)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S399, &_S400, _s_dOut_0.vals_0[int(3)]);
    *(&(&_S374)->vals_0[int(2)]) = 0.0f;
    DiffPair_float_0 _S401;
    (&_S401)->primal_1 = 0.0f;
    (&_S401)->differential_0 = 0.0f;
    DiffPair_float_0 _S402;
    (&_S402)->primal_1 = _s_diff_ctx_3._S317.vals_1[int(2)];
    (&_S402)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S401, &_S402, _s_dOut_0.vals_0[int(2)]);
    *(&(&_S374)->vals_0[int(1)]) = 0.0f;
    DiffPair_float_0 _S403;
    (&_S403)->primal_1 = 0.0f;
    (&_S403)->differential_0 = 0.0f;
    DiffPair_float_0 _S404;
    (&_S404)->primal_1 = _s_diff_ctx_3._S317.vals_1[int(1)];
    (&_S404)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S403, &_S404, _s_dOut_0.vals_0[int(1)]);
    *(&(&_S374)->vals_0[int(0)]) = 0.0f;
    DiffPair_float_0 _S405;
    (&_S405)->primal_1 = 0.0f;
    (&_S405)->differential_0 = 0.0f;
    DiffPair_float_0 _S406;
    (&_S406)->primal_1 = _s_diff_ctx_3._S317.vals_1[int(0)];
    (&_S406)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S405, &_S406, _s_dOut_0.vals_0[int(0)]);
    FixedArray<float, 16>  _S407;
    *(&_S407[int(0)]) = 0.0f;
    *(&_S407[int(1)]) = 0.0f;
    *(&_S407[int(2)]) = 0.0f;
    *(&_S407[int(3)]) = 0.0f;
    *(&_S407[int(4)]) = 0.0f;
    *(&_S407[int(5)]) = 0.0f;
    *(&_S407[int(6)]) = 0.0f;
    *(&_S407[int(7)]) = 0.0f;
    *(&_S407[int(8)]) = 0.0f;
    *(&_S407[int(9)]) = 0.0f;
    *(&_S407[int(10)]) = 0.0f;
    *(&_S407[int(11)]) = 0.0f;
    *(&_S407[int(12)]) = 0.0f;
    *(&_S407[int(13)]) = 0.0f;
    *(&_S407[int(14)]) = 0.0f;
    *(&_S407[int(15)]) = 0.0f;
    *(&_S407[int(15)]) = _S376.differential_0;
    *(&_S407[int(14)]) = _S378.differential_0;
    *(&_S407[int(13)]) = _S380.differential_0;
    *(&_S407[int(12)]) = _S382.differential_0;
    *(&_S407[int(11)]) = _S384.differential_0;
    *(&_S407[int(10)]) = _S386.differential_0;
    *(&_S407[int(9)]) = _S388.differential_0;
    *(&_S407[int(8)]) = _S390.differential_0;
    *(&_S407[int(7)]) = _S392.differential_0;
    *(&_S407[int(6)]) = _S394.differential_0;
    *(&_S407[int(5)]) = _S396.differential_0;
    *(&_S407[int(4)]) = _S398.differential_0;
    *(&_S407[int(3)]) = _S400.differential_0;
    *(&_S407[int(2)]) = _S402.differential_0;
    *(&_S407[int(1)]) = _S404.differential_0;
    *(&_S407[int(0)]) = _S406.differential_0;
    s_diff_Feature_0 _S408 = Feature_x24_syn_dzero_0();
    s_diff_Feature_0 _S409 = _S408;
    (&_S409)->vals_0 = _S407;
    s_diff_Feature_0 _S410 = Feature_x24_syn_dadd_0(_S374, _S409);
    DiffPair_Feature_0 _S411;
    (&_S411)->primal_1 = _S373;
    (&_S411)->differential_0 = _S408;
    s_bwd_prop_Linear_eval_0(this_30.layers_0[int(2)], &_S411, _S410, _S369);
    _S374 = _S411.differential_0;
    *(&(&_S374)->vals_0[int(15)]) = 0.0f;
    DiffPair_float_0 _S412;
    (&_S412)->primal_1 = 0.0f;
    (&_S412)->differential_0 = 0.0f;
    DiffPair_float_0 _S413;
    (&_S413)->primal_1 = _s_diff_ctx_3._S316.vals_1[int(15)];
    (&_S413)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S412, &_S413, _S411.differential_0.vals_0[int(15)]);
    *(&(&_S374)->vals_0[int(14)]) = 0.0f;
    DiffPair_float_0 _S414;
    (&_S414)->primal_1 = 0.0f;
    (&_S414)->differential_0 = 0.0f;
    DiffPair_float_0 _S415;
    (&_S415)->primal_1 = _s_diff_ctx_3._S316.vals_1[int(14)];
    (&_S415)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S414, &_S415, _S411.differential_0.vals_0[int(14)]);
    *(&(&_S374)->vals_0[int(13)]) = 0.0f;
    DiffPair_float_0 _S416;
    (&_S416)->primal_1 = 0.0f;
    (&_S416)->differential_0 = 0.0f;
    DiffPair_float_0 _S417;
    (&_S417)->primal_1 = _s_diff_ctx_3._S316.vals_1[int(13)];
    (&_S417)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S416, &_S417, _S411.differential_0.vals_0[int(13)]);
    *(&(&_S374)->vals_0[int(12)]) = 0.0f;
    DiffPair_float_0 _S418;
    (&_S418)->primal_1 = 0.0f;
    (&_S418)->differential_0 = 0.0f;
    DiffPair_float_0 _S419;
    (&_S419)->primal_1 = _s_diff_ctx_3._S316.vals_1[int(12)];
    (&_S419)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S418, &_S419, _S411.differential_0.vals_0[int(12)]);
    *(&(&_S374)->vals_0[int(11)]) = 0.0f;
    DiffPair_float_0 _S420;
    (&_S420)->primal_1 = 0.0f;
    (&_S420)->differential_0 = 0.0f;
    DiffPair_float_0 _S421;
    (&_S421)->primal_1 = _s_diff_ctx_3._S316.vals_1[int(11)];
    (&_S421)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S420, &_S421, _S411.differential_0.vals_0[int(11)]);
    *(&(&_S374)->vals_0[int(10)]) = 0.0f;
    DiffPair_float_0 _S422;
    (&_S422)->primal_1 = 0.0f;
    (&_S422)->differential_0 = 0.0f;
    DiffPair_float_0 _S423;
    (&_S423)->primal_1 = _s_diff_ctx_3._S316.vals_1[int(10)];
    (&_S423)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S422, &_S423, _S411.differential_0.vals_0[int(10)]);
    *(&(&_S374)->vals_0[int(9)]) = 0.0f;
    DiffPair_float_0 _S424;
    (&_S424)->primal_1 = 0.0f;
    (&_S424)->differential_0 = 0.0f;
    DiffPair_float_0 _S425;
    (&_S425)->primal_1 = _s_diff_ctx_3._S316.vals_1[int(9)];
    (&_S425)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S424, &_S425, _S411.differential_0.vals_0[int(9)]);
    *(&(&_S374)->vals_0[int(8)]) = 0.0f;
    DiffPair_float_0 _S426;
    (&_S426)->primal_1 = 0.0f;
    (&_S426)->differential_0 = 0.0f;
    DiffPair_float_0 _S427;
    (&_S427)->primal_1 = _s_diff_ctx_3._S316.vals_1[int(8)];
    (&_S427)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S426, &_S427, _S411.differential_0.vals_0[int(8)]);
    *(&(&_S374)->vals_0[int(7)]) = 0.0f;
    DiffPair_float_0 _S428;
    (&_S428)->primal_1 = 0.0f;
    (&_S428)->differential_0 = 0.0f;
    DiffPair_float_0 _S429;
    (&_S429)->primal_1 = _s_diff_ctx_3._S316.vals_1[int(7)];
    (&_S429)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S428, &_S429, _S411.differential_0.vals_0[int(7)]);
    *(&(&_S374)->vals_0[int(6)]) = 0.0f;
    DiffPair_float_0 _S430;
    (&_S430)->primal_1 = 0.0f;
    (&_S430)->differential_0 = 0.0f;
    DiffPair_float_0 _S431;
    (&_S431)->primal_1 = _s_diff_ctx_3._S316.vals_1[int(6)];
    (&_S431)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S430, &_S431, _S411.differential_0.vals_0[int(6)]);
    *(&(&_S374)->vals_0[int(5)]) = 0.0f;
    DiffPair_float_0 _S432;
    (&_S432)->primal_1 = 0.0f;
    (&_S432)->differential_0 = 0.0f;
    DiffPair_float_0 _S433;
    (&_S433)->primal_1 = _s_diff_ctx_3._S316.vals_1[int(5)];
    (&_S433)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S432, &_S433, _S411.differential_0.vals_0[int(5)]);
    *(&(&_S374)->vals_0[int(4)]) = 0.0f;
    DiffPair_float_0 _S434;
    (&_S434)->primal_1 = 0.0f;
    (&_S434)->differential_0 = 0.0f;
    DiffPair_float_0 _S435;
    (&_S435)->primal_1 = _s_diff_ctx_3._S316.vals_1[int(4)];
    (&_S435)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S434, &_S435, _S411.differential_0.vals_0[int(4)]);
    *(&(&_S374)->vals_0[int(3)]) = 0.0f;
    DiffPair_float_0 _S436;
    (&_S436)->primal_1 = 0.0f;
    (&_S436)->differential_0 = 0.0f;
    DiffPair_float_0 _S437;
    (&_S437)->primal_1 = _s_diff_ctx_3._S316.vals_1[int(3)];
    (&_S437)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S436, &_S437, _S411.differential_0.vals_0[int(3)]);
    *(&(&_S374)->vals_0[int(2)]) = 0.0f;
    DiffPair_float_0 _S438;
    (&_S438)->primal_1 = 0.0f;
    (&_S438)->differential_0 = 0.0f;
    DiffPair_float_0 _S439;
    (&_S439)->primal_1 = _s_diff_ctx_3._S316.vals_1[int(2)];
    (&_S439)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S438, &_S439, _S411.differential_0.vals_0[int(2)]);
    *(&(&_S374)->vals_0[int(1)]) = 0.0f;
    DiffPair_float_0 _S440;
    (&_S440)->primal_1 = 0.0f;
    (&_S440)->differential_0 = 0.0f;
    DiffPair_float_0 _S441;
    (&_S441)->primal_1 = _s_diff_ctx_3._S316.vals_1[int(1)];
    (&_S441)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S440, &_S441, _S411.differential_0.vals_0[int(1)]);
    *(&(&_S374)->vals_0[int(0)]) = 0.0f;
    DiffPair_float_0 _S442;
    (&_S442)->primal_1 = 0.0f;
    (&_S442)->differential_0 = 0.0f;
    DiffPair_float_0 _S443;
    (&_S443)->primal_1 = _s_diff_ctx_3._S316.vals_1[int(0)];
    (&_S443)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S442, &_S443, _S411.differential_0.vals_0[int(0)]);
    FixedArray<float, 16>  _S444;
    *(&_S444[int(0)]) = 0.0f;
    *(&_S444[int(1)]) = 0.0f;
    *(&_S444[int(2)]) = 0.0f;
    *(&_S444[int(3)]) = 0.0f;
    *(&_S444[int(4)]) = 0.0f;
    *(&_S444[int(5)]) = 0.0f;
    *(&_S444[int(6)]) = 0.0f;
    *(&_S444[int(7)]) = 0.0f;
    *(&_S444[int(8)]) = 0.0f;
    *(&_S444[int(9)]) = 0.0f;
    *(&_S444[int(10)]) = 0.0f;
    *(&_S444[int(11)]) = 0.0f;
    *(&_S444[int(12)]) = 0.0f;
    *(&_S444[int(13)]) = 0.0f;
    *(&_S444[int(14)]) = 0.0f;
    *(&_S444[int(15)]) = 0.0f;
    *(&_S444[int(15)]) = _S413.differential_0;
    *(&_S444[int(14)]) = _S415.differential_0;
    *(&_S444[int(13)]) = _S417.differential_0;
    *(&_S444[int(12)]) = _S419.differential_0;
    *(&_S444[int(11)]) = _S421.differential_0;
    *(&_S444[int(10)]) = _S423.differential_0;
    *(&_S444[int(9)]) = _S425.differential_0;
    *(&_S444[int(8)]) = _S427.differential_0;
    *(&_S444[int(7)]) = _S429.differential_0;
    *(&_S444[int(6)]) = _S431.differential_0;
    *(&_S444[int(5)]) = _S433.differential_0;
    *(&_S444[int(4)]) = _S435.differential_0;
    *(&_S444[int(3)]) = _S437.differential_0;
    *(&_S444[int(2)]) = _S439.differential_0;
    *(&_S444[int(1)]) = _S441.differential_0;
    *(&_S444[int(0)]) = _S443.differential_0;
    s_diff_Feature_0 _S445 = _S408;
    (&_S445)->vals_0 = _S444;
    s_diff_Feature_0 _S446 = Feature_x24_syn_dadd_0(_S374, _S445);
    DiffPair_Feature_0 _S447;
    (&_S447)->primal_1 = _S371;
    (&_S447)->differential_0 = _S408;
    s_bwd_prop_Linear_eval_0(this_30.layers_0[int(1)], &_S447, _S446, _S369);
    _S374 = _S447.differential_0;
    *(&(&_S374)->vals_0[int(15)]) = 0.0f;
    DiffPair_float_0 _S448;
    (&_S448)->primal_1 = 0.0f;
    (&_S448)->differential_0 = 0.0f;
    DiffPair_float_0 _S449;
    (&_S449)->primal_1 = _s_diff_ctx_3._S315.vals_1[int(15)];
    (&_S449)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S448, &_S449, _S447.differential_0.vals_0[int(15)]);
    *(&(&_S374)->vals_0[int(14)]) = 0.0f;
    DiffPair_float_0 _S450;
    (&_S450)->primal_1 = 0.0f;
    (&_S450)->differential_0 = 0.0f;
    DiffPair_float_0 _S451;
    (&_S451)->primal_1 = _s_diff_ctx_3._S315.vals_1[int(14)];
    (&_S451)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S450, &_S451, _S447.differential_0.vals_0[int(14)]);
    *(&(&_S374)->vals_0[int(13)]) = 0.0f;
    DiffPair_float_0 _S452;
    (&_S452)->primal_1 = 0.0f;
    (&_S452)->differential_0 = 0.0f;
    DiffPair_float_0 _S453;
    (&_S453)->primal_1 = _s_diff_ctx_3._S315.vals_1[int(13)];
    (&_S453)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S452, &_S453, _S447.differential_0.vals_0[int(13)]);
    *(&(&_S374)->vals_0[int(12)]) = 0.0f;
    DiffPair_float_0 _S454;
    (&_S454)->primal_1 = 0.0f;
    (&_S454)->differential_0 = 0.0f;
    DiffPair_float_0 _S455;
    (&_S455)->primal_1 = _s_diff_ctx_3._S315.vals_1[int(12)];
    (&_S455)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S454, &_S455, _S447.differential_0.vals_0[int(12)]);
    *(&(&_S374)->vals_0[int(11)]) = 0.0f;
    DiffPair_float_0 _S456;
    (&_S456)->primal_1 = 0.0f;
    (&_S456)->differential_0 = 0.0f;
    DiffPair_float_0 _S457;
    (&_S457)->primal_1 = _s_diff_ctx_3._S315.vals_1[int(11)];
    (&_S457)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S456, &_S457, _S447.differential_0.vals_0[int(11)]);
    *(&(&_S374)->vals_0[int(10)]) = 0.0f;
    DiffPair_float_0 _S458;
    (&_S458)->primal_1 = 0.0f;
    (&_S458)->differential_0 = 0.0f;
    DiffPair_float_0 _S459;
    (&_S459)->primal_1 = _s_diff_ctx_3._S315.vals_1[int(10)];
    (&_S459)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S458, &_S459, _S447.differential_0.vals_0[int(10)]);
    *(&(&_S374)->vals_0[int(9)]) = 0.0f;
    DiffPair_float_0 _S460;
    (&_S460)->primal_1 = 0.0f;
    (&_S460)->differential_0 = 0.0f;
    DiffPair_float_0 _S461;
    (&_S461)->primal_1 = _s_diff_ctx_3._S315.vals_1[int(9)];
    (&_S461)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S460, &_S461, _S447.differential_0.vals_0[int(9)]);
    *(&(&_S374)->vals_0[int(8)]) = 0.0f;
    DiffPair_float_0 _S462;
    (&_S462)->primal_1 = 0.0f;
    (&_S462)->differential_0 = 0.0f;
    DiffPair_float_0 _S463;
    (&_S463)->primal_1 = _s_diff_ctx_3._S315.vals_1[int(8)];
    (&_S463)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S462, &_S463, _S447.differential_0.vals_0[int(8)]);
    *(&(&_S374)->vals_0[int(7)]) = 0.0f;
    DiffPair_float_0 _S464;
    (&_S464)->primal_1 = 0.0f;
    (&_S464)->differential_0 = 0.0f;
    DiffPair_float_0 _S465;
    (&_S465)->primal_1 = _s_diff_ctx_3._S315.vals_1[int(7)];
    (&_S465)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S464, &_S465, _S447.differential_0.vals_0[int(7)]);
    *(&(&_S374)->vals_0[int(6)]) = 0.0f;
    DiffPair_float_0 _S466;
    (&_S466)->primal_1 = 0.0f;
    (&_S466)->differential_0 = 0.0f;
    DiffPair_float_0 _S467;
    (&_S467)->primal_1 = _s_diff_ctx_3._S315.vals_1[int(6)];
    (&_S467)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S466, &_S467, _S447.differential_0.vals_0[int(6)]);
    *(&(&_S374)->vals_0[int(5)]) = 0.0f;
    DiffPair_float_0 _S468;
    (&_S468)->primal_1 = 0.0f;
    (&_S468)->differential_0 = 0.0f;
    DiffPair_float_0 _S469;
    (&_S469)->primal_1 = _s_diff_ctx_3._S315.vals_1[int(5)];
    (&_S469)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S468, &_S469, _S447.differential_0.vals_0[int(5)]);
    *(&(&_S374)->vals_0[int(4)]) = 0.0f;
    DiffPair_float_0 _S470;
    (&_S470)->primal_1 = 0.0f;
    (&_S470)->differential_0 = 0.0f;
    DiffPair_float_0 _S471;
    (&_S471)->primal_1 = _s_diff_ctx_3._S315.vals_1[int(4)];
    (&_S471)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S470, &_S471, _S447.differential_0.vals_0[int(4)]);
    *(&(&_S374)->vals_0[int(3)]) = 0.0f;
    DiffPair_float_0 _S472;
    (&_S472)->primal_1 = 0.0f;
    (&_S472)->differential_0 = 0.0f;
    DiffPair_float_0 _S473;
    (&_S473)->primal_1 = _s_diff_ctx_3._S315.vals_1[int(3)];
    (&_S473)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S472, &_S473, _S447.differential_0.vals_0[int(3)]);
    *(&(&_S374)->vals_0[int(2)]) = 0.0f;
    DiffPair_float_0 _S474;
    (&_S474)->primal_1 = 0.0f;
    (&_S474)->differential_0 = 0.0f;
    DiffPair_float_0 _S475;
    (&_S475)->primal_1 = _s_diff_ctx_3._S315.vals_1[int(2)];
    (&_S475)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S474, &_S475, _S447.differential_0.vals_0[int(2)]);
    *(&(&_S374)->vals_0[int(1)]) = 0.0f;
    DiffPair_float_0 _S476;
    (&_S476)->primal_1 = 0.0f;
    (&_S476)->differential_0 = 0.0f;
    DiffPair_float_0 _S477;
    (&_S477)->primal_1 = _s_diff_ctx_3._S315.vals_1[int(1)];
    (&_S477)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S476, &_S477, _S447.differential_0.vals_0[int(1)]);
    *(&(&_S374)->vals_0[int(0)]) = 0.0f;
    DiffPair_float_0 _S478;
    (&_S478)->primal_1 = 0.0f;
    (&_S478)->differential_0 = 0.0f;
    DiffPair_float_0 _S479;
    (&_S479)->primal_1 = _s_diff_ctx_3._S315.vals_1[int(0)];
    (&_S479)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S478, &_S479, _S447.differential_0.vals_0[int(0)]);
    FixedArray<float, 16>  _S480;
    *(&_S480[int(0)]) = 0.0f;
    *(&_S480[int(1)]) = 0.0f;
    *(&_S480[int(2)]) = 0.0f;
    *(&_S480[int(3)]) = 0.0f;
    *(&_S480[int(4)]) = 0.0f;
    *(&_S480[int(5)]) = 0.0f;
    *(&_S480[int(6)]) = 0.0f;
    *(&_S480[int(7)]) = 0.0f;
    *(&_S480[int(8)]) = 0.0f;
    *(&_S480[int(9)]) = 0.0f;
    *(&_S480[int(10)]) = 0.0f;
    *(&_S480[int(11)]) = 0.0f;
    *(&_S480[int(12)]) = 0.0f;
    *(&_S480[int(13)]) = 0.0f;
    *(&_S480[int(14)]) = 0.0f;
    *(&_S480[int(15)]) = 0.0f;
    *(&_S480[int(15)]) = _S449.differential_0;
    *(&_S480[int(14)]) = _S451.differential_0;
    *(&_S480[int(13)]) = _S453.differential_0;
    *(&_S480[int(12)]) = _S455.differential_0;
    *(&_S480[int(11)]) = _S457.differential_0;
    *(&_S480[int(10)]) = _S459.differential_0;
    *(&_S480[int(9)]) = _S461.differential_0;
    *(&_S480[int(8)]) = _S463.differential_0;
    *(&_S480[int(7)]) = _S465.differential_0;
    *(&_S480[int(6)]) = _S467.differential_0;
    *(&_S480[int(5)]) = _S469.differential_0;
    *(&_S480[int(4)]) = _S471.differential_0;
    *(&_S480[int(3)]) = _S473.differential_0;
    *(&_S480[int(2)]) = _S475.differential_0;
    *(&_S480[int(1)]) = _S477.differential_0;
    *(&_S480[int(0)]) = _S479.differential_0;
    s_diff_Feature_0 _S481 = _S408;
    (&_S481)->vals_0 = _S480;
    s_diff_Feature_0 _S482 = Feature_x24_syn_dadd_0(_S374, _S481);
    DiffPair_Feature_0 _S483;
    (&_S483)->primal_1 = (*dpin_feature_1).primal_1;
    (&_S483)->differential_0 = _S408;
    s_bwd_prop_Linear_eval_0(this_30.layers_0[int(0)], &_S483, _S482, _S369);
    dpin_feature_1->primal_1 = (*dpin_feature_1).primal_1;
    dpin_feature_1->differential_0 = _S483.differential_0;
    return;
}

__device__ void s_bwd_prop_DiffTensorView_load_0(DiffTensorView_0 _S484, uint4  _S485, float _S486)
{
    DiffTensorView_load_backward_0(_S484, _S485, _S486);
    return;
}

__device__ void s_bwd_prop_getInFeature_0(DiffTensorView_0 input_9, uint3  idx_2, s_diff_Feature_0 _s_dOut_1, s_bwd_prop_getInFeature_Intermediates_0 _s_diff_ctx_4)
{
    uint _S487 = idx_2.x;
    uint _S488 = idx_2.y;
    uint _S489 = idx_2.z;
    uint4  _S490 = make_uint4 (_S487, _S488, _S489, 15U);
    s_diff_Feature_0 _S491 = Feature_x24_syn_dzero_0();
    s_diff_Feature_0 _S492 = _s_dOut_1;
    *(&(&_S492)->vals_0[15U]) = 0.0f;
    int _S493 = _s_diff_ctx_4._S314 - int(1);
    uint4  _S494 = make_uint4 (0U);
    s_diff_Feature_0 _S495 = Feature_x24_syn_dadd_0(_S492, _S491);
    s_bwd_prop_DiffTensorView_load_0(input_9, _S490, _s_dOut_1.vals_0[15U]);
    int _dc_0 = _S493;
    _S492 = _S495;
    for(;;)
    {
        if(_dc_0 >= int(0))
        {
        }
        else
        {
            break;
        }
        bool _S496 = _dc_0 < int(15);
        uint4  _S497;
        if(_S496)
        {
            _S497 = make_uint4 (_S487, _S488, _S489, uint(_dc_0));
        }
        else
        {
            _S497 = _S494;
        }
        s_diff_Feature_0 _S498 = Feature_x24_syn_dadd_0(_S492, _S491);
        if(_S496)
        {
            s_diff_Feature_0 _S499 = _S498;
            *(&(&_S499)->vals_0[_dc_0]) = 0.0f;
            s_bwd_prop_DiffTensorView_load_0(input_9, _S497, _S498.vals_0[_dc_0]);
            _S492 = Feature_x24_syn_dadd_0(_S499, _S491);
        }
        else
        {
            _S492 = Feature_x24_syn_dadd_0(_S498, _S491);
        }
        _dc_0 = _dc_0 - int(1);
    }
    return;
}

__device__ void s_bwd_prop_DiffTensorView_storeOnce_0(DiffTensorView_0 _S500, uint4  _S501, DiffPair_float_0 * _S502)
{
    DiffTensorView_storeOnce_backward_0(_S500, _S501, _S502);
    return;
}

__device__ void s_bwd_prop_renderImage_0(MLP_0 mlp_1, DiffTensorView_0 featureGrid_1, DiffTensorView_0 imageOutput_1, s_bwd_prop_renderImage_Intermediates_0 _s_diff_ctx_5, uint _S503)
{
    uint3  dispatchIdx_1 = ((threadIdx)) + ((blockIdx)) * ((blockDim));
    uint _S504 = dispatchIdx_1.x;
    uint _S505 = dispatchIdx_1.y;
    uint _S506 = dispatchIdx_1.z;
    bool _S507 = !(_S504 >= _s_diff_ctx_5._S323 || _S505 >= _s_diff_ctx_5._S324 || _S506 >= _s_diff_ctx_5._S325);
    uint _S508 = __ballot_sync(_S503, _S507);
    int _dc_1;
    uint _S509;
    uint _S510;
    if(_S507)
    {
        uint3  idx_3 = make_uint3 (_S504, _S505, _S506);
        s_bwd_prop_getInFeature_Intermediates_0 _S511 = _s_diff_ctx_5._S318;
        Feature_0 _S512 = s_bwd_primal_getInFeature_0(featureGrid_1, idx_3, &_S511);
        s_bwd_prop_MLP_eval_Intermediates_0 _S513 = _s_diff_ctx_5._S320;
        Feature_0 _S514 = s_bwd_primal_MLP_eval_0(mlp_1, _S512, &_S513);
        bool _bflag_2 = true;
        _dc_1 = int(0);
        int _pc_2 = int(0);
        _S509 = _S508;
        for(;;)
        {
            uint _S515 = 0U;
            uint _S516 = __ballot_sync(_S509, _bflag_2);
            if(_bflag_2)
            {
                uint _S517 = __ballot_sync(_S509, true);
                _S515 = _S517;
            }
            else
            {
                uint _S518 = __ballot_sync(_S509, false);
                uint _S519 = __ballot_sync(_S508, true);
                break;
            }
            s_primal_ctx_DiffTensorView_storeOnce_0(imageOutput_1, make_uint4 (_S504, _S505, _S506, uint(_dc_1)), _S514.vals_1[_dc_1]);
            int i_12 = _dc_1 + int(1);
            bool _S520 = i_12 < int(4);
            uint _S521 = __ballot_sync(_S515, _S520);
            if(_S520)
            {
                uint _S522 = __ballot_sync(_S515, true);
                _S510 = _S522;
            }
            else
            {
                uint _S523 = __ballot_sync(_S515, true);
                _bflag_2 = false;
                _S510 = _S523;
            }
            uint _S524 = __ballot_sync(_S510, _bflag_2);
            if(_bflag_2)
            {
                uint _S525 = __ballot_sync(_S510, true);
                _dc_1 = i_12;
                _S509 = _S525;
            }
            else
            {
                uint _S526 = __ballot_sync(_S510, true);
                _S509 = _S526;
            }
            _pc_2 = _pc_2 + int(1);
        }
        uint _S527 = __ballot_sync(_S503, true);
        _S509 = _S527;
    }
    else
    {
        uint _S528 = __ballot_sync(_S503, true);
        _S509 = _S528;
    }
    uint3  _S529 = make_uint3 (0U);
    uint _S530 = __ballot_sync(_S509, _S507);
    uint3  _S531;
    FixedArray<float, 16>  _S532;
    if(_S507)
    {
        uint3  idx_4 = make_uint3 (_S504, _S505, _S506);
        uint _S533 = __ballot_sync(_S509, true);
        _S531 = idx_4;
        _S532 = _s_diff_ctx_5._S321.vals_1;
        _S509 = _S533;
    }
    else
    {
        uint _S534 = __ballot_sync(_S509, true);
        _S531 = _S529;
        *(&_S532[int(0)]) = 0.0f;
        *(&_S532[int(1)]) = 0.0f;
        *(&_S532[int(2)]) = 0.0f;
        *(&_S532[int(3)]) = 0.0f;
        *(&_S532[int(4)]) = 0.0f;
        *(&_S532[int(5)]) = 0.0f;
        *(&_S532[int(6)]) = 0.0f;
        *(&_S532[int(7)]) = 0.0f;
        *(&_S532[int(8)]) = 0.0f;
        *(&_S532[int(9)]) = 0.0f;
        *(&_S532[int(10)]) = 0.0f;
        *(&_S532[int(11)]) = 0.0f;
        *(&_S532[int(12)]) = 0.0f;
        *(&_S532[int(13)]) = 0.0f;
        *(&_S532[int(14)]) = 0.0f;
        *(&_S532[int(15)]) = 0.0f;
        _S509 = _S534;
    }
    uint _S535 = __ballot_sync(_S509, _S507);
    if(_S507)
    {
        uint _S536 = 0U;
        s_diff_Feature_0 _S537 = Feature_x24_syn_dzero_0();
        _dc_1 = _s_diff_ctx_5._S322 - int(1);
        FixedArray<float, 16>  _S538;
        *(&_S538[int(0)]) = 0.0f;
        *(&_S538[int(1)]) = 0.0f;
        *(&_S538[int(2)]) = 0.0f;
        *(&_S538[int(3)]) = 0.0f;
        *(&_S538[int(4)]) = 0.0f;
        *(&_S538[int(5)]) = 0.0f;
        *(&_S538[int(6)]) = 0.0f;
        *(&_S538[int(7)]) = 0.0f;
        *(&_S538[int(8)]) = 0.0f;
        *(&_S538[int(9)]) = 0.0f;
        *(&_S538[int(10)]) = 0.0f;
        *(&_S538[int(11)]) = 0.0f;
        *(&_S538[int(12)]) = 0.0f;
        *(&_S538[int(13)]) = 0.0f;
        *(&_S538[int(14)]) = 0.0f;
        *(&_S538[int(15)]) = 0.0f;
        _S510 = _S535;
        for(;;)
        {
            bool _S539 = _dc_1 >= int(0);
            uint _S540 = __ballot_sync(_S510, _S539);
            if(_S539)
            {
                uint _S541 = __ballot_sync(_S510, true);
            }
            else
            {
                uint _S542 = __ballot_sync(_S510, false);
                uint _S543 = __ballot_sync(_S510, false);
                uint _S544 = __ballot_sync(_S535, true);
                _S536 = _S544;
                break;
            }
            uint4  _S545 = make_uint4 (_S504, _S505, _S506, uint(_dc_1));
            DiffPair_float_0 _S546;
            (&_S546)->primal_1 = _S532[_dc_1];
            (&_S546)->differential_0 = 0.0f;
            s_bwd_prop_DiffTensorView_storeOnce_0(imageOutput_1, _S545, &_S546);
            FixedArray<float, 16>  _S547;
            *(&_S547[int(0)]) = 0.0f;
            *(&_S547[int(1)]) = 0.0f;
            *(&_S547[int(2)]) = 0.0f;
            *(&_S547[int(3)]) = 0.0f;
            *(&_S547[int(4)]) = 0.0f;
            *(&_S547[int(5)]) = 0.0f;
            *(&_S547[int(6)]) = 0.0f;
            *(&_S547[int(7)]) = 0.0f;
            *(&_S547[int(8)]) = 0.0f;
            *(&_S547[int(9)]) = 0.0f;
            *(&_S547[int(10)]) = 0.0f;
            *(&_S547[int(11)]) = 0.0f;
            *(&_S547[int(12)]) = 0.0f;
            *(&_S547[int(13)]) = 0.0f;
            *(&_S547[int(14)]) = 0.0f;
            *(&_S547[int(15)]) = 0.0f;
            *(&_S547[_dc_1]) = _S546.differential_0;
            float _S548 = _S538[int(0)] + _S547[int(0)];
            float _S549 = _S538[int(1)] + _S547[int(1)];
            float _S550 = _S538[int(2)] + _S547[int(2)];
            float _S551 = _S538[int(3)] + _S547[int(3)];
            float _S552 = _S538[int(4)] + _S547[int(4)];
            float _S553 = _S538[int(5)] + _S547[int(5)];
            float _S554 = _S538[int(6)] + _S547[int(6)];
            float _S555 = _S538[int(7)] + _S547[int(7)];
            float _S556 = _S538[int(8)] + _S547[int(8)];
            float _S557 = _S538[int(9)] + _S547[int(9)];
            float _S558 = _S538[int(10)] + _S547[int(10)];
            float _S559 = _S538[int(11)] + _S547[int(11)];
            float _S560 = _S538[int(12)] + _S547[int(12)];
            float _S561 = _S538[int(13)] + _S547[int(13)];
            float _S562 = _S538[int(14)] + _S547[int(14)];
            float _S563 = _S538[int(15)] + _S547[int(15)];
            uint _S564 = __ballot_sync(_S510, true);
            _dc_1 = _dc_1 - int(1);
            *(&_S538[int(0)]) = _S548;
            *(&_S538[int(1)]) = _S549;
            *(&_S538[int(2)]) = _S550;
            *(&_S538[int(3)]) = _S551;
            *(&_S538[int(4)]) = _S552;
            *(&_S538[int(5)]) = _S553;
            *(&_S538[int(6)]) = _S554;
            *(&_S538[int(7)]) = _S555;
            *(&_S538[int(8)]) = _S556;
            *(&_S538[int(9)]) = _S557;
            *(&_S538[int(10)]) = _S558;
            *(&_S538[int(11)]) = _S559;
            *(&_S538[int(12)]) = _S560;
            *(&_S538[int(13)]) = _S561;
            *(&_S538[int(14)]) = _S562;
            *(&_S538[int(15)]) = _S563;
            _S510 = _S564;
        }
        s_diff_Feature_0 _S565 = _S537;
        (&_S565)->vals_0 = _S538;
        DiffPair_Feature_0 _S566;
        (&_S566)->primal_1 = _s_diff_ctx_5._S319;
        (&_S566)->differential_0 = _S537;
        s_bwd_prop_MLP_eval_0(mlp_1, &_S566, _S565, _s_diff_ctx_5._S320, _S536);
        s_bwd_prop_getInFeature_0(featureGrid_1, _S531, _S566.differential_0, _s_diff_ctx_5._S318);
        uint _S567 = __ballot_sync(_S509, true);
    }
    return;
}

__device__ void s_bwd_renderImage_0(MLP_0 _S568, DiffTensorView_0 _S569, DiffTensorView_0 _S570, uint _S571)
{
    s_bwd_prop_renderImage_Intermediates_0 _S572;
    s_bwd_primal_renderImage_0(_S568, _S569, _S570, &_S572);
    s_bwd_prop_renderImage_0(_S568, _S569, _S570, _S572, _S571);
    return;
}

extern "C" {
__global__ void __kernel__renderImage_bwd_diff(MLP_0 _S573, DiffTensorView_0 _S574, DiffTensorView_0 _S575)
{
    uint _S576 = __ballot_sync(4294967295U, true);
    s_bwd_renderImage_0(_S573, _S574, _S575, _S576);
    return;
}

}
__device__ DiffPair_Feature_0 s_fwd_getInFeature_0(DiffTensorView_0 input_10, uint3  idx_5)
{
    FixedArray<float, 16>  _S577 = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    uint _S578 = idx_5.x;
    uint _S579 = idx_5.y;
    uint _S580 = idx_5.z;
    uint4  _S581 = make_uint4 (_S578, _S579, _S580, 15U);
    int i_13 = int(0);
    Feature_0 output_4;
    (&output_4)->vals_1 = _S577;
    s_diff_Feature_0 s_diff_output_0;
    (&s_diff_output_0)->vals_0 = _S577;
    for(;;)
    {
        if(i_13 < int(15))
        {
        }
        else
        {
            break;
        }
        DiffPair_float_0 _S582 = DiffTensorView_load_forward_0(input_10, make_uint4 (_S578, _S579, _S580, uint(i_13)));
        *(&(&output_4)->vals_1[i_13]) = _S582.primal_1;
        *(&(&s_diff_output_0)->vals_0[i_13]) = _S582.differential_0;
        i_13 = i_13 + int(1);
    }
    DiffPair_float_0 _S583 = DiffTensorView_load_forward_0(input_10, _S581);
    *(&(&output_4)->vals_1[15U]) = _S583.primal_1;
    *(&(&s_diff_output_0)->vals_0[15U]) = _S583.differential_0;
    DiffPair_Feature_0 _S584 = { output_4, s_diff_output_0 };
    return _S584;
}

__device__ DiffPair_Feature_0 s_fwd_Linear_eval_0(Linear_0 this_31, DiffPair_Feature_0 dpin_feature_2)
{
    float * inPtr_4 = Linear_moveInputsToSharedMem_0(this_31, dpin_feature_2.primal_1.vals_1);
    float * wtPtr_5 = Linear_moveWeightsToSharedMem_0(this_31);
    float * outPtr_4 = Linear_outBufferForCurrentWarp_0(this_31);
    float _S585 = *inPtr_4;
    float _S586 = *wtPtr_5;
    float _S587 = *outPtr_4;
    _inline_matmul_1(&_S585, &_S586, &_S587);
    *outPtr_4 = _S587;
    *wtPtr_5 = _S586;
    *inPtr_4 = _S585;
    FixedArray<float, 16>  _S588 = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    float * _S589 = ((this_31.bias_0.primal_0).data_ptr<float>());
    FixedArray<float, 16>  _S590;
    *(&_S590[int(0)]) = 0.0f;
    *(&_S590[int(1)]) = 0.0f;
    *(&_S590[int(2)]) = 0.0f;
    *(&_S590[int(3)]) = 0.0f;
    *(&_S590[int(4)]) = 0.0f;
    *(&_S590[int(5)]) = 0.0f;
    *(&_S590[int(6)]) = 0.0f;
    *(&_S590[int(7)]) = 0.0f;
    *(&_S590[int(8)]) = 0.0f;
    *(&_S590[int(9)]) = 0.0f;
    *(&_S590[int(10)]) = 0.0f;
    *(&_S590[int(11)]) = 0.0f;
    *(&_S590[int(12)]) = 0.0f;
    *(&_S590[int(13)]) = 0.0f;
    *(&_S590[int(14)]) = 0.0f;
    *(&_S590[int(15)]) = 0.0f;
    float _S591 = *_S589;
    Linear_moveOutputsToLocalArray_0(this_31, &_S590, &_S591);
    *_S589 = _S591;
    Feature_0 _S592 = { _S590 };
    s_diff_Feature_0 _S593 = { _S588 };
    DiffPair_Feature_0 _S594 = { _S592, _S593 };
    return _S594;
}

__device__ DiffPair_Feature_0 s_fwd_MLP_eval_0(MLP_0 this_32, DiffPair_Feature_0 dpin_feature_3)
{
    DiffPair_Feature_0 _S595 = { dpin_feature_3.primal_1, dpin_feature_3.differential_0 };
    DiffPair_float_0 _S596 = { 0.0f, 0.0f };
    DiffPair_Feature_0 _S597 = s_fwd_Linear_eval_0(this_32.layers_0[int(0)], _S595);
    DiffPair_float_0 _S598 = { _S597.primal_1.vals_1[int(0)], _S597.differential_0.vals_0[int(0)] };
    DiffPair_float_0 _S599 = _d_max_1(_S596, _S598);
    Feature_0 _S600 = _S597.primal_1;
    *(&(&_S600)->vals_1[int(0)]) = _S599.primal_1;
    s_diff_Feature_0 _S601 = _S597.differential_0;
    *(&(&_S601)->vals_0[int(0)]) = _S599.differential_0;
    DiffPair_float_0 _S602 = { _S597.primal_1.vals_1[int(1)], _S597.differential_0.vals_0[int(1)] };
    DiffPair_float_0 _S603 = _d_max_1(_S596, _S602);
    *(&(&_S600)->vals_1[int(1)]) = _S603.primal_1;
    *(&(&_S601)->vals_0[int(1)]) = _S603.differential_0;
    DiffPair_float_0 _S604 = { _S597.primal_1.vals_1[int(2)], _S597.differential_0.vals_0[int(2)] };
    DiffPair_float_0 _S605 = _d_max_1(_S596, _S604);
    *(&(&_S600)->vals_1[int(2)]) = _S605.primal_1;
    *(&(&_S601)->vals_0[int(2)]) = _S605.differential_0;
    DiffPair_float_0 _S606 = { _S597.primal_1.vals_1[int(3)], _S597.differential_0.vals_0[int(3)] };
    DiffPair_float_0 _S607 = _d_max_1(_S596, _S606);
    *(&(&_S600)->vals_1[int(3)]) = _S607.primal_1;
    *(&(&_S601)->vals_0[int(3)]) = _S607.differential_0;
    DiffPair_float_0 _S608 = { _S597.primal_1.vals_1[int(4)], _S597.differential_0.vals_0[int(4)] };
    DiffPair_float_0 _S609 = _d_max_1(_S596, _S608);
    *(&(&_S600)->vals_1[int(4)]) = _S609.primal_1;
    *(&(&_S601)->vals_0[int(4)]) = _S609.differential_0;
    DiffPair_float_0 _S610 = { _S597.primal_1.vals_1[int(5)], _S597.differential_0.vals_0[int(5)] };
    DiffPair_float_0 _S611 = _d_max_1(_S596, _S610);
    *(&(&_S600)->vals_1[int(5)]) = _S611.primal_1;
    *(&(&_S601)->vals_0[int(5)]) = _S611.differential_0;
    DiffPair_float_0 _S612 = { _S597.primal_1.vals_1[int(6)], _S597.differential_0.vals_0[int(6)] };
    DiffPair_float_0 _S613 = _d_max_1(_S596, _S612);
    *(&(&_S600)->vals_1[int(6)]) = _S613.primal_1;
    *(&(&_S601)->vals_0[int(6)]) = _S613.differential_0;
    DiffPair_float_0 _S614 = { _S597.primal_1.vals_1[int(7)], _S597.differential_0.vals_0[int(7)] };
    DiffPair_float_0 _S615 = _d_max_1(_S596, _S614);
    *(&(&_S600)->vals_1[int(7)]) = _S615.primal_1;
    *(&(&_S601)->vals_0[int(7)]) = _S615.differential_0;
    DiffPair_float_0 _S616 = { _S597.primal_1.vals_1[int(8)], _S597.differential_0.vals_0[int(8)] };
    DiffPair_float_0 _S617 = _d_max_1(_S596, _S616);
    *(&(&_S600)->vals_1[int(8)]) = _S617.primal_1;
    *(&(&_S601)->vals_0[int(8)]) = _S617.differential_0;
    DiffPair_float_0 _S618 = { _S597.primal_1.vals_1[int(9)], _S597.differential_0.vals_0[int(9)] };
    DiffPair_float_0 _S619 = _d_max_1(_S596, _S618);
    *(&(&_S600)->vals_1[int(9)]) = _S619.primal_1;
    *(&(&_S601)->vals_0[int(9)]) = _S619.differential_0;
    DiffPair_float_0 _S620 = { _S597.primal_1.vals_1[int(10)], _S597.differential_0.vals_0[int(10)] };
    DiffPair_float_0 _S621 = _d_max_1(_S596, _S620);
    *(&(&_S600)->vals_1[int(10)]) = _S621.primal_1;
    *(&(&_S601)->vals_0[int(10)]) = _S621.differential_0;
    DiffPair_float_0 _S622 = { _S597.primal_1.vals_1[int(11)], _S597.differential_0.vals_0[int(11)] };
    DiffPair_float_0 _S623 = _d_max_1(_S596, _S622);
    *(&(&_S600)->vals_1[int(11)]) = _S623.primal_1;
    *(&(&_S601)->vals_0[int(11)]) = _S623.differential_0;
    DiffPair_float_0 _S624 = { _S597.primal_1.vals_1[int(12)], _S597.differential_0.vals_0[int(12)] };
    DiffPair_float_0 _S625 = _d_max_1(_S596, _S624);
    *(&(&_S600)->vals_1[int(12)]) = _S625.primal_1;
    *(&(&_S601)->vals_0[int(12)]) = _S625.differential_0;
    DiffPair_float_0 _S626 = { _S597.primal_1.vals_1[int(13)], _S597.differential_0.vals_0[int(13)] };
    DiffPair_float_0 _S627 = _d_max_1(_S596, _S626);
    *(&(&_S600)->vals_1[int(13)]) = _S627.primal_1;
    *(&(&_S601)->vals_0[int(13)]) = _S627.differential_0;
    DiffPair_float_0 _S628 = { _S597.primal_1.vals_1[int(14)], _S597.differential_0.vals_0[int(14)] };
    DiffPair_float_0 _S629 = _d_max_1(_S596, _S628);
    *(&(&_S600)->vals_1[int(14)]) = _S629.primal_1;
    *(&(&_S601)->vals_0[int(14)]) = _S629.differential_0;
    DiffPair_float_0 _S630 = { _S597.primal_1.vals_1[int(15)], _S597.differential_0.vals_0[int(15)] };
    DiffPair_float_0 _S631 = _d_max_1(_S596, _S630);
    *(&(&_S600)->vals_1[int(15)]) = _S631.primal_1;
    *(&(&_S601)->vals_0[int(15)]) = _S631.differential_0;
    DiffPair_Feature_0 _S632 = { _S600, _S601 };
    DiffPair_Feature_0 _S633 = s_fwd_Linear_eval_0(this_32.layers_0[int(1)], _S632);
    DiffPair_float_0 _S634 = { _S633.primal_1.vals_1[int(0)], _S633.differential_0.vals_0[int(0)] };
    DiffPair_float_0 _S635 = _d_max_1(_S596, _S634);
    _S600 = _S633.primal_1;
    *(&(&_S600)->vals_1[int(0)]) = _S635.primal_1;
    _S601 = _S633.differential_0;
    *(&(&_S601)->vals_0[int(0)]) = _S635.differential_0;
    DiffPair_float_0 _S636 = { _S633.primal_1.vals_1[int(1)], _S633.differential_0.vals_0[int(1)] };
    DiffPair_float_0 _S637 = _d_max_1(_S596, _S636);
    *(&(&_S600)->vals_1[int(1)]) = _S637.primal_1;
    *(&(&_S601)->vals_0[int(1)]) = _S637.differential_0;
    DiffPair_float_0 _S638 = { _S633.primal_1.vals_1[int(2)], _S633.differential_0.vals_0[int(2)] };
    DiffPair_float_0 _S639 = _d_max_1(_S596, _S638);
    *(&(&_S600)->vals_1[int(2)]) = _S639.primal_1;
    *(&(&_S601)->vals_0[int(2)]) = _S639.differential_0;
    DiffPair_float_0 _S640 = { _S633.primal_1.vals_1[int(3)], _S633.differential_0.vals_0[int(3)] };
    DiffPair_float_0 _S641 = _d_max_1(_S596, _S640);
    *(&(&_S600)->vals_1[int(3)]) = _S641.primal_1;
    *(&(&_S601)->vals_0[int(3)]) = _S641.differential_0;
    DiffPair_float_0 _S642 = { _S633.primal_1.vals_1[int(4)], _S633.differential_0.vals_0[int(4)] };
    DiffPair_float_0 _S643 = _d_max_1(_S596, _S642);
    *(&(&_S600)->vals_1[int(4)]) = _S643.primal_1;
    *(&(&_S601)->vals_0[int(4)]) = _S643.differential_0;
    DiffPair_float_0 _S644 = { _S633.primal_1.vals_1[int(5)], _S633.differential_0.vals_0[int(5)] };
    DiffPair_float_0 _S645 = _d_max_1(_S596, _S644);
    *(&(&_S600)->vals_1[int(5)]) = _S645.primal_1;
    *(&(&_S601)->vals_0[int(5)]) = _S645.differential_0;
    DiffPair_float_0 _S646 = { _S633.primal_1.vals_1[int(6)], _S633.differential_0.vals_0[int(6)] };
    DiffPair_float_0 _S647 = _d_max_1(_S596, _S646);
    *(&(&_S600)->vals_1[int(6)]) = _S647.primal_1;
    *(&(&_S601)->vals_0[int(6)]) = _S647.differential_0;
    DiffPair_float_0 _S648 = { _S633.primal_1.vals_1[int(7)], _S633.differential_0.vals_0[int(7)] };
    DiffPair_float_0 _S649 = _d_max_1(_S596, _S648);
    *(&(&_S600)->vals_1[int(7)]) = _S649.primal_1;
    *(&(&_S601)->vals_0[int(7)]) = _S649.differential_0;
    DiffPair_float_0 _S650 = { _S633.primal_1.vals_1[int(8)], _S633.differential_0.vals_0[int(8)] };
    DiffPair_float_0 _S651 = _d_max_1(_S596, _S650);
    *(&(&_S600)->vals_1[int(8)]) = _S651.primal_1;
    *(&(&_S601)->vals_0[int(8)]) = _S651.differential_0;
    DiffPair_float_0 _S652 = { _S633.primal_1.vals_1[int(9)], _S633.differential_0.vals_0[int(9)] };
    DiffPair_float_0 _S653 = _d_max_1(_S596, _S652);
    *(&(&_S600)->vals_1[int(9)]) = _S653.primal_1;
    *(&(&_S601)->vals_0[int(9)]) = _S653.differential_0;
    DiffPair_float_0 _S654 = { _S633.primal_1.vals_1[int(10)], _S633.differential_0.vals_0[int(10)] };
    DiffPair_float_0 _S655 = _d_max_1(_S596, _S654);
    *(&(&_S600)->vals_1[int(10)]) = _S655.primal_1;
    *(&(&_S601)->vals_0[int(10)]) = _S655.differential_0;
    DiffPair_float_0 _S656 = { _S633.primal_1.vals_1[int(11)], _S633.differential_0.vals_0[int(11)] };
    DiffPair_float_0 _S657 = _d_max_1(_S596, _S656);
    *(&(&_S600)->vals_1[int(11)]) = _S657.primal_1;
    *(&(&_S601)->vals_0[int(11)]) = _S657.differential_0;
    DiffPair_float_0 _S658 = { _S633.primal_1.vals_1[int(12)], _S633.differential_0.vals_0[int(12)] };
    DiffPair_float_0 _S659 = _d_max_1(_S596, _S658);
    *(&(&_S600)->vals_1[int(12)]) = _S659.primal_1;
    *(&(&_S601)->vals_0[int(12)]) = _S659.differential_0;
    DiffPair_float_0 _S660 = { _S633.primal_1.vals_1[int(13)], _S633.differential_0.vals_0[int(13)] };
    DiffPair_float_0 _S661 = _d_max_1(_S596, _S660);
    *(&(&_S600)->vals_1[int(13)]) = _S661.primal_1;
    *(&(&_S601)->vals_0[int(13)]) = _S661.differential_0;
    DiffPair_float_0 _S662 = { _S633.primal_1.vals_1[int(14)], _S633.differential_0.vals_0[int(14)] };
    DiffPair_float_0 _S663 = _d_max_1(_S596, _S662);
    *(&(&_S600)->vals_1[int(14)]) = _S663.primal_1;
    *(&(&_S601)->vals_0[int(14)]) = _S663.differential_0;
    DiffPair_float_0 _S664 = { _S633.primal_1.vals_1[int(15)], _S633.differential_0.vals_0[int(15)] };
    DiffPair_float_0 _S665 = _d_max_1(_S596, _S664);
    *(&(&_S600)->vals_1[int(15)]) = _S665.primal_1;
    *(&(&_S601)->vals_0[int(15)]) = _S665.differential_0;
    DiffPair_Feature_0 _S666 = { _S600, _S601 };
    DiffPair_Feature_0 _S667 = s_fwd_Linear_eval_0(this_32.layers_0[int(2)], _S666);
    DiffPair_float_0 _S668 = { _S667.primal_1.vals_1[int(0)], _S667.differential_0.vals_0[int(0)] };
    DiffPair_float_0 _S669 = _d_max_1(_S596, _S668);
    _S600 = _S667.primal_1;
    *(&(&_S600)->vals_1[int(0)]) = _S669.primal_1;
    _S601 = _S667.differential_0;
    *(&(&_S601)->vals_0[int(0)]) = _S669.differential_0;
    DiffPair_float_0 _S670 = { _S667.primal_1.vals_1[int(1)], _S667.differential_0.vals_0[int(1)] };
    DiffPair_float_0 _S671 = _d_max_1(_S596, _S670);
    *(&(&_S600)->vals_1[int(1)]) = _S671.primal_1;
    *(&(&_S601)->vals_0[int(1)]) = _S671.differential_0;
    DiffPair_float_0 _S672 = { _S667.primal_1.vals_1[int(2)], _S667.differential_0.vals_0[int(2)] };
    DiffPair_float_0 _S673 = _d_max_1(_S596, _S672);
    *(&(&_S600)->vals_1[int(2)]) = _S673.primal_1;
    *(&(&_S601)->vals_0[int(2)]) = _S673.differential_0;
    DiffPair_float_0 _S674 = { _S667.primal_1.vals_1[int(3)], _S667.differential_0.vals_0[int(3)] };
    DiffPair_float_0 _S675 = _d_max_1(_S596, _S674);
    *(&(&_S600)->vals_1[int(3)]) = _S675.primal_1;
    *(&(&_S601)->vals_0[int(3)]) = _S675.differential_0;
    DiffPair_float_0 _S676 = { _S667.primal_1.vals_1[int(4)], _S667.differential_0.vals_0[int(4)] };
    DiffPair_float_0 _S677 = _d_max_1(_S596, _S676);
    *(&(&_S600)->vals_1[int(4)]) = _S677.primal_1;
    *(&(&_S601)->vals_0[int(4)]) = _S677.differential_0;
    DiffPair_float_0 _S678 = { _S667.primal_1.vals_1[int(5)], _S667.differential_0.vals_0[int(5)] };
    DiffPair_float_0 _S679 = _d_max_1(_S596, _S678);
    *(&(&_S600)->vals_1[int(5)]) = _S679.primal_1;
    *(&(&_S601)->vals_0[int(5)]) = _S679.differential_0;
    DiffPair_float_0 _S680 = { _S667.primal_1.vals_1[int(6)], _S667.differential_0.vals_0[int(6)] };
    DiffPair_float_0 _S681 = _d_max_1(_S596, _S680);
    *(&(&_S600)->vals_1[int(6)]) = _S681.primal_1;
    *(&(&_S601)->vals_0[int(6)]) = _S681.differential_0;
    DiffPair_float_0 _S682 = { _S667.primal_1.vals_1[int(7)], _S667.differential_0.vals_0[int(7)] };
    DiffPair_float_0 _S683 = _d_max_1(_S596, _S682);
    *(&(&_S600)->vals_1[int(7)]) = _S683.primal_1;
    *(&(&_S601)->vals_0[int(7)]) = _S683.differential_0;
    DiffPair_float_0 _S684 = { _S667.primal_1.vals_1[int(8)], _S667.differential_0.vals_0[int(8)] };
    DiffPair_float_0 _S685 = _d_max_1(_S596, _S684);
    *(&(&_S600)->vals_1[int(8)]) = _S685.primal_1;
    *(&(&_S601)->vals_0[int(8)]) = _S685.differential_0;
    DiffPair_float_0 _S686 = { _S667.primal_1.vals_1[int(9)], _S667.differential_0.vals_0[int(9)] };
    DiffPair_float_0 _S687 = _d_max_1(_S596, _S686);
    *(&(&_S600)->vals_1[int(9)]) = _S687.primal_1;
    *(&(&_S601)->vals_0[int(9)]) = _S687.differential_0;
    DiffPair_float_0 _S688 = { _S667.primal_1.vals_1[int(10)], _S667.differential_0.vals_0[int(10)] };
    DiffPair_float_0 _S689 = _d_max_1(_S596, _S688);
    *(&(&_S600)->vals_1[int(10)]) = _S689.primal_1;
    *(&(&_S601)->vals_0[int(10)]) = _S689.differential_0;
    DiffPair_float_0 _S690 = { _S667.primal_1.vals_1[int(11)], _S667.differential_0.vals_0[int(11)] };
    DiffPair_float_0 _S691 = _d_max_1(_S596, _S690);
    *(&(&_S600)->vals_1[int(11)]) = _S691.primal_1;
    *(&(&_S601)->vals_0[int(11)]) = _S691.differential_0;
    DiffPair_float_0 _S692 = { _S667.primal_1.vals_1[int(12)], _S667.differential_0.vals_0[int(12)] };
    DiffPair_float_0 _S693 = _d_max_1(_S596, _S692);
    *(&(&_S600)->vals_1[int(12)]) = _S693.primal_1;
    *(&(&_S601)->vals_0[int(12)]) = _S693.differential_0;
    DiffPair_float_0 _S694 = { _S667.primal_1.vals_1[int(13)], _S667.differential_0.vals_0[int(13)] };
    DiffPair_float_0 _S695 = _d_max_1(_S596, _S694);
    *(&(&_S600)->vals_1[int(13)]) = _S695.primal_1;
    *(&(&_S601)->vals_0[int(13)]) = _S695.differential_0;
    DiffPair_float_0 _S696 = { _S667.primal_1.vals_1[int(14)], _S667.differential_0.vals_0[int(14)] };
    DiffPair_float_0 _S697 = _d_max_1(_S596, _S696);
    *(&(&_S600)->vals_1[int(14)]) = _S697.primal_1;
    *(&(&_S601)->vals_0[int(14)]) = _S697.differential_0;
    DiffPair_float_0 _S698 = { _S667.primal_1.vals_1[int(15)], _S667.differential_0.vals_0[int(15)] };
    DiffPair_float_0 _S699 = _d_max_1(_S596, _S698);
    *(&(&_S600)->vals_1[int(15)]) = _S699.primal_1;
    *(&(&_S601)->vals_0[int(15)]) = _S699.differential_0;
    DiffPair_Feature_0 _S700 = { _S600, _S601 };
    return _S700;
}

__device__ void s_fwd_renderImage_0(MLP_0 mlp_2, DiffTensorView_0 featureGrid_2, DiffTensorView_0 imageOutput_2)
{
    uint3  dispatchIdx_2 = ((threadIdx)) + ((blockIdx)) * ((blockDim));
    uint _S701 = dispatchIdx_2.x;
    uint _S702 = dispatchIdx_2.y;
    uint _S703 = dispatchIdx_2.z;
    if(_S701 >= DiffTensorView_size_0(imageOutput_2, 0U) || _S702 >= DiffTensorView_size_0(imageOutput_2, 1U) || _S703 >= DiffTensorView_size_0(imageOutput_2, 2U))
    {
        return;
    }
    DiffPair_Feature_0 _S704 = s_fwd_getInFeature_0(featureGrid_2, make_uint3 (_S701, _S702, _S703));
    DiffPair_Feature_0 _S705 = { _S704.primal_1, _S704.differential_0 };
    DiffPair_Feature_0 _S706 = s_fwd_MLP_eval_0(mlp_2, _S705);
    int i_14 = int(0);
    for(;;)
    {
        DiffPair_float_0 _S707 = { _S706.primal_1.vals_1[i_14], _S706.differential_0.vals_0[i_14] };
        DiffTensorView_storeOnce_forward_0(imageOutput_2, make_uint4 (_S701, _S702, _S703, uint(i_14)), _S707);
        int i_15 = i_14 + int(1);
        if(i_15 < int(4))
        {
        }
        else
        {
            break;
        }
        i_14 = i_15;
    }
    return;
}

extern "C" {
__global__ void __kernel__renderImage_fwd_diff(MLP_0 _S708, DiffTensorView_0 _S709, DiffTensorView_0 _S710)
{
    s_fwd_renderImage_0(_S708, _S709, _S710);
    return;
}

}
extern "C" {
__global__ void __kernel__renderImage(MLP_0 mlp_3, DiffTensorView_0 featureGrid_3, DiffTensorView_0 imageOutput_3)
{
    uint3  dispatchIdx_3 = ((threadIdx)) + ((blockIdx)) * ((blockDim));
    uint _S711 = dispatchIdx_3.x;
    uint _S712 = dispatchIdx_3.y;
    uint _S713 = dispatchIdx_3.z;
    if(_S711 >= DiffTensorView_size_0(imageOutput_3, 0U) || _S712 >= DiffTensorView_size_0(imageOutput_3, 1U) || _S713 >= DiffTensorView_size_0(imageOutput_3, 2U))
    {
        return;
    }
    Feature_0 _S714 = MLP_eval_0(mlp_3, getInFeature_0(featureGrid_3, make_uint3 (_S711, _S712, _S713)));
    int i_16 = int(0);
    for(;;)
    {
        DiffTensorView_storeOnce_0(imageOutput_3, make_uint4 (_S711, _S712, _S713, uint(i_16)), _S714.vals_1[i_16]);
        int i_17 = i_16 + int(1);
        if(i_17 < int(4))
        {
        }
        else
        {
            break;
        }
        i_16 = i_17;
    }
    return;
}

}
