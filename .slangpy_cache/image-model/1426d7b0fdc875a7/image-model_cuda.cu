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

__device__ float AtomicAdd_load_forward_0(AtomicAdd_0 this_0, uint3  i_0)
{
    float _S1 = ((this_0.diff_0).load<float>((i_0)));
    return _S1;
}

__device__ void AtomicAdd_load_backward_0(AtomicAdd_0 this_1, uint3  i_1, float dOut_0)
{
    float oldVal_0;
    *((&oldVal_0)) = atomicAdd((this_1.diff_0).data_ptr_at<float>((i_1)), (dOut_0));
    return;
}

__device__ void AtomicAdd_storeOnce_forward_0(AtomicAdd_0 this_2, uint3  i_2, float dx_0)
{
    (this_2.diff_0).store<float>((i_2), (dx_0));
    return;
}

__device__ float AtomicAdd_storeOnce_backward_0(AtomicAdd_0 this_3, uint3  i_3)
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

__device__ DiffPair_float_0 DiffTensorView_load_forward_0(DiffTensorView_0 this_5, uint3  x_0)
{
    float _S19 = ((this_5.primal_0).load<float>((x_0)));
    DiffPair_float_0 _S20 = { _S19, AtomicAdd_load_forward_0(this_5.diff_1, x_0) };
    return _S20;
}

__device__ void DiffTensorView_load_backward_0(DiffTensorView_0 this_6, uint3  x_1, float dOut_1)
{
    AtomicAdd_load_backward_0(this_6.diff_1, x_1, dOut_1);
    return;
}

__device__ float DiffTensorView_load_0(DiffTensorView_0 this_7, uint3  i_5)
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

__device__ Feature_0 computeInterpolatedFeature_0(DiffTensorView_0 featureGrid_0, uint2  frameDim_0)
{
    uint dim1_0 = DiffTensorView_size_0(featureGrid_0, 1U) - 1U;
    uint _S23 = frameDim_0.x / (DiffTensorView_size_0(featureGrid_0, 0U) - 1U);
    uint _S24 = frameDim_0.y / dim1_0;
    uint2  cellSize_0 = make_uint2 (_S23, _S24);
    uint2  pixelId_0 = uint2 {(((threadIdx)) + ((blockIdx)) * ((blockDim))).x, (((threadIdx)) + ((blockIdx)) * ((blockDim))).y};
    uint2  cellId_0 = pixelId_0 / cellSize_0;
    float2  _S25 = make_float2 ((float)pixelId_0.x, (float)pixelId_0.y);
    float2  _S26 = make_float2 ((float)cellSize_0.x, (float)cellSize_0.y);
    float2  _S27 = make_float2 ((float)cellId_0.x, (float)cellId_0.y);
    float2  weights_0 = _S25 / _S26 - _S27;
    Feature_0 feature_0;
    uint _S28 = cellId_0.x;
    uint _S29 = cellId_0.y;
    float _S30 = weights_0.x;
    float _S31 = 1.0f - _S30;
    float _S32 = weights_0.y;
    float _S33 = 1.0f - _S32;
    uint _S34 = _S28 + 1U;
    uint _S35 = _S29 + 1U;
    uint3  _S36 = make_uint3 (_S28, _S29, 1U);
    uint3  _S37 = make_uint3 (_S34, _S29, 1U);
    uint3  _S38 = make_uint3 (_S28, _S35, 1U);
    uint3  _S39 = make_uint3 (_S34, _S35, 1U);
    uint3  _S40 = make_uint3 (_S28, _S29, 2U);
    uint3  _S41 = make_uint3 (_S34, _S29, 2U);
    uint3  _S42 = make_uint3 (_S28, _S35, 2U);
    uint3  _S43 = make_uint3 (_S34, _S35, 2U);
    uint3  _S44 = make_uint3 (_S28, _S29, 3U);
    uint3  _S45 = make_uint3 (_S34, _S29, 3U);
    uint3  _S46 = make_uint3 (_S28, _S35, 3U);
    uint3  _S47 = make_uint3 (_S34, _S35, 3U);
    uint3  _S48 = make_uint3 (_S28, _S29, 4U);
    uint3  _S49 = make_uint3 (_S34, _S29, 4U);
    uint3  _S50 = make_uint3 (_S28, _S35, 4U);
    uint3  _S51 = make_uint3 (_S34, _S35, 4U);
    uint3  _S52 = make_uint3 (_S28, _S29, 5U);
    uint3  _S53 = make_uint3 (_S34, _S29, 5U);
    uint3  _S54 = make_uint3 (_S28, _S35, 5U);
    uint3  _S55 = make_uint3 (_S34, _S35, 5U);
    uint3  _S56 = make_uint3 (_S28, _S29, 6U);
    uint3  _S57 = make_uint3 (_S34, _S29, 6U);
    uint3  _S58 = make_uint3 (_S28, _S35, 6U);
    uint3  _S59 = make_uint3 (_S34, _S35, 6U);
    uint3  _S60 = make_uint3 (_S28, _S29, 7U);
    uint3  _S61 = make_uint3 (_S34, _S29, 7U);
    uint3  _S62 = make_uint3 (_S28, _S35, 7U);
    uint3  _S63 = make_uint3 (_S34, _S35, 7U);
    uint3  _S64 = make_uint3 (_S28, _S29, 8U);
    uint3  _S65 = make_uint3 (_S34, _S29, 8U);
    uint3  _S66 = make_uint3 (_S28, _S35, 8U);
    uint3  _S67 = make_uint3 (_S34, _S35, 8U);
    uint3  _S68 = make_uint3 (_S28, _S29, 9U);
    uint3  _S69 = make_uint3 (_S34, _S29, 9U);
    uint3  _S70 = make_uint3 (_S28, _S35, 9U);
    uint3  _S71 = make_uint3 (_S34, _S35, 9U);
    uint3  _S72 = make_uint3 (_S28, _S29, 10U);
    uint3  _S73 = make_uint3 (_S34, _S29, 10U);
    uint3  _S74 = make_uint3 (_S28, _S35, 10U);
    uint3  _S75 = make_uint3 (_S34, _S35, 10U);
    uint3  _S76 = make_uint3 (_S28, _S29, 11U);
    uint3  _S77 = make_uint3 (_S34, _S29, 11U);
    uint3  _S78 = make_uint3 (_S28, _S35, 11U);
    uint3  _S79 = make_uint3 (_S34, _S35, 11U);
    uint3  _S80 = make_uint3 (_S28, _S29, 12U);
    uint3  _S81 = make_uint3 (_S34, _S29, 12U);
    uint3  _S82 = make_uint3 (_S28, _S35, 12U);
    uint3  _S83 = make_uint3 (_S34, _S35, 12U);
    uint3  _S84 = make_uint3 (_S28, _S29, 13U);
    uint3  _S85 = make_uint3 (_S34, _S29, 13U);
    uint3  _S86 = make_uint3 (_S28, _S35, 13U);
    uint3  _S87 = make_uint3 (_S34, _S35, 13U);
    *(&(&feature_0)->vals_1[int(0)]) = DiffTensorView_load_0(featureGrid_0, make_uint3 (_S28, _S29, 0U)) * _S31 * _S33 + DiffTensorView_load_0(featureGrid_0, make_uint3 (_S34, _S29, 0U)) * _S30 * _S33 + DiffTensorView_load_0(featureGrid_0, make_uint3 (_S28, _S35, 0U)) * _S31 * _S32 + DiffTensorView_load_0(featureGrid_0, make_uint3 (_S34, _S35, 0U)) * _S30 * _S32;
    *(&(&feature_0)->vals_1[int(1)]) = DiffTensorView_load_0(featureGrid_0, _S36) * _S31 * _S33 + DiffTensorView_load_0(featureGrid_0, _S37) * _S30 * _S33 + DiffTensorView_load_0(featureGrid_0, _S38) * _S31 * _S32 + DiffTensorView_load_0(featureGrid_0, _S39) * _S30 * _S32;
    *(&(&feature_0)->vals_1[int(2)]) = DiffTensorView_load_0(featureGrid_0, _S40) * _S31 * _S33 + DiffTensorView_load_0(featureGrid_0, _S41) * _S30 * _S33 + DiffTensorView_load_0(featureGrid_0, _S42) * _S31 * _S32 + DiffTensorView_load_0(featureGrid_0, _S43) * _S30 * _S32;
    *(&(&feature_0)->vals_1[int(3)]) = DiffTensorView_load_0(featureGrid_0, _S44) * _S31 * _S33 + DiffTensorView_load_0(featureGrid_0, _S45) * _S30 * _S33 + DiffTensorView_load_0(featureGrid_0, _S46) * _S31 * _S32 + DiffTensorView_load_0(featureGrid_0, _S47) * _S30 * _S32;
    *(&(&feature_0)->vals_1[int(4)]) = DiffTensorView_load_0(featureGrid_0, _S48) * _S31 * _S33 + DiffTensorView_load_0(featureGrid_0, _S49) * _S30 * _S33 + DiffTensorView_load_0(featureGrid_0, _S50) * _S31 * _S32 + DiffTensorView_load_0(featureGrid_0, _S51) * _S30 * _S32;
    *(&(&feature_0)->vals_1[int(5)]) = DiffTensorView_load_0(featureGrid_0, _S52) * _S31 * _S33 + DiffTensorView_load_0(featureGrid_0, _S53) * _S30 * _S33 + DiffTensorView_load_0(featureGrid_0, _S54) * _S31 * _S32 + DiffTensorView_load_0(featureGrid_0, _S55) * _S30 * _S32;
    *(&(&feature_0)->vals_1[int(6)]) = DiffTensorView_load_0(featureGrid_0, _S56) * _S31 * _S33 + DiffTensorView_load_0(featureGrid_0, _S57) * _S30 * _S33 + DiffTensorView_load_0(featureGrid_0, _S58) * _S31 * _S32 + DiffTensorView_load_0(featureGrid_0, _S59) * _S30 * _S32;
    *(&(&feature_0)->vals_1[int(7)]) = DiffTensorView_load_0(featureGrid_0, _S60) * _S31 * _S33 + DiffTensorView_load_0(featureGrid_0, _S61) * _S30 * _S33 + DiffTensorView_load_0(featureGrid_0, _S62) * _S31 * _S32 + DiffTensorView_load_0(featureGrid_0, _S63) * _S30 * _S32;
    *(&(&feature_0)->vals_1[int(8)]) = DiffTensorView_load_0(featureGrid_0, _S64) * _S31 * _S33 + DiffTensorView_load_0(featureGrid_0, _S65) * _S30 * _S33 + DiffTensorView_load_0(featureGrid_0, _S66) * _S31 * _S32 + DiffTensorView_load_0(featureGrid_0, _S67) * _S30 * _S32;
    *(&(&feature_0)->vals_1[int(9)]) = DiffTensorView_load_0(featureGrid_0, _S68) * _S31 * _S33 + DiffTensorView_load_0(featureGrid_0, _S69) * _S30 * _S33 + DiffTensorView_load_0(featureGrid_0, _S70) * _S31 * _S32 + DiffTensorView_load_0(featureGrid_0, _S71) * _S30 * _S32;
    *(&(&feature_0)->vals_1[int(10)]) = DiffTensorView_load_0(featureGrid_0, _S72) * _S31 * _S33 + DiffTensorView_load_0(featureGrid_0, _S73) * _S30 * _S33 + DiffTensorView_load_0(featureGrid_0, _S74) * _S31 * _S32 + DiffTensorView_load_0(featureGrid_0, _S75) * _S30 * _S32;
    *(&(&feature_0)->vals_1[int(11)]) = DiffTensorView_load_0(featureGrid_0, _S76) * _S31 * _S33 + DiffTensorView_load_0(featureGrid_0, _S77) * _S30 * _S33 + DiffTensorView_load_0(featureGrid_0, _S78) * _S31 * _S32 + DiffTensorView_load_0(featureGrid_0, _S79) * _S30 * _S32;
    *(&(&feature_0)->vals_1[int(12)]) = DiffTensorView_load_0(featureGrid_0, _S80) * _S31 * _S33 + DiffTensorView_load_0(featureGrid_0, _S81) * _S30 * _S33 + DiffTensorView_load_0(featureGrid_0, _S82) * _S31 * _S32 + DiffTensorView_load_0(featureGrid_0, _S83) * _S30 * _S32;
    *(&(&feature_0)->vals_1[int(13)]) = DiffTensorView_load_0(featureGrid_0, _S84) * _S31 * _S33 + DiffTensorView_load_0(featureGrid_0, _S85) * _S30 * _S33 + DiffTensorView_load_0(featureGrid_0, _S86) * _S31 * _S32 + DiffTensorView_load_0(featureGrid_0, _S87) * _S30 * _S32;
    *(&(&feature_0)->vals_1[14U]) = _S30;
    *(&(&feature_0)->vals_1[15U]) = _S32;
    return feature_0;
}

struct Linear_0
{
    DiffTensorView_0 weights_1;
    DiffTensorView_0 bias_0;
};

__device__ void Linear_loadArray_0(Linear_0 this_9, float * memptr_0, FixedArray<float, 16>  * input_0)
{
    int _S88 = int(uint(int(((threadIdx)).x) % int(32))) * int(16);
    *(&(*input_0)[int(0)]) = *(&memptr_0[_S88]);
    *(&(*input_0)[int(1)]) = *(&memptr_0[_S88 + int(1)]);
    *(&(*input_0)[int(2)]) = *(&memptr_0[_S88 + int(2)]);
    *(&(*input_0)[int(3)]) = *(&memptr_0[_S88 + int(3)]);
    *(&(*input_0)[int(4)]) = *(&memptr_0[_S88 + int(4)]);
    *(&(*input_0)[int(5)]) = *(&memptr_0[_S88 + int(5)]);
    *(&(*input_0)[int(6)]) = *(&memptr_0[_S88 + int(6)]);
    *(&(*input_0)[int(7)]) = *(&memptr_0[_S88 + int(7)]);
    *(&(*input_0)[int(8)]) = *(&memptr_0[_S88 + int(8)]);
    *(&(*input_0)[int(9)]) = *(&memptr_0[_S88 + int(9)]);
    *(&(*input_0)[int(10)]) = *(&memptr_0[_S88 + int(10)]);
    *(&(*input_0)[int(11)]) = *(&memptr_0[_S88 + int(11)]);
    *(&(*input_0)[int(12)]) = *(&memptr_0[_S88 + int(12)]);
    *(&(*input_0)[int(13)]) = *(&memptr_0[_S88 + int(13)]);
    *(&(*input_0)[int(14)]) = *(&memptr_0[_S88 + int(14)]);
    *(&(*input_0)[int(15)]) = *(&memptr_0[_S88 + int(15)]);
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
    int i_7 = int(0);
    for(;;)
    {
        if(i_7 < int(16))
        {
        }
        else
        {
            break;
        }
        *(&(*outputs_0)[i_7]) = *(&(*outputs_0)[i_7]) + *(&bias_1[i_7]);
        i_7 = i_7 + int(1);
    }
    return;
}

__device__ void _inline_matmul_0(float * input_1, float * weights_2, float * output_0)
{
    wmma_inline_matmul< (int(16)), (int(32)), (int(16)), (int(16)), (int(16)), (int(8)) >((input_1), (weights_2), (output_0));
    return;
}

__device__ void _inline_matmul_1(float * input_2, float * weights_3, float * output_1)
{
    wmma_inline_matmul< (int(32)), (int(16)), (int(16)), (int(16)), (int(16)), (int(8)) >((input_2), (weights_3), (output_1));
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
    bool _S89 = threadIdInWarp_0 >= int(16);
    uint _S90 = uint(threadIdInWarp_0);
    uint2  _S91 = make_uint2 (_S90, 0U);
    uint2  _S92 = make_uint2 (_S90, 1U);
    uint2  _S93 = make_uint2 (_S90, 2U);
    uint2  _S94 = make_uint2 (_S90, 3U);
    uint2  _S95 = make_uint2 (_S90, 4U);
    uint2  _S96 = make_uint2 (_S90, 5U);
    uint2  _S97 = make_uint2 (_S90, 6U);
    uint2  _S98 = make_uint2 (_S90, 7U);
    uint2  _S99 = make_uint2 (_S90, 8U);
    uint2  _S100 = make_uint2 (_S90, 9U);
    uint2  _S101 = make_uint2 (_S90, 10U);
    uint2  _S102 = make_uint2 (_S90, 11U);
    uint2  _S103 = make_uint2 (_S90, 12U);
    uint2  _S104 = make_uint2 (_S90, 13U);
    uint2  _S105 = make_uint2 (_S90, 14U);
    uint2  _S106 = make_uint2 (_S90, 15U);
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
                            if(_S89)
                            {
                                break;
                            }
                            *(&wtPtr_0[threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_1, _S91);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S89)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(16) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_1, _S92);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S89)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(32) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_1, _S93);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S89)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(48) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_1, _S94);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S89)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(64) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_1, _S95);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S89)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(80) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_1, _S96);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S89)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(96) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_1, _S97);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S89)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(112) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_1, _S98);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S89)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(128) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_1, _S99);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S89)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(144) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_1, _S100);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S89)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(160) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_1, _S101);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S89)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(176) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_1, _S102);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S89)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(192) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_1, _S103);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S89)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(208) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_1, _S104);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S89)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(224) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_1, _S105);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S89)
                            {
                                break;
                            }
                            *(&wtPtr_0[int(240) + threadIdInWarp_0]) = DiffTensorView_load_1(this_15.weights_1, _S106);
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
    bool _S107 = threadIdInWarp_1 >= int(16);
    int _S108 = threadIdInWarp_1 * int(16);
    uint _S109 = uint(threadIdInWarp_1);
    uint2  _S110 = make_uint2 (_S109, 0U);
    uint2  _S111 = make_uint2 (_S109, 1U);
    uint2  _S112 = make_uint2 (_S109, 2U);
    uint2  _S113 = make_uint2 (_S109, 3U);
    uint2  _S114 = make_uint2 (_S109, 4U);
    uint2  _S115 = make_uint2 (_S109, 5U);
    uint2  _S116 = make_uint2 (_S109, 6U);
    uint2  _S117 = make_uint2 (_S109, 7U);
    uint2  _S118 = make_uint2 (_S109, 8U);
    uint2  _S119 = make_uint2 (_S109, 9U);
    uint2  _S120 = make_uint2 (_S109, 10U);
    uint2  _S121 = make_uint2 (_S109, 11U);
    uint2  _S122 = make_uint2 (_S109, 12U);
    uint2  _S123 = make_uint2 (_S109, 13U);
    uint2  _S124 = make_uint2 (_S109, 14U);
    uint2  _S125 = make_uint2 (_S109, 15U);
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
                            if(_S107)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S108]) = DiffTensorView_load_1(this_16.weights_1, _S110);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S107)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S108 + int(1)]) = DiffTensorView_load_1(this_16.weights_1, _S111);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S107)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S108 + int(2)]) = DiffTensorView_load_1(this_16.weights_1, _S112);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S107)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S108 + int(3)]) = DiffTensorView_load_1(this_16.weights_1, _S113);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S107)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S108 + int(4)]) = DiffTensorView_load_1(this_16.weights_1, _S114);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S107)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S108 + int(5)]) = DiffTensorView_load_1(this_16.weights_1, _S115);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S107)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S108 + int(6)]) = DiffTensorView_load_1(this_16.weights_1, _S116);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S107)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S108 + int(7)]) = DiffTensorView_load_1(this_16.weights_1, _S117);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S107)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S108 + int(8)]) = DiffTensorView_load_1(this_16.weights_1, _S118);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S107)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S108 + int(9)]) = DiffTensorView_load_1(this_16.weights_1, _S119);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S107)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S108 + int(10)]) = DiffTensorView_load_1(this_16.weights_1, _S120);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S107)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S108 + int(11)]) = DiffTensorView_load_1(this_16.weights_1, _S121);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S107)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S108 + int(12)]) = DiffTensorView_load_1(this_16.weights_1, _S122);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S107)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S108 + int(13)]) = DiffTensorView_load_1(this_16.weights_1, _S123);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S107)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S108 + int(14)]) = DiffTensorView_load_1(this_16.weights_1, _S124);
                            break;
                        }
                        break;
                    }
                    for(;;)
                    {
                        for(;;)
                        {
                            if(_S107)
                            {
                                break;
                            }
                            *(&wtPtr_1[_S108 + int(15)]) = DiffTensorView_load_1(this_16.weights_1, _S125);
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

__device__ float * Linear_storeArray_0(Linear_0 this_17, float * memptr_1, FixedArray<float, 16>  input_3)
{
    int _S126 = int(uint(int(((threadIdx)).x) % int(32)));
    *(&memptr_1[_S126]) = input_3[int(0)];
    *(&memptr_1[int(32) + _S126]) = input_3[int(1)];
    *(&memptr_1[int(64) + _S126]) = input_3[int(2)];
    *(&memptr_1[int(96) + _S126]) = input_3[int(3)];
    *(&memptr_1[int(128) + _S126]) = input_3[int(4)];
    *(&memptr_1[int(160) + _S126]) = input_3[int(5)];
    *(&memptr_1[int(192) + _S126]) = input_3[int(6)];
    *(&memptr_1[int(224) + _S126]) = input_3[int(7)];
    *(&memptr_1[int(256) + _S126]) = input_3[int(8)];
    *(&memptr_1[int(288) + _S126]) = input_3[int(9)];
    *(&memptr_1[int(320) + _S126]) = input_3[int(10)];
    *(&memptr_1[int(352) + _S126]) = input_3[int(11)];
    *(&memptr_1[int(384) + _S126]) = input_3[int(12)];
    *(&memptr_1[int(416) + _S126]) = input_3[int(13)];
    *(&memptr_1[int(448) + _S126]) = input_3[int(14)];
    *(&memptr_1[int(480) + _S126]) = input_3[int(15)];
    return memptr_1;
}

__device__ float * Linear_storeArray_1(Linear_0 this_18, float * memptr_2, FixedArray<float, 16>  input_4)
{
    int _S127 = int(uint(int(((threadIdx)).x) % int(32))) * int(16);
    *(&memptr_2[_S127]) = input_4[int(0)];
    *(&memptr_2[_S127 + int(1)]) = input_4[int(1)];
    *(&memptr_2[_S127 + int(2)]) = input_4[int(2)];
    *(&memptr_2[_S127 + int(3)]) = input_4[int(3)];
    *(&memptr_2[_S127 + int(4)]) = input_4[int(4)];
    *(&memptr_2[_S127 + int(5)]) = input_4[int(5)];
    *(&memptr_2[_S127 + int(6)]) = input_4[int(6)];
    *(&memptr_2[_S127 + int(7)]) = input_4[int(7)];
    *(&memptr_2[_S127 + int(8)]) = input_4[int(8)];
    *(&memptr_2[_S127 + int(9)]) = input_4[int(9)];
    *(&memptr_2[_S127 + int(10)]) = input_4[int(10)];
    *(&memptr_2[_S127 + int(11)]) = input_4[int(11)];
    *(&memptr_2[_S127 + int(12)]) = input_4[int(12)];
    *(&memptr_2[_S127 + int(13)]) = input_4[int(13)];
    *(&memptr_2[_S127 + int(14)]) = input_4[int(14)];
    *(&memptr_2[_S127 + int(15)]) = input_4[int(15)];
    return memptr_2;
}

__device__ float * Linear_inpBufferForCurrentWarp_0(Linear_0 this_19)
{
    return &(*(&shared_inputs_buffer_0))[Linear_calcOffset_0(this_19)];
}

__device__ float * Linear_moveInputsToSharedMem_0(Linear_0 this_20, FixedArray<float, 16>  input_5)
{
    float * inPtr_0 = Linear_inpBufferForCurrentWarp_0(this_20);
    float * _S128 = Linear_storeArray_1(this_20, inPtr_0, input_5);
    return _S128;
}

__device__ float * Linear_moveDOutputsToSharedMem_0(Linear_0 this_21, FixedArray<float, 16>  d_output_0)
{
    float * outPtr_1 = Linear_outBufferForCurrentWarp_0(this_21);
    float * _S129 = Linear_storeArray_0(this_21, outPtr_1, d_output_0);
    return _S129;
}

__device__ float * Linear_moveDInputsToSharedMem_0(Linear_0 this_22, FixedArray<float, 16>  input_6)
{
    float * inPtr_1 = Linear_inpBufferForCurrentWarp_0(this_22);
    float * _S130 = Linear_storeArray_0(this_22, inPtr_1, input_6);
    return _S130;
}

__device__ uint WaveGetActiveMask_0(uint _S131)
{
    return _S131;
}

__device__ float WaveActiveSum_0(float expr_0, uint _S132)
{
    float _S133 = (_waveSum((WaveGetActiveMask_0(_S132)), (expr_0)));
    return _S133;
}

__device__ bool WaveIsFirstLane_0(uint _S134)
{
    uint _S135 = WaveGetActiveMask_0(_S134);
    bool _S136 = ((((_S135) & -(_S135)) == (WarpMask(1) << _getLaneId())));
    return _S136;
}

struct DiffPair_Feature_0
{
    Feature_0 primal_1;
    s_diff_Feature_0 differential_0;
};

__device__ void Linear_eval_bwd_0(Linear_0 this_23, DiffPair_Feature_0 * in_feature_pair_0, s_diff_Feature_0 d_output_1, uint _S137)
{
    uint _S138 = 0U;
    float * dOutPtr_0 = Linear_moveInputsToSharedMem_0(this_23, d_output_1.vals_0);
    float * wtPtr_2 = Linear_moveWeightsToSharedMem_1(this_23);
    float * dInPtr_0 = Linear_outBufferForCurrentWarp_0(this_23);
    _inline_matmul_1(dOutPtr_0, wtPtr_2, dInPtr_0);
    s_diff_Feature_0 d_input_feature_0;
    Linear_loadArray_0(this_23, dInPtr_0, &(&d_input_feature_0)->vals_0);
    DiffPair_Feature_0 _S139 = *in_feature_pair_0;
    in_feature_pair_0->primal_1 = (*in_feature_pair_0).primal_1;
    in_feature_pair_0->differential_0 = d_input_feature_0;
    float * inPtr_2 = Linear_moveDInputsToSharedMem_0(this_23, _S139.primal_1.vals_1);
    float * outPtr_2 = Linear_moveDOutputsToSharedMem_0(this_23, d_output_1.vals_0);
    float * wtPtr_3 = Linear_wtBufferForCurrentWarp_0(this_23);
    _inline_matmul_0(outPtr_2, inPtr_2, wtPtr_3);
    int threadIdInWarp_2 = int(((threadIdx)).x) % int(32);
    bool _S140 = threadIdInWarp_2 >= int(16);
    uint _S141 = uint(threadIdInWarp_2);
    int _S142 = threadIdInWarp_2 * int(16);
    uint2  _S143 = make_uint2 (0U, _S141);
    uint2  _S144 = make_uint2 (1U, _S141);
    uint2  _S145 = make_uint2 (2U, _S141);
    uint2  _S146 = make_uint2 (3U, _S141);
    uint2  _S147 = make_uint2 (4U, _S141);
    uint2  _S148 = make_uint2 (5U, _S141);
    uint2  _S149 = make_uint2 (6U, _S141);
    uint2  _S150 = make_uint2 (7U, _S141);
    uint2  _S151 = make_uint2 (8U, _S141);
    uint2  _S152 = make_uint2 (9U, _S141);
    uint2  _S153 = make_uint2 (10U, _S141);
    uint2  _S154 = make_uint2 (11U, _S141);
    uint2  _S155 = make_uint2 (12U, _S141);
    uint2  _S156 = make_uint2 (13U, _S141);
    uint2  _S157 = make_uint2 (14U, _S141);
    uint2  _S158 = make_uint2 (15U, _S141);
    for(;;)
    {
        uint _S159 = 0U;
        for(;;)
        {
            for(;;)
            {
                for(;;)
                {
                    uint _S160 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S161 = __ballot_sync(_S137, _S140);
                            if(_S140)
                            {
                                uint _S162 = __ballot_sync(_S137, false);
                                uint _S163 = __ballot_sync(_S137, true);
                                break;
                            }
                            else
                            {
                                uint _S164 = __ballot_sync(_S137, true);
                            }
                            float oldVal_1;
                            *((&oldVal_1)) = atomicAdd((this_23.weights_1.diff_1.diff_0).data_ptr_at<float>((_S143)), (*(&wtPtr_3[_S142])));
                            uint _S165 = __ballot_sync(_S137, true);
                            break;
                        }
                        uint _S166 = __ballot_sync(_S137, true);
                        _S160 = _S166;
                        break;
                    }
                    uint _S167 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S168 = __ballot_sync(_S160, _S140);
                            if(_S140)
                            {
                                uint _S169 = __ballot_sync(_S160, false);
                                uint _S170 = __ballot_sync(_S160, true);
                                break;
                            }
                            else
                            {
                                uint _S171 = __ballot_sync(_S160, true);
                            }
                            float oldVal_2;
                            *((&oldVal_2)) = atomicAdd((this_23.weights_1.diff_1.diff_0).data_ptr_at<float>((_S144)), (*(&wtPtr_3[_S142 + int(1)])));
                            uint _S172 = __ballot_sync(_S160, true);
                            break;
                        }
                        uint _S173 = __ballot_sync(_S160, true);
                        _S167 = _S173;
                        break;
                    }
                    uint _S174 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S175 = __ballot_sync(_S167, _S140);
                            if(_S140)
                            {
                                uint _S176 = __ballot_sync(_S167, false);
                                uint _S177 = __ballot_sync(_S167, true);
                                break;
                            }
                            else
                            {
                                uint _S178 = __ballot_sync(_S167, true);
                            }
                            float oldVal_3;
                            *((&oldVal_3)) = atomicAdd((this_23.weights_1.diff_1.diff_0).data_ptr_at<float>((_S145)), (*(&wtPtr_3[_S142 + int(2)])));
                            uint _S179 = __ballot_sync(_S167, true);
                            break;
                        }
                        uint _S180 = __ballot_sync(_S167, true);
                        _S174 = _S180;
                        break;
                    }
                    uint _S181 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S182 = __ballot_sync(_S174, _S140);
                            if(_S140)
                            {
                                uint _S183 = __ballot_sync(_S174, false);
                                uint _S184 = __ballot_sync(_S174, true);
                                break;
                            }
                            else
                            {
                                uint _S185 = __ballot_sync(_S174, true);
                            }
                            float oldVal_4;
                            *((&oldVal_4)) = atomicAdd((this_23.weights_1.diff_1.diff_0).data_ptr_at<float>((_S146)), (*(&wtPtr_3[_S142 + int(3)])));
                            uint _S186 = __ballot_sync(_S174, true);
                            break;
                        }
                        uint _S187 = __ballot_sync(_S174, true);
                        _S181 = _S187;
                        break;
                    }
                    uint _S188 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S189 = __ballot_sync(_S181, _S140);
                            if(_S140)
                            {
                                uint _S190 = __ballot_sync(_S181, false);
                                uint _S191 = __ballot_sync(_S181, true);
                                break;
                            }
                            else
                            {
                                uint _S192 = __ballot_sync(_S181, true);
                            }
                            float oldVal_5;
                            *((&oldVal_5)) = atomicAdd((this_23.weights_1.diff_1.diff_0).data_ptr_at<float>((_S147)), (*(&wtPtr_3[_S142 + int(4)])));
                            uint _S193 = __ballot_sync(_S181, true);
                            break;
                        }
                        uint _S194 = __ballot_sync(_S181, true);
                        _S188 = _S194;
                        break;
                    }
                    uint _S195 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S196 = __ballot_sync(_S188, _S140);
                            if(_S140)
                            {
                                uint _S197 = __ballot_sync(_S188, false);
                                uint _S198 = __ballot_sync(_S188, true);
                                break;
                            }
                            else
                            {
                                uint _S199 = __ballot_sync(_S188, true);
                            }
                            float oldVal_6;
                            *((&oldVal_6)) = atomicAdd((this_23.weights_1.diff_1.diff_0).data_ptr_at<float>((_S148)), (*(&wtPtr_3[_S142 + int(5)])));
                            uint _S200 = __ballot_sync(_S188, true);
                            break;
                        }
                        uint _S201 = __ballot_sync(_S188, true);
                        _S195 = _S201;
                        break;
                    }
                    uint _S202 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S203 = __ballot_sync(_S195, _S140);
                            if(_S140)
                            {
                                uint _S204 = __ballot_sync(_S195, false);
                                uint _S205 = __ballot_sync(_S195, true);
                                break;
                            }
                            else
                            {
                                uint _S206 = __ballot_sync(_S195, true);
                            }
                            float oldVal_7;
                            *((&oldVal_7)) = atomicAdd((this_23.weights_1.diff_1.diff_0).data_ptr_at<float>((_S149)), (*(&wtPtr_3[_S142 + int(6)])));
                            uint _S207 = __ballot_sync(_S195, true);
                            break;
                        }
                        uint _S208 = __ballot_sync(_S195, true);
                        _S202 = _S208;
                        break;
                    }
                    uint _S209 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S210 = __ballot_sync(_S202, _S140);
                            if(_S140)
                            {
                                uint _S211 = __ballot_sync(_S202, false);
                                uint _S212 = __ballot_sync(_S202, true);
                                break;
                            }
                            else
                            {
                                uint _S213 = __ballot_sync(_S202, true);
                            }
                            float oldVal_8;
                            *((&oldVal_8)) = atomicAdd((this_23.weights_1.diff_1.diff_0).data_ptr_at<float>((_S150)), (*(&wtPtr_3[_S142 + int(7)])));
                            uint _S214 = __ballot_sync(_S202, true);
                            break;
                        }
                        uint _S215 = __ballot_sync(_S202, true);
                        _S209 = _S215;
                        break;
                    }
                    uint _S216 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S217 = __ballot_sync(_S209, _S140);
                            if(_S140)
                            {
                                uint _S218 = __ballot_sync(_S209, false);
                                uint _S219 = __ballot_sync(_S209, true);
                                break;
                            }
                            else
                            {
                                uint _S220 = __ballot_sync(_S209, true);
                            }
                            float oldVal_9;
                            *((&oldVal_9)) = atomicAdd((this_23.weights_1.diff_1.diff_0).data_ptr_at<float>((_S151)), (*(&wtPtr_3[_S142 + int(8)])));
                            uint _S221 = __ballot_sync(_S209, true);
                            break;
                        }
                        uint _S222 = __ballot_sync(_S209, true);
                        _S216 = _S222;
                        break;
                    }
                    uint _S223 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S224 = __ballot_sync(_S216, _S140);
                            if(_S140)
                            {
                                uint _S225 = __ballot_sync(_S216, false);
                                uint _S226 = __ballot_sync(_S216, true);
                                break;
                            }
                            else
                            {
                                uint _S227 = __ballot_sync(_S216, true);
                            }
                            float oldVal_10;
                            *((&oldVal_10)) = atomicAdd((this_23.weights_1.diff_1.diff_0).data_ptr_at<float>((_S152)), (*(&wtPtr_3[_S142 + int(9)])));
                            uint _S228 = __ballot_sync(_S216, true);
                            break;
                        }
                        uint _S229 = __ballot_sync(_S216, true);
                        _S223 = _S229;
                        break;
                    }
                    uint _S230 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S231 = __ballot_sync(_S223, _S140);
                            if(_S140)
                            {
                                uint _S232 = __ballot_sync(_S223, false);
                                uint _S233 = __ballot_sync(_S223, true);
                                break;
                            }
                            else
                            {
                                uint _S234 = __ballot_sync(_S223, true);
                            }
                            float oldVal_11;
                            *((&oldVal_11)) = atomicAdd((this_23.weights_1.diff_1.diff_0).data_ptr_at<float>((_S153)), (*(&wtPtr_3[_S142 + int(10)])));
                            uint _S235 = __ballot_sync(_S223, true);
                            break;
                        }
                        uint _S236 = __ballot_sync(_S223, true);
                        _S230 = _S236;
                        break;
                    }
                    uint _S237 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S238 = __ballot_sync(_S230, _S140);
                            if(_S140)
                            {
                                uint _S239 = __ballot_sync(_S230, false);
                                uint _S240 = __ballot_sync(_S230, true);
                                break;
                            }
                            else
                            {
                                uint _S241 = __ballot_sync(_S230, true);
                            }
                            float oldVal_12;
                            *((&oldVal_12)) = atomicAdd((this_23.weights_1.diff_1.diff_0).data_ptr_at<float>((_S154)), (*(&wtPtr_3[_S142 + int(11)])));
                            uint _S242 = __ballot_sync(_S230, true);
                            break;
                        }
                        uint _S243 = __ballot_sync(_S230, true);
                        _S237 = _S243;
                        break;
                    }
                    uint _S244 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S245 = __ballot_sync(_S237, _S140);
                            if(_S140)
                            {
                                uint _S246 = __ballot_sync(_S237, false);
                                uint _S247 = __ballot_sync(_S237, true);
                                break;
                            }
                            else
                            {
                                uint _S248 = __ballot_sync(_S237, true);
                            }
                            float oldVal_13;
                            *((&oldVal_13)) = atomicAdd((this_23.weights_1.diff_1.diff_0).data_ptr_at<float>((_S155)), (*(&wtPtr_3[_S142 + int(12)])));
                            uint _S249 = __ballot_sync(_S237, true);
                            break;
                        }
                        uint _S250 = __ballot_sync(_S237, true);
                        _S244 = _S250;
                        break;
                    }
                    uint _S251 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S252 = __ballot_sync(_S244, _S140);
                            if(_S140)
                            {
                                uint _S253 = __ballot_sync(_S244, false);
                                uint _S254 = __ballot_sync(_S244, true);
                                break;
                            }
                            else
                            {
                                uint _S255 = __ballot_sync(_S244, true);
                            }
                            float oldVal_14;
                            *((&oldVal_14)) = atomicAdd((this_23.weights_1.diff_1.diff_0).data_ptr_at<float>((_S156)), (*(&wtPtr_3[_S142 + int(13)])));
                            uint _S256 = __ballot_sync(_S244, true);
                            break;
                        }
                        uint _S257 = __ballot_sync(_S244, true);
                        _S251 = _S257;
                        break;
                    }
                    uint _S258 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S259 = __ballot_sync(_S251, _S140);
                            if(_S140)
                            {
                                uint _S260 = __ballot_sync(_S251, false);
                                uint _S261 = __ballot_sync(_S251, true);
                                break;
                            }
                            else
                            {
                                uint _S262 = __ballot_sync(_S251, true);
                            }
                            float oldVal_15;
                            *((&oldVal_15)) = atomicAdd((this_23.weights_1.diff_1.diff_0).data_ptr_at<float>((_S157)), (*(&wtPtr_3[_S142 + int(14)])));
                            uint _S263 = __ballot_sync(_S251, true);
                            break;
                        }
                        uint _S264 = __ballot_sync(_S251, true);
                        _S258 = _S264;
                        break;
                    }
                    uint _S265 = 0U;
                    for(;;)
                    {
                        for(;;)
                        {
                            uint _S266 = __ballot_sync(_S258, _S140);
                            if(_S140)
                            {
                                uint _S267 = __ballot_sync(_S258, false);
                                uint _S268 = __ballot_sync(_S258, true);
                                break;
                            }
                            else
                            {
                                uint _S269 = __ballot_sync(_S258, true);
                            }
                            float oldVal_16;
                            *((&oldVal_16)) = atomicAdd((this_23.weights_1.diff_1.diff_0).data_ptr_at<float>((_S158)), (*(&wtPtr_3[_S142 + int(15)])));
                            uint _S270 = __ballot_sync(_S258, true);
                            break;
                        }
                        uint _S271 = __ballot_sync(_S258, true);
                        _S265 = _S271;
                        break;
                    }
                    uint _S272 = __ballot_sync(_S265, false);
                    uint _S273 = __ballot_sync(_S265, false);
                    uint _S274 = __ballot_sync(_S137, true);
                    break;
                }
                uint _S275 = __ballot_sync(_S137, true);
                break;
            }
            uint _S276 = __ballot_sync(_S137, true);
            _S159 = _S276;
            break;
        }
        uint _S277 = __ballot_sync(_S159, false);
        uint _S278 = __ballot_sync(_S159, false);
        uint _S279 = __ballot_sync(_S137, true);
        _S138 = _S279;
        break;
    }
    float _S280 = WaveActiveSum_0(d_output_1.vals_0[int(0)], _S138);
    bool _S281 = WaveIsFirstLane_0(_S138);
    uint _S282 = __ballot_sync(_S138, _S281);
    uint _S283;
    if(_S281)
    {
        float oldVal_17;
        *((&oldVal_17)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((0U)), (_S280));
        uint _S284 = __ballot_sync(_S138, true);
        _S283 = _S284;
    }
    else
    {
        uint _S285 = __ballot_sync(_S138, true);
        _S283 = _S285;
    }
    float _S286 = WaveActiveSum_0(d_output_1.vals_0[int(1)], _S283);
    bool _S287 = WaveIsFirstLane_0(_S283);
    uint _S288 = __ballot_sync(_S283, _S287);
    if(_S287)
    {
        float oldVal_18;
        *((&oldVal_18)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((1U)), (_S286));
        uint _S289 = __ballot_sync(_S283, true);
        _S283 = _S289;
    }
    else
    {
        uint _S290 = __ballot_sync(_S283, true);
        _S283 = _S290;
    }
    float _S291 = WaveActiveSum_0(d_output_1.vals_0[int(2)], _S283);
    bool _S292 = WaveIsFirstLane_0(_S283);
    uint _S293 = __ballot_sync(_S283, _S292);
    if(_S292)
    {
        float oldVal_19;
        *((&oldVal_19)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((2U)), (_S291));
        uint _S294 = __ballot_sync(_S283, true);
        _S283 = _S294;
    }
    else
    {
        uint _S295 = __ballot_sync(_S283, true);
        _S283 = _S295;
    }
    float _S296 = WaveActiveSum_0(d_output_1.vals_0[int(3)], _S283);
    bool _S297 = WaveIsFirstLane_0(_S283);
    uint _S298 = __ballot_sync(_S283, _S297);
    if(_S297)
    {
        float oldVal_20;
        *((&oldVal_20)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((3U)), (_S296));
        uint _S299 = __ballot_sync(_S283, true);
        _S283 = _S299;
    }
    else
    {
        uint _S300 = __ballot_sync(_S283, true);
        _S283 = _S300;
    }
    float _S301 = WaveActiveSum_0(d_output_1.vals_0[int(4)], _S283);
    bool _S302 = WaveIsFirstLane_0(_S283);
    uint _S303 = __ballot_sync(_S283, _S302);
    if(_S302)
    {
        float oldVal_21;
        *((&oldVal_21)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((4U)), (_S301));
        uint _S304 = __ballot_sync(_S283, true);
        _S283 = _S304;
    }
    else
    {
        uint _S305 = __ballot_sync(_S283, true);
        _S283 = _S305;
    }
    float _S306 = WaveActiveSum_0(d_output_1.vals_0[int(5)], _S283);
    bool _S307 = WaveIsFirstLane_0(_S283);
    uint _S308 = __ballot_sync(_S283, _S307);
    if(_S307)
    {
        float oldVal_22;
        *((&oldVal_22)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((5U)), (_S306));
        uint _S309 = __ballot_sync(_S283, true);
        _S283 = _S309;
    }
    else
    {
        uint _S310 = __ballot_sync(_S283, true);
        _S283 = _S310;
    }
    float _S311 = WaveActiveSum_0(d_output_1.vals_0[int(6)], _S283);
    bool _S312 = WaveIsFirstLane_0(_S283);
    uint _S313 = __ballot_sync(_S283, _S312);
    if(_S312)
    {
        float oldVal_23;
        *((&oldVal_23)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((6U)), (_S311));
        uint _S314 = __ballot_sync(_S283, true);
        _S283 = _S314;
    }
    else
    {
        uint _S315 = __ballot_sync(_S283, true);
        _S283 = _S315;
    }
    float _S316 = WaveActiveSum_0(d_output_1.vals_0[int(7)], _S283);
    bool _S317 = WaveIsFirstLane_0(_S283);
    uint _S318 = __ballot_sync(_S283, _S317);
    if(_S317)
    {
        float oldVal_24;
        *((&oldVal_24)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((7U)), (_S316));
        uint _S319 = __ballot_sync(_S283, true);
        _S283 = _S319;
    }
    else
    {
        uint _S320 = __ballot_sync(_S283, true);
        _S283 = _S320;
    }
    float _S321 = WaveActiveSum_0(d_output_1.vals_0[int(8)], _S283);
    bool _S322 = WaveIsFirstLane_0(_S283);
    uint _S323 = __ballot_sync(_S283, _S322);
    if(_S322)
    {
        float oldVal_25;
        *((&oldVal_25)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((8U)), (_S321));
        uint _S324 = __ballot_sync(_S283, true);
        _S283 = _S324;
    }
    else
    {
        uint _S325 = __ballot_sync(_S283, true);
        _S283 = _S325;
    }
    float _S326 = WaveActiveSum_0(d_output_1.vals_0[int(9)], _S283);
    bool _S327 = WaveIsFirstLane_0(_S283);
    uint _S328 = __ballot_sync(_S283, _S327);
    if(_S327)
    {
        float oldVal_26;
        *((&oldVal_26)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((9U)), (_S326));
        uint _S329 = __ballot_sync(_S283, true);
        _S283 = _S329;
    }
    else
    {
        uint _S330 = __ballot_sync(_S283, true);
        _S283 = _S330;
    }
    float _S331 = WaveActiveSum_0(d_output_1.vals_0[int(10)], _S283);
    bool _S332 = WaveIsFirstLane_0(_S283);
    uint _S333 = __ballot_sync(_S283, _S332);
    if(_S332)
    {
        float oldVal_27;
        *((&oldVal_27)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((10U)), (_S331));
        uint _S334 = __ballot_sync(_S283, true);
        _S283 = _S334;
    }
    else
    {
        uint _S335 = __ballot_sync(_S283, true);
        _S283 = _S335;
    }
    float _S336 = WaveActiveSum_0(d_output_1.vals_0[int(11)], _S283);
    bool _S337 = WaveIsFirstLane_0(_S283);
    uint _S338 = __ballot_sync(_S283, _S337);
    if(_S337)
    {
        float oldVal_28;
        *((&oldVal_28)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((11U)), (_S336));
        uint _S339 = __ballot_sync(_S283, true);
        _S283 = _S339;
    }
    else
    {
        uint _S340 = __ballot_sync(_S283, true);
        _S283 = _S340;
    }
    float _S341 = WaveActiveSum_0(d_output_1.vals_0[int(12)], _S283);
    bool _S342 = WaveIsFirstLane_0(_S283);
    uint _S343 = __ballot_sync(_S283, _S342);
    if(_S342)
    {
        float oldVal_29;
        *((&oldVal_29)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((12U)), (_S341));
        uint _S344 = __ballot_sync(_S283, true);
        _S283 = _S344;
    }
    else
    {
        uint _S345 = __ballot_sync(_S283, true);
        _S283 = _S345;
    }
    float _S346 = WaveActiveSum_0(d_output_1.vals_0[int(13)], _S283);
    bool _S347 = WaveIsFirstLane_0(_S283);
    uint _S348 = __ballot_sync(_S283, _S347);
    if(_S347)
    {
        float oldVal_30;
        *((&oldVal_30)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((13U)), (_S346));
        uint _S349 = __ballot_sync(_S283, true);
        _S283 = _S349;
    }
    else
    {
        uint _S350 = __ballot_sync(_S283, true);
        _S283 = _S350;
    }
    float _S351 = WaveActiveSum_0(d_output_1.vals_0[int(14)], _S283);
    bool _S352 = WaveIsFirstLane_0(_S283);
    uint _S353 = __ballot_sync(_S283, _S352);
    if(_S352)
    {
        float oldVal_31;
        *((&oldVal_31)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((14U)), (_S351));
        uint _S354 = __ballot_sync(_S283, true);
        _S283 = _S354;
    }
    else
    {
        uint _S355 = __ballot_sync(_S283, true);
        _S283 = _S355;
    }
    float _S356 = WaveActiveSum_0(d_output_1.vals_0[int(15)], _S283);
    bool _S357 = WaveIsFirstLane_0(_S283);
    uint _S358 = __ballot_sync(_S283, _S357);
    if(_S357)
    {
        float oldVal_32;
        *((&oldVal_32)) = atomicAdd((this_23.bias_0.diff_1.diff_0).data_ptr_at<float>((15U)), (_S356));
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
    float * _S359 = ((this_24.bias_0.primal_0).data_ptr<float>());
    Linear_moveOutputsToLocalArray_0(this_24, &(&out_feature_0)->vals_1, _S359);
    return out_feature_0;
}

__device__ void _d_max_0(DiffPair_float_0 * dpx_0, DiffPair_float_0 * dpy_0, float dOut_2)
{
    DiffPair_float_0 _S360 = *dpx_0;
    float _S361;
    if((*dpx_0).primal_1 > (*dpy_0).primal_1)
    {
        _S361 = dOut_2;
    }
    else
    {
        _S361 = 0.0f;
    }
    dpx_0->primal_1 = _S360.primal_1;
    dpx_0->differential_0 = _S361;
    DiffPair_float_0 _S362 = *dpy_0;
    if((*dpy_0).primal_1 > _S360.primal_1)
    {
        _S361 = dOut_2;
    }
    else
    {
        _S361 = 0.0f;
    }
    dpy_0->primal_1 = _S362.primal_1;
    dpy_0->differential_0 = _S361;
    return;
}

__device__ DiffPair_float_0 _d_max_1(DiffPair_float_0 dpx_1, DiffPair_float_0 dpy_1)
{
    float _S363 = (F32_max((dpx_1.primal_1), (dpy_1.primal_1)));
    float _S364;
    if(dpx_1.primal_1 > dpy_1.primal_1)
    {
        _S364 = dpx_1.differential_0;
    }
    else
    {
        _S364 = dpy_1.differential_0;
    }
    DiffPair_float_0 _S365 = { _S363, _S364 };
    return _S365;
}

__device__ float s_primal_ctx_max_0(float _S366, float _S367)
{
    return (F32_max((_S366), (_S367)));
}

__device__ void s_bwd_prop_max_0(DiffPair_float_0 * _S368, DiffPair_float_0 * _S369, float _S370)
{
    _d_max_0(_S368, _S369, _S370);
    return;
}

struct MLP_0
{
    FixedArray<Linear_0, 3>  layers_0;
};

__device__ Feature_0 MLP_eval_0(MLP_0 this_25, Feature_0 in_feature_1)
{
    Feature_0 out_feature_1;
    Feature_0 _S371 = Linear_eval_0(this_25.layers_0[int(0)], in_feature_1);
    out_feature_1 = _S371;
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
    Feature_0 _S372 = Linear_eval_0(this_25.layers_0[int(1)], out_feature_1);
    out_feature_1 = _S372;
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
    Feature_0 _S373 = Linear_eval_0(this_25.layers_0[int(2)], out_feature_1);
    out_feature_1 = _S373;
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

__device__ void DiffTensorView_storeOnce_forward_0(DiffTensorView_0 this_26, uint3  x_2, DiffPair_float_0 dpval_0)
{
    (this_26.primal_0).store<float>((x_2), (dpval_0.primal_1));
    AtomicAdd_storeOnce_forward_0(this_26.diff_1, x_2, dpval_0.differential_0);
    return;
}

__device__ void DiffTensorView_storeOnce_backward_0(DiffTensorView_0 this_27, uint3  x_3, DiffPair_float_0 * dpval_1)
{
    float _S374 = AtomicAdd_storeOnce_backward_0(this_27.diff_1, x_3);
    dpval_1->primal_1 = (*dpval_1).primal_1;
    dpval_1->differential_0 = _S374;
    return;
}

__device__ void DiffTensorView_storeOnce_0(DiffTensorView_0 this_28, uint3  x_4, float val_0)
{
    (this_28.primal_0).store<float>((x_4), (val_0));
    return;
}

struct s_bwd_prop_computeInterpolatedFeature_Intermediates_0
{
    uint _S375;
    uint _S376;
    float _S377;
    float _S378;
    float _S379;
    float _S380;
    float _S381;
    float _S382;
    float _S383;
    float _S384;
    float _S385;
    float _S386;
    float _S387;
    float _S388;
    float _S389;
    float _S390;
    float _S391;
    float _S392;
    float _S393;
    float _S394;
    float _S395;
    float _S396;
    float _S397;
    float _S398;
    float _S399;
    float _S400;
    float _S401;
    float _S402;
    float _S403;
    float _S404;
    float _S405;
    float _S406;
    float _S407;
    float _S408;
    float _S409;
    float _S410;
    float _S411;
    float _S412;
    float _S413;
    float _S414;
    float _S415;
    float _S416;
    float _S417;
    float _S418;
    float _S419;
    float _S420;
    float _S421;
    float _S422;
    float _S423;
    float _S424;
    float _S425;
    float _S426;
    float _S427;
    float _S428;
    float _S429;
    float _S430;
    float _S431;
    float _S432;
};

struct s_bwd_prop_MLP_eval_Intermediates_0
{
    Feature_0 _S433;
    Feature_0 _S434;
    Feature_0 _S435;
};

struct s_bwd_prop_renderImage_Intermediates_0
{
    s_bwd_prop_computeInterpolatedFeature_Intermediates_0 _S436;
    Feature_0 _S437;
    s_bwd_prop_MLP_eval_Intermediates_0 _S438;
    Feature_0 _S439;
    uint _S440;
    uint _S441;
};

__device__ float s_primal_ctx_DiffTensorView_load_0(DiffTensorView_0 _S442, uint3  _S443)
{
    return DiffTensorView_load_0(_S442, _S443);
}

__device__ Feature_0 s_bwd_primal_computeInterpolatedFeature_0(DiffTensorView_0 featureGrid_1, uint2  frameDim_1, s_bwd_prop_computeInterpolatedFeature_Intermediates_0 * _s_diff_ctx_0)
{
    _s_diff_ctx_0->_S375 = 0U;
    _s_diff_ctx_0->_S376 = 0U;
    _s_diff_ctx_0->_S377 = 0.0f;
    _s_diff_ctx_0->_S378 = 0.0f;
    _s_diff_ctx_0->_S379 = 0.0f;
    _s_diff_ctx_0->_S380 = 0.0f;
    _s_diff_ctx_0->_S381 = 0.0f;
    _s_diff_ctx_0->_S382 = 0.0f;
    _s_diff_ctx_0->_S383 = 0.0f;
    _s_diff_ctx_0->_S384 = 0.0f;
    _s_diff_ctx_0->_S385 = 0.0f;
    _s_diff_ctx_0->_S386 = 0.0f;
    _s_diff_ctx_0->_S387 = 0.0f;
    _s_diff_ctx_0->_S388 = 0.0f;
    _s_diff_ctx_0->_S389 = 0.0f;
    _s_diff_ctx_0->_S390 = 0.0f;
    _s_diff_ctx_0->_S391 = 0.0f;
    _s_diff_ctx_0->_S392 = 0.0f;
    _s_diff_ctx_0->_S393 = 0.0f;
    _s_diff_ctx_0->_S394 = 0.0f;
    _s_diff_ctx_0->_S395 = 0.0f;
    _s_diff_ctx_0->_S396 = 0.0f;
    _s_diff_ctx_0->_S397 = 0.0f;
    _s_diff_ctx_0->_S398 = 0.0f;
    _s_diff_ctx_0->_S399 = 0.0f;
    _s_diff_ctx_0->_S400 = 0.0f;
    _s_diff_ctx_0->_S401 = 0.0f;
    _s_diff_ctx_0->_S402 = 0.0f;
    _s_diff_ctx_0->_S403 = 0.0f;
    _s_diff_ctx_0->_S404 = 0.0f;
    _s_diff_ctx_0->_S405 = 0.0f;
    _s_diff_ctx_0->_S406 = 0.0f;
    _s_diff_ctx_0->_S407 = 0.0f;
    _s_diff_ctx_0->_S408 = 0.0f;
    _s_diff_ctx_0->_S409 = 0.0f;
    _s_diff_ctx_0->_S410 = 0.0f;
    _s_diff_ctx_0->_S411 = 0.0f;
    _s_diff_ctx_0->_S412 = 0.0f;
    _s_diff_ctx_0->_S413 = 0.0f;
    _s_diff_ctx_0->_S414 = 0.0f;
    _s_diff_ctx_0->_S415 = 0.0f;
    _s_diff_ctx_0->_S416 = 0.0f;
    _s_diff_ctx_0->_S417 = 0.0f;
    _s_diff_ctx_0->_S418 = 0.0f;
    _s_diff_ctx_0->_S419 = 0.0f;
    _s_diff_ctx_0->_S420 = 0.0f;
    _s_diff_ctx_0->_S421 = 0.0f;
    _s_diff_ctx_0->_S422 = 0.0f;
    _s_diff_ctx_0->_S423 = 0.0f;
    _s_diff_ctx_0->_S424 = 0.0f;
    _s_diff_ctx_0->_S425 = 0.0f;
    _s_diff_ctx_0->_S426 = 0.0f;
    _s_diff_ctx_0->_S427 = 0.0f;
    _s_diff_ctx_0->_S428 = 0.0f;
    _s_diff_ctx_0->_S429 = 0.0f;
    _s_diff_ctx_0->_S430 = 0.0f;
    _s_diff_ctx_0->_S431 = 0.0f;
    _s_diff_ctx_0->_S432 = 0.0f;
    uint _S444 = DiffTensorView_size_0(featureGrid_1, 0U);
    _s_diff_ctx_0->_S375 = _S444;
    uint dim0_0 = _S444 - 1U;
    uint _S445 = DiffTensorView_size_0(featureGrid_1, 1U);
    _s_diff_ctx_0->_S376 = _S445;
    uint dim1_1 = _S445 - 1U;
    uint _S446 = frameDim_1.x / dim0_0;
    uint _S447 = frameDim_1.y / dim1_1;
    uint2  cellSize_1 = make_uint2 (_S446, _S447);
    uint2  pixelId_1 = uint2 {(((threadIdx)) + ((blockIdx)) * ((blockDim))).x, (((threadIdx)) + ((blockIdx)) * ((blockDim))).y};
    uint2  cellId_1 = pixelId_1 / cellSize_1;
    float2  _S448 = make_float2 ((float)pixelId_1.x, (float)pixelId_1.y);
    float2  _S449 = make_float2 ((float)cellSize_1.x, (float)cellSize_1.y);
    float2  _S450 = make_float2 ((float)cellId_1.x, (float)cellId_1.y);
    float2  weights_4 = _S448 / _S449 - _S450;
    FixedArray<float, 16>  _S451 = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    uint _S452 = cellId_1.x;
    uint _S453 = cellId_1.y;
    float _S454 = weights_4.x;
    float _S455 = 1.0f - _S454;
    float _S456 = weights_4.y;
    float _S457 = 1.0f - _S456;
    uint _S458 = _S452 + 1U;
    uint _S459 = _S453 + 1U;
    float _S460 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S453, 0U));
    _s_diff_ctx_0->_S377 = _S460;
    float _S461 = _S460 * _S455 * _S457;
    float _S462 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S453, 0U));
    _s_diff_ctx_0->_S378 = _S462;
    float _S463 = _S461 + _S462 * _S454 * _S457;
    float _S464 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S459, 0U));
    _s_diff_ctx_0->_S379 = _S464;
    float _S465 = _S463 + _S464 * _S455 * _S456;
    float _S466 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S459, 0U));
    _s_diff_ctx_0->_S380 = _S466;
    float _S467 = _S465 + _S466 * _S454 * _S456;
    Feature_0 _S468;
    (&_S468)->vals_1 = _S451;
    *(&(&_S468)->vals_1[int(0)]) = _S467;
    float _S469 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S453, 1U));
    _s_diff_ctx_0->_S381 = _S469;
    float _S470 = _S469 * _S455 * _S457;
    float _S471 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S453, 1U));
    _s_diff_ctx_0->_S382 = _S471;
    float _S472 = _S470 + _S471 * _S454 * _S457;
    float _S473 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S459, 1U));
    _s_diff_ctx_0->_S383 = _S473;
    float _S474 = _S472 + _S473 * _S455 * _S456;
    float _S475 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S459, 1U));
    _s_diff_ctx_0->_S384 = _S475;
    *(&(&_S468)->vals_1[int(1)]) = _S474 + _S475 * _S454 * _S456;
    float _S476 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S453, 2U));
    _s_diff_ctx_0->_S385 = _S476;
    float _S477 = _S476 * _S455 * _S457;
    float _S478 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S453, 2U));
    _s_diff_ctx_0->_S386 = _S478;
    float _S479 = _S477 + _S478 * _S454 * _S457;
    float _S480 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S459, 2U));
    _s_diff_ctx_0->_S387 = _S480;
    float _S481 = _S479 + _S480 * _S455 * _S456;
    float _S482 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S459, 2U));
    _s_diff_ctx_0->_S388 = _S482;
    *(&(&_S468)->vals_1[int(2)]) = _S481 + _S482 * _S454 * _S456;
    float _S483 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S453, 3U));
    _s_diff_ctx_0->_S389 = _S483;
    float _S484 = _S483 * _S455 * _S457;
    float _S485 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S453, 3U));
    _s_diff_ctx_0->_S390 = _S485;
    float _S486 = _S484 + _S485 * _S454 * _S457;
    float _S487 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S459, 3U));
    _s_diff_ctx_0->_S391 = _S487;
    float _S488 = _S486 + _S487 * _S455 * _S456;
    float _S489 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S459, 3U));
    _s_diff_ctx_0->_S392 = _S489;
    *(&(&_S468)->vals_1[int(3)]) = _S488 + _S489 * _S454 * _S456;
    float _S490 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S453, 4U));
    _s_diff_ctx_0->_S393 = _S490;
    float _S491 = _S490 * _S455 * _S457;
    float _S492 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S453, 4U));
    _s_diff_ctx_0->_S394 = _S492;
    float _S493 = _S491 + _S492 * _S454 * _S457;
    float _S494 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S459, 4U));
    _s_diff_ctx_0->_S395 = _S494;
    float _S495 = _S493 + _S494 * _S455 * _S456;
    float _S496 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S459, 4U));
    _s_diff_ctx_0->_S396 = _S496;
    *(&(&_S468)->vals_1[int(4)]) = _S495 + _S496 * _S454 * _S456;
    float _S497 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S453, 5U));
    _s_diff_ctx_0->_S397 = _S497;
    float _S498 = _S497 * _S455 * _S457;
    float _S499 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S453, 5U));
    _s_diff_ctx_0->_S398 = _S499;
    float _S500 = _S498 + _S499 * _S454 * _S457;
    float _S501 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S459, 5U));
    _s_diff_ctx_0->_S399 = _S501;
    float _S502 = _S500 + _S501 * _S455 * _S456;
    float _S503 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S459, 5U));
    _s_diff_ctx_0->_S400 = _S503;
    *(&(&_S468)->vals_1[int(5)]) = _S502 + _S503 * _S454 * _S456;
    float _S504 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S453, 6U));
    _s_diff_ctx_0->_S401 = _S504;
    float _S505 = _S504 * _S455 * _S457;
    float _S506 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S453, 6U));
    _s_diff_ctx_0->_S402 = _S506;
    float _S507 = _S505 + _S506 * _S454 * _S457;
    float _S508 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S459, 6U));
    _s_diff_ctx_0->_S403 = _S508;
    float _S509 = _S507 + _S508 * _S455 * _S456;
    float _S510 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S459, 6U));
    _s_diff_ctx_0->_S404 = _S510;
    *(&(&_S468)->vals_1[int(6)]) = _S509 + _S510 * _S454 * _S456;
    float _S511 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S453, 7U));
    _s_diff_ctx_0->_S405 = _S511;
    float _S512 = _S511 * _S455 * _S457;
    float _S513 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S453, 7U));
    _s_diff_ctx_0->_S406 = _S513;
    float _S514 = _S512 + _S513 * _S454 * _S457;
    float _S515 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S459, 7U));
    _s_diff_ctx_0->_S407 = _S515;
    float _S516 = _S514 + _S515 * _S455 * _S456;
    float _S517 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S459, 7U));
    _s_diff_ctx_0->_S408 = _S517;
    *(&(&_S468)->vals_1[int(7)]) = _S516 + _S517 * _S454 * _S456;
    float _S518 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S453, 8U));
    _s_diff_ctx_0->_S409 = _S518;
    float _S519 = _S518 * _S455 * _S457;
    float _S520 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S453, 8U));
    _s_diff_ctx_0->_S410 = _S520;
    float _S521 = _S519 + _S520 * _S454 * _S457;
    float _S522 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S459, 8U));
    _s_diff_ctx_0->_S411 = _S522;
    float _S523 = _S521 + _S522 * _S455 * _S456;
    float _S524 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S459, 8U));
    _s_diff_ctx_0->_S412 = _S524;
    *(&(&_S468)->vals_1[int(8)]) = _S523 + _S524 * _S454 * _S456;
    float _S525 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S453, 9U));
    _s_diff_ctx_0->_S413 = _S525;
    float _S526 = _S525 * _S455 * _S457;
    float _S527 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S453, 9U));
    _s_diff_ctx_0->_S414 = _S527;
    float _S528 = _S526 + _S527 * _S454 * _S457;
    float _S529 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S459, 9U));
    _s_diff_ctx_0->_S415 = _S529;
    float _S530 = _S528 + _S529 * _S455 * _S456;
    float _S531 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S459, 9U));
    _s_diff_ctx_0->_S416 = _S531;
    *(&(&_S468)->vals_1[int(9)]) = _S530 + _S531 * _S454 * _S456;
    float _S532 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S453, 10U));
    _s_diff_ctx_0->_S417 = _S532;
    float _S533 = _S532 * _S455 * _S457;
    float _S534 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S453, 10U));
    _s_diff_ctx_0->_S418 = _S534;
    float _S535 = _S533 + _S534 * _S454 * _S457;
    float _S536 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S459, 10U));
    _s_diff_ctx_0->_S419 = _S536;
    float _S537 = _S535 + _S536 * _S455 * _S456;
    float _S538 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S459, 10U));
    _s_diff_ctx_0->_S420 = _S538;
    *(&(&_S468)->vals_1[int(10)]) = _S537 + _S538 * _S454 * _S456;
    float _S539 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S453, 11U));
    _s_diff_ctx_0->_S421 = _S539;
    float _S540 = _S539 * _S455 * _S457;
    float _S541 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S453, 11U));
    _s_diff_ctx_0->_S422 = _S541;
    float _S542 = _S540 + _S541 * _S454 * _S457;
    float _S543 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S459, 11U));
    _s_diff_ctx_0->_S423 = _S543;
    float _S544 = _S542 + _S543 * _S455 * _S456;
    float _S545 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S459, 11U));
    _s_diff_ctx_0->_S424 = _S545;
    *(&(&_S468)->vals_1[int(11)]) = _S544 + _S545 * _S454 * _S456;
    float _S546 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S453, 12U));
    _s_diff_ctx_0->_S425 = _S546;
    float _S547 = _S546 * _S455 * _S457;
    float _S548 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S453, 12U));
    _s_diff_ctx_0->_S426 = _S548;
    float _S549 = _S547 + _S548 * _S454 * _S457;
    float _S550 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S459, 12U));
    _s_diff_ctx_0->_S427 = _S550;
    float _S551 = _S549 + _S550 * _S455 * _S456;
    float _S552 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S459, 12U));
    _s_diff_ctx_0->_S428 = _S552;
    *(&(&_S468)->vals_1[int(12)]) = _S551 + _S552 * _S454 * _S456;
    float _S553 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S453, 13U));
    _s_diff_ctx_0->_S429 = _S553;
    float _S554 = _S553 * _S455 * _S457;
    float _S555 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S453, 13U));
    _s_diff_ctx_0->_S430 = _S555;
    float _S556 = _S554 + _S555 * _S454 * _S457;
    float _S557 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S452, _S459, 13U));
    _s_diff_ctx_0->_S431 = _S557;
    float _S558 = _S556 + _S557 * _S455 * _S456;
    float _S559 = s_primal_ctx_DiffTensorView_load_0(featureGrid_1, make_uint3 (_S458, _S459, 13U));
    _s_diff_ctx_0->_S432 = _S559;
    *(&(&_S468)->vals_1[int(13)]) = _S558 + _S559 * _S454 * _S456;
    *(&(&_S468)->vals_1[14U]) = _S454;
    *(&(&_S468)->vals_1[15U]) = _S456;
    return _S468;
}

__device__ Feature_0 s_primal_ctx_Linear_eval_0(Linear_0 _S560, Feature_0 _S561)
{
    Feature_0 _S562 = Linear_eval_0(_S560, _S561);
    return _S562;
}

__device__ Feature_0 s_bwd_primal_MLP_eval_0(MLP_0 this_29, Feature_0 dpin_feature_0, s_bwd_prop_MLP_eval_Intermediates_0 * _s_diff_ctx_1)
{
    FixedArray<float, 16>  _S563 = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    Feature_0 _S564 = { _S563 };
    _s_diff_ctx_1->_S433 = _S564;
    _s_diff_ctx_1->_S434 = _S564;
    _s_diff_ctx_1->_S435 = _S564;
    Feature_0 _S565 = s_primal_ctx_Linear_eval_0(this_29.layers_0[int(0)], dpin_feature_0);
    _s_diff_ctx_1->_S433 = _S565;
    float _S566 = s_primal_ctx_max_0(0.0f, _S565.vals_1[int(0)]);
    Feature_0 _S567 = _S565;
    *(&(&_S567)->vals_1[int(0)]) = _S566;
    *(&(&_S567)->vals_1[int(1)]) = s_primal_ctx_max_0(0.0f, _S565.vals_1[int(1)]);
    *(&(&_S567)->vals_1[int(2)]) = s_primal_ctx_max_0(0.0f, _S565.vals_1[int(2)]);
    *(&(&_S567)->vals_1[int(3)]) = s_primal_ctx_max_0(0.0f, _S565.vals_1[int(3)]);
    *(&(&_S567)->vals_1[int(4)]) = s_primal_ctx_max_0(0.0f, _S565.vals_1[int(4)]);
    *(&(&_S567)->vals_1[int(5)]) = s_primal_ctx_max_0(0.0f, _S565.vals_1[int(5)]);
    *(&(&_S567)->vals_1[int(6)]) = s_primal_ctx_max_0(0.0f, _S565.vals_1[int(6)]);
    *(&(&_S567)->vals_1[int(7)]) = s_primal_ctx_max_0(0.0f, _S565.vals_1[int(7)]);
    *(&(&_S567)->vals_1[int(8)]) = s_primal_ctx_max_0(0.0f, _S565.vals_1[int(8)]);
    *(&(&_S567)->vals_1[int(9)]) = s_primal_ctx_max_0(0.0f, _S565.vals_1[int(9)]);
    *(&(&_S567)->vals_1[int(10)]) = s_primal_ctx_max_0(0.0f, _S565.vals_1[int(10)]);
    *(&(&_S567)->vals_1[int(11)]) = s_primal_ctx_max_0(0.0f, _S565.vals_1[int(11)]);
    *(&(&_S567)->vals_1[int(12)]) = s_primal_ctx_max_0(0.0f, _S565.vals_1[int(12)]);
    *(&(&_S567)->vals_1[int(13)]) = s_primal_ctx_max_0(0.0f, _S565.vals_1[int(13)]);
    *(&(&_S567)->vals_1[int(14)]) = s_primal_ctx_max_0(0.0f, _S565.vals_1[int(14)]);
    *(&(&_S567)->vals_1[int(15)]) = s_primal_ctx_max_0(0.0f, _S565.vals_1[int(15)]);
    Feature_0 _S568 = s_primal_ctx_Linear_eval_0(this_29.layers_0[int(1)], _S567);
    _s_diff_ctx_1->_S434 = _S568;
    float _S569 = s_primal_ctx_max_0(0.0f, _S568.vals_1[int(0)]);
    _S567 = _S568;
    *(&(&_S567)->vals_1[int(0)]) = _S569;
    *(&(&_S567)->vals_1[int(1)]) = s_primal_ctx_max_0(0.0f, _S568.vals_1[int(1)]);
    *(&(&_S567)->vals_1[int(2)]) = s_primal_ctx_max_0(0.0f, _S568.vals_1[int(2)]);
    *(&(&_S567)->vals_1[int(3)]) = s_primal_ctx_max_0(0.0f, _S568.vals_1[int(3)]);
    *(&(&_S567)->vals_1[int(4)]) = s_primal_ctx_max_0(0.0f, _S568.vals_1[int(4)]);
    *(&(&_S567)->vals_1[int(5)]) = s_primal_ctx_max_0(0.0f, _S568.vals_1[int(5)]);
    *(&(&_S567)->vals_1[int(6)]) = s_primal_ctx_max_0(0.0f, _S568.vals_1[int(6)]);
    *(&(&_S567)->vals_1[int(7)]) = s_primal_ctx_max_0(0.0f, _S568.vals_1[int(7)]);
    *(&(&_S567)->vals_1[int(8)]) = s_primal_ctx_max_0(0.0f, _S568.vals_1[int(8)]);
    *(&(&_S567)->vals_1[int(9)]) = s_primal_ctx_max_0(0.0f, _S568.vals_1[int(9)]);
    *(&(&_S567)->vals_1[int(10)]) = s_primal_ctx_max_0(0.0f, _S568.vals_1[int(10)]);
    *(&(&_S567)->vals_1[int(11)]) = s_primal_ctx_max_0(0.0f, _S568.vals_1[int(11)]);
    *(&(&_S567)->vals_1[int(12)]) = s_primal_ctx_max_0(0.0f, _S568.vals_1[int(12)]);
    *(&(&_S567)->vals_1[int(13)]) = s_primal_ctx_max_0(0.0f, _S568.vals_1[int(13)]);
    *(&(&_S567)->vals_1[int(14)]) = s_primal_ctx_max_0(0.0f, _S568.vals_1[int(14)]);
    *(&(&_S567)->vals_1[int(15)]) = s_primal_ctx_max_0(0.0f, _S568.vals_1[int(15)]);
    Feature_0 _S570 = s_primal_ctx_Linear_eval_0(this_29.layers_0[int(2)], _S567);
    _s_diff_ctx_1->_S435 = _S570;
    float _S571 = s_primal_ctx_max_0(0.0f, _S570.vals_1[int(0)]);
    _S567 = _S570;
    *(&(&_S567)->vals_1[int(0)]) = _S571;
    *(&(&_S567)->vals_1[int(1)]) = s_primal_ctx_max_0(0.0f, _S570.vals_1[int(1)]);
    *(&(&_S567)->vals_1[int(2)]) = s_primal_ctx_max_0(0.0f, _S570.vals_1[int(2)]);
    *(&(&_S567)->vals_1[int(3)]) = s_primal_ctx_max_0(0.0f, _S570.vals_1[int(3)]);
    *(&(&_S567)->vals_1[int(4)]) = s_primal_ctx_max_0(0.0f, _S570.vals_1[int(4)]);
    *(&(&_S567)->vals_1[int(5)]) = s_primal_ctx_max_0(0.0f, _S570.vals_1[int(5)]);
    *(&(&_S567)->vals_1[int(6)]) = s_primal_ctx_max_0(0.0f, _S570.vals_1[int(6)]);
    *(&(&_S567)->vals_1[int(7)]) = s_primal_ctx_max_0(0.0f, _S570.vals_1[int(7)]);
    *(&(&_S567)->vals_1[int(8)]) = s_primal_ctx_max_0(0.0f, _S570.vals_1[int(8)]);
    *(&(&_S567)->vals_1[int(9)]) = s_primal_ctx_max_0(0.0f, _S570.vals_1[int(9)]);
    *(&(&_S567)->vals_1[int(10)]) = s_primal_ctx_max_0(0.0f, _S570.vals_1[int(10)]);
    *(&(&_S567)->vals_1[int(11)]) = s_primal_ctx_max_0(0.0f, _S570.vals_1[int(11)]);
    *(&(&_S567)->vals_1[int(12)]) = s_primal_ctx_max_0(0.0f, _S570.vals_1[int(12)]);
    *(&(&_S567)->vals_1[int(13)]) = s_primal_ctx_max_0(0.0f, _S570.vals_1[int(13)]);
    *(&(&_S567)->vals_1[int(14)]) = s_primal_ctx_max_0(0.0f, _S570.vals_1[int(14)]);
    *(&(&_S567)->vals_1[int(15)]) = s_primal_ctx_max_0(0.0f, _S570.vals_1[int(15)]);
    return _S567;
}

__device__ void s_primal_ctx_DiffTensorView_storeOnce_0(DiffTensorView_0 _S572, uint3  _S573, float _S574)
{
    DiffTensorView_storeOnce_0(_S572, _S573, _S574);
    return;
}

__device__ void s_bwd_primal_renderImage_0(MLP_0 mlp_0, DiffTensorView_0 featureGrid_2, DiffTensorView_0 imageOutput_0, s_bwd_prop_renderImage_Intermediates_0 * _s_diff_ctx_2)
{
    s_bwd_prop_computeInterpolatedFeature_Intermediates_0 _S575 = { 0U, 0U, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    FixedArray<float, 16>  _S576 = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    Feature_0 _S577 = { _S576 };
    s_bwd_prop_MLP_eval_Intermediates_0 _S578 = { _S577, _S577, _S577 };
    _s_diff_ctx_2->_S436 = _S575;
    _s_diff_ctx_2->_S437 = _S577;
    _s_diff_ctx_2->_S438 = _S578;
    _s_diff_ctx_2->_S439 = _S577;
    _s_diff_ctx_2->_S440 = 0U;
    _s_diff_ctx_2->_S441 = 0U;
    (&_s_diff_ctx_2->_S437)->vals_1 = _S576;
    (&_s_diff_ctx_2->_S439)->vals_1 = _S576;
    uint3  dispatchIdx_0 = ((threadIdx)) + ((blockIdx)) * ((blockDim));
    uint _S579 = dispatchIdx_0.x;
    uint _S580 = DiffTensorView_size_0(imageOutput_0, 0U);
    _s_diff_ctx_2->_S440 = _S580;
    bool _S581 = _S579 >= _S580;
    uint _S582 = dispatchIdx_0.y;
    uint _S583 = DiffTensorView_size_0(imageOutput_0, 1U);
    _s_diff_ctx_2->_S441 = _S583;
    if(!(_S581 || _S582 >= _S583))
    {
        Feature_0 _S584 = s_bwd_primal_computeInterpolatedFeature_0(featureGrid_2, make_uint2 (_S580, _S583), &_s_diff_ctx_2->_S436);
        _s_diff_ctx_2->_S437 = _S584;
        Feature_0 _S585 = s_bwd_primal_MLP_eval_0(mlp_0, _S584, &_s_diff_ctx_2->_S438);
        _s_diff_ctx_2->_S439 = _S585;
        s_primal_ctx_DiffTensorView_storeOnce_0(imageOutput_0, make_uint3 (_S579, _S582, 0U), _S585.vals_1[int(0)]);
        s_primal_ctx_DiffTensorView_storeOnce_0(imageOutput_0, make_uint3 (_S579, _S582, 1U), _S585.vals_1[int(1)]);
        s_primal_ctx_DiffTensorView_storeOnce_0(imageOutput_0, make_uint3 (_S579, _S582, 2U), _S585.vals_1[int(2)]);
    }
    return;
}

__device__ void s_bwd_prop_DiffTensorView_storeOnce_0(DiffTensorView_0 _S586, uint3  _S587, DiffPair_float_0 * _S588)
{
    DiffTensorView_storeOnce_backward_0(_S586, _S587, _S588);
    return;
}

__device__ void s_bwd_prop_Linear_eval_0(Linear_0 _S589, DiffPair_Feature_0 * _S590, s_diff_Feature_0 _S591, uint _S592)
{
    Linear_eval_bwd_0(_S589, _S590, _S591, _S592);
    return;
}

__device__ void s_bwd_prop_MLP_eval_0(MLP_0 this_30, DiffPair_Feature_0 * dpin_feature_1, s_diff_Feature_0 _s_dOut_0, s_bwd_prop_MLP_eval_Intermediates_0 _s_diff_ctx_3, uint _S593)
{
    float _S594 = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S433.vals_1[int(0)]);
    Feature_0 _S595 = _s_diff_ctx_3._S433;
    *(&(&_S595)->vals_1[int(0)]) = _S594;
    *(&(&_S595)->vals_1[int(1)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S433.vals_1[int(1)]);
    *(&(&_S595)->vals_1[int(2)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S433.vals_1[int(2)]);
    *(&(&_S595)->vals_1[int(3)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S433.vals_1[int(3)]);
    *(&(&_S595)->vals_1[int(4)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S433.vals_1[int(4)]);
    *(&(&_S595)->vals_1[int(5)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S433.vals_1[int(5)]);
    *(&(&_S595)->vals_1[int(6)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S433.vals_1[int(6)]);
    *(&(&_S595)->vals_1[int(7)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S433.vals_1[int(7)]);
    *(&(&_S595)->vals_1[int(8)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S433.vals_1[int(8)]);
    *(&(&_S595)->vals_1[int(9)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S433.vals_1[int(9)]);
    *(&(&_S595)->vals_1[int(10)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S433.vals_1[int(10)]);
    *(&(&_S595)->vals_1[int(11)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S433.vals_1[int(11)]);
    *(&(&_S595)->vals_1[int(12)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S433.vals_1[int(12)]);
    *(&(&_S595)->vals_1[int(13)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S433.vals_1[int(13)]);
    *(&(&_S595)->vals_1[int(14)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S433.vals_1[int(14)]);
    *(&(&_S595)->vals_1[int(15)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S433.vals_1[int(15)]);
    float _S596 = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S434.vals_1[int(0)]);
    Feature_0 _S597 = _s_diff_ctx_3._S434;
    *(&(&_S597)->vals_1[int(0)]) = _S596;
    *(&(&_S597)->vals_1[int(1)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S434.vals_1[int(1)]);
    *(&(&_S597)->vals_1[int(2)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S434.vals_1[int(2)]);
    *(&(&_S597)->vals_1[int(3)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S434.vals_1[int(3)]);
    *(&(&_S597)->vals_1[int(4)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S434.vals_1[int(4)]);
    *(&(&_S597)->vals_1[int(5)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S434.vals_1[int(5)]);
    *(&(&_S597)->vals_1[int(6)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S434.vals_1[int(6)]);
    *(&(&_S597)->vals_1[int(7)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S434.vals_1[int(7)]);
    *(&(&_S597)->vals_1[int(8)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S434.vals_1[int(8)]);
    *(&(&_S597)->vals_1[int(9)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S434.vals_1[int(9)]);
    *(&(&_S597)->vals_1[int(10)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S434.vals_1[int(10)]);
    *(&(&_S597)->vals_1[int(11)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S434.vals_1[int(11)]);
    *(&(&_S597)->vals_1[int(12)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S434.vals_1[int(12)]);
    *(&(&_S597)->vals_1[int(13)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S434.vals_1[int(13)]);
    *(&(&_S597)->vals_1[int(14)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S434.vals_1[int(14)]);
    *(&(&_S597)->vals_1[int(15)]) = s_primal_ctx_max_0(0.0f, _s_diff_ctx_3._S434.vals_1[int(15)]);
    s_diff_Feature_0 _S598 = _s_dOut_0;
    *(&(&_S598)->vals_0[int(15)]) = 0.0f;
    DiffPair_float_0 _S599;
    (&_S599)->primal_1 = 0.0f;
    (&_S599)->differential_0 = 0.0f;
    DiffPair_float_0 _S600;
    (&_S600)->primal_1 = _s_diff_ctx_3._S435.vals_1[int(15)];
    (&_S600)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S599, &_S600, _s_dOut_0.vals_0[int(15)]);
    *(&(&_S598)->vals_0[int(14)]) = 0.0f;
    DiffPair_float_0 _S601;
    (&_S601)->primal_1 = 0.0f;
    (&_S601)->differential_0 = 0.0f;
    DiffPair_float_0 _S602;
    (&_S602)->primal_1 = _s_diff_ctx_3._S435.vals_1[int(14)];
    (&_S602)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S601, &_S602, _s_dOut_0.vals_0[int(14)]);
    *(&(&_S598)->vals_0[int(13)]) = 0.0f;
    DiffPair_float_0 _S603;
    (&_S603)->primal_1 = 0.0f;
    (&_S603)->differential_0 = 0.0f;
    DiffPair_float_0 _S604;
    (&_S604)->primal_1 = _s_diff_ctx_3._S435.vals_1[int(13)];
    (&_S604)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S603, &_S604, _s_dOut_0.vals_0[int(13)]);
    *(&(&_S598)->vals_0[int(12)]) = 0.0f;
    DiffPair_float_0 _S605;
    (&_S605)->primal_1 = 0.0f;
    (&_S605)->differential_0 = 0.0f;
    DiffPair_float_0 _S606;
    (&_S606)->primal_1 = _s_diff_ctx_3._S435.vals_1[int(12)];
    (&_S606)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S605, &_S606, _s_dOut_0.vals_0[int(12)]);
    *(&(&_S598)->vals_0[int(11)]) = 0.0f;
    DiffPair_float_0 _S607;
    (&_S607)->primal_1 = 0.0f;
    (&_S607)->differential_0 = 0.0f;
    DiffPair_float_0 _S608;
    (&_S608)->primal_1 = _s_diff_ctx_3._S435.vals_1[int(11)];
    (&_S608)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S607, &_S608, _s_dOut_0.vals_0[int(11)]);
    *(&(&_S598)->vals_0[int(10)]) = 0.0f;
    DiffPair_float_0 _S609;
    (&_S609)->primal_1 = 0.0f;
    (&_S609)->differential_0 = 0.0f;
    DiffPair_float_0 _S610;
    (&_S610)->primal_1 = _s_diff_ctx_3._S435.vals_1[int(10)];
    (&_S610)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S609, &_S610, _s_dOut_0.vals_0[int(10)]);
    *(&(&_S598)->vals_0[int(9)]) = 0.0f;
    DiffPair_float_0 _S611;
    (&_S611)->primal_1 = 0.0f;
    (&_S611)->differential_0 = 0.0f;
    DiffPair_float_0 _S612;
    (&_S612)->primal_1 = _s_diff_ctx_3._S435.vals_1[int(9)];
    (&_S612)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S611, &_S612, _s_dOut_0.vals_0[int(9)]);
    *(&(&_S598)->vals_0[int(8)]) = 0.0f;
    DiffPair_float_0 _S613;
    (&_S613)->primal_1 = 0.0f;
    (&_S613)->differential_0 = 0.0f;
    DiffPair_float_0 _S614;
    (&_S614)->primal_1 = _s_diff_ctx_3._S435.vals_1[int(8)];
    (&_S614)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S613, &_S614, _s_dOut_0.vals_0[int(8)]);
    *(&(&_S598)->vals_0[int(7)]) = 0.0f;
    DiffPair_float_0 _S615;
    (&_S615)->primal_1 = 0.0f;
    (&_S615)->differential_0 = 0.0f;
    DiffPair_float_0 _S616;
    (&_S616)->primal_1 = _s_diff_ctx_3._S435.vals_1[int(7)];
    (&_S616)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S615, &_S616, _s_dOut_0.vals_0[int(7)]);
    *(&(&_S598)->vals_0[int(6)]) = 0.0f;
    DiffPair_float_0 _S617;
    (&_S617)->primal_1 = 0.0f;
    (&_S617)->differential_0 = 0.0f;
    DiffPair_float_0 _S618;
    (&_S618)->primal_1 = _s_diff_ctx_3._S435.vals_1[int(6)];
    (&_S618)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S617, &_S618, _s_dOut_0.vals_0[int(6)]);
    *(&(&_S598)->vals_0[int(5)]) = 0.0f;
    DiffPair_float_0 _S619;
    (&_S619)->primal_1 = 0.0f;
    (&_S619)->differential_0 = 0.0f;
    DiffPair_float_0 _S620;
    (&_S620)->primal_1 = _s_diff_ctx_3._S435.vals_1[int(5)];
    (&_S620)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S619, &_S620, _s_dOut_0.vals_0[int(5)]);
    *(&(&_S598)->vals_0[int(4)]) = 0.0f;
    DiffPair_float_0 _S621;
    (&_S621)->primal_1 = 0.0f;
    (&_S621)->differential_0 = 0.0f;
    DiffPair_float_0 _S622;
    (&_S622)->primal_1 = _s_diff_ctx_3._S435.vals_1[int(4)];
    (&_S622)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S621, &_S622, _s_dOut_0.vals_0[int(4)]);
    *(&(&_S598)->vals_0[int(3)]) = 0.0f;
    DiffPair_float_0 _S623;
    (&_S623)->primal_1 = 0.0f;
    (&_S623)->differential_0 = 0.0f;
    DiffPair_float_0 _S624;
    (&_S624)->primal_1 = _s_diff_ctx_3._S435.vals_1[int(3)];
    (&_S624)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S623, &_S624, _s_dOut_0.vals_0[int(3)]);
    *(&(&_S598)->vals_0[int(2)]) = 0.0f;
    DiffPair_float_0 _S625;
    (&_S625)->primal_1 = 0.0f;
    (&_S625)->differential_0 = 0.0f;
    DiffPair_float_0 _S626;
    (&_S626)->primal_1 = _s_diff_ctx_3._S435.vals_1[int(2)];
    (&_S626)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S625, &_S626, _s_dOut_0.vals_0[int(2)]);
    *(&(&_S598)->vals_0[int(1)]) = 0.0f;
    DiffPair_float_0 _S627;
    (&_S627)->primal_1 = 0.0f;
    (&_S627)->differential_0 = 0.0f;
    DiffPair_float_0 _S628;
    (&_S628)->primal_1 = _s_diff_ctx_3._S435.vals_1[int(1)];
    (&_S628)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S627, &_S628, _s_dOut_0.vals_0[int(1)]);
    *(&(&_S598)->vals_0[int(0)]) = 0.0f;
    DiffPair_float_0 _S629;
    (&_S629)->primal_1 = 0.0f;
    (&_S629)->differential_0 = 0.0f;
    DiffPair_float_0 _S630;
    (&_S630)->primal_1 = _s_diff_ctx_3._S435.vals_1[int(0)];
    (&_S630)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S629, &_S630, _s_dOut_0.vals_0[int(0)]);
    FixedArray<float, 16>  _S631;
    *(&_S631[int(0)]) = 0.0f;
    *(&_S631[int(1)]) = 0.0f;
    *(&_S631[int(2)]) = 0.0f;
    *(&_S631[int(3)]) = 0.0f;
    *(&_S631[int(4)]) = 0.0f;
    *(&_S631[int(5)]) = 0.0f;
    *(&_S631[int(6)]) = 0.0f;
    *(&_S631[int(7)]) = 0.0f;
    *(&_S631[int(8)]) = 0.0f;
    *(&_S631[int(9)]) = 0.0f;
    *(&_S631[int(10)]) = 0.0f;
    *(&_S631[int(11)]) = 0.0f;
    *(&_S631[int(12)]) = 0.0f;
    *(&_S631[int(13)]) = 0.0f;
    *(&_S631[int(14)]) = 0.0f;
    *(&_S631[int(15)]) = 0.0f;
    *(&_S631[int(15)]) = _S600.differential_0;
    *(&_S631[int(14)]) = _S602.differential_0;
    *(&_S631[int(13)]) = _S604.differential_0;
    *(&_S631[int(12)]) = _S606.differential_0;
    *(&_S631[int(11)]) = _S608.differential_0;
    *(&_S631[int(10)]) = _S610.differential_0;
    *(&_S631[int(9)]) = _S612.differential_0;
    *(&_S631[int(8)]) = _S614.differential_0;
    *(&_S631[int(7)]) = _S616.differential_0;
    *(&_S631[int(6)]) = _S618.differential_0;
    *(&_S631[int(5)]) = _S620.differential_0;
    *(&_S631[int(4)]) = _S622.differential_0;
    *(&_S631[int(3)]) = _S624.differential_0;
    *(&_S631[int(2)]) = _S626.differential_0;
    *(&_S631[int(1)]) = _S628.differential_0;
    *(&_S631[int(0)]) = _S630.differential_0;
    s_diff_Feature_0 _S632 = Feature_x24_syn_dzero_0();
    s_diff_Feature_0 _S633 = _S632;
    (&_S633)->vals_0 = _S631;
    s_diff_Feature_0 _S634 = Feature_x24_syn_dadd_0(_S598, _S633);
    DiffPair_Feature_0 _S635;
    (&_S635)->primal_1 = _S597;
    (&_S635)->differential_0 = _S632;
    s_bwd_prop_Linear_eval_0(this_30.layers_0[int(2)], &_S635, _S634, _S593);
    _S598 = _S635.differential_0;
    *(&(&_S598)->vals_0[int(15)]) = 0.0f;
    DiffPair_float_0 _S636;
    (&_S636)->primal_1 = 0.0f;
    (&_S636)->differential_0 = 0.0f;
    DiffPair_float_0 _S637;
    (&_S637)->primal_1 = _s_diff_ctx_3._S434.vals_1[int(15)];
    (&_S637)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S636, &_S637, _S635.differential_0.vals_0[int(15)]);
    *(&(&_S598)->vals_0[int(14)]) = 0.0f;
    DiffPair_float_0 _S638;
    (&_S638)->primal_1 = 0.0f;
    (&_S638)->differential_0 = 0.0f;
    DiffPair_float_0 _S639;
    (&_S639)->primal_1 = _s_diff_ctx_3._S434.vals_1[int(14)];
    (&_S639)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S638, &_S639, _S635.differential_0.vals_0[int(14)]);
    *(&(&_S598)->vals_0[int(13)]) = 0.0f;
    DiffPair_float_0 _S640;
    (&_S640)->primal_1 = 0.0f;
    (&_S640)->differential_0 = 0.0f;
    DiffPair_float_0 _S641;
    (&_S641)->primal_1 = _s_diff_ctx_3._S434.vals_1[int(13)];
    (&_S641)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S640, &_S641, _S635.differential_0.vals_0[int(13)]);
    *(&(&_S598)->vals_0[int(12)]) = 0.0f;
    DiffPair_float_0 _S642;
    (&_S642)->primal_1 = 0.0f;
    (&_S642)->differential_0 = 0.0f;
    DiffPair_float_0 _S643;
    (&_S643)->primal_1 = _s_diff_ctx_3._S434.vals_1[int(12)];
    (&_S643)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S642, &_S643, _S635.differential_0.vals_0[int(12)]);
    *(&(&_S598)->vals_0[int(11)]) = 0.0f;
    DiffPair_float_0 _S644;
    (&_S644)->primal_1 = 0.0f;
    (&_S644)->differential_0 = 0.0f;
    DiffPair_float_0 _S645;
    (&_S645)->primal_1 = _s_diff_ctx_3._S434.vals_1[int(11)];
    (&_S645)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S644, &_S645, _S635.differential_0.vals_0[int(11)]);
    *(&(&_S598)->vals_0[int(10)]) = 0.0f;
    DiffPair_float_0 _S646;
    (&_S646)->primal_1 = 0.0f;
    (&_S646)->differential_0 = 0.0f;
    DiffPair_float_0 _S647;
    (&_S647)->primal_1 = _s_diff_ctx_3._S434.vals_1[int(10)];
    (&_S647)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S646, &_S647, _S635.differential_0.vals_0[int(10)]);
    *(&(&_S598)->vals_0[int(9)]) = 0.0f;
    DiffPair_float_0 _S648;
    (&_S648)->primal_1 = 0.0f;
    (&_S648)->differential_0 = 0.0f;
    DiffPair_float_0 _S649;
    (&_S649)->primal_1 = _s_diff_ctx_3._S434.vals_1[int(9)];
    (&_S649)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S648, &_S649, _S635.differential_0.vals_0[int(9)]);
    *(&(&_S598)->vals_0[int(8)]) = 0.0f;
    DiffPair_float_0 _S650;
    (&_S650)->primal_1 = 0.0f;
    (&_S650)->differential_0 = 0.0f;
    DiffPair_float_0 _S651;
    (&_S651)->primal_1 = _s_diff_ctx_3._S434.vals_1[int(8)];
    (&_S651)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S650, &_S651, _S635.differential_0.vals_0[int(8)]);
    *(&(&_S598)->vals_0[int(7)]) = 0.0f;
    DiffPair_float_0 _S652;
    (&_S652)->primal_1 = 0.0f;
    (&_S652)->differential_0 = 0.0f;
    DiffPair_float_0 _S653;
    (&_S653)->primal_1 = _s_diff_ctx_3._S434.vals_1[int(7)];
    (&_S653)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S652, &_S653, _S635.differential_0.vals_0[int(7)]);
    *(&(&_S598)->vals_0[int(6)]) = 0.0f;
    DiffPair_float_0 _S654;
    (&_S654)->primal_1 = 0.0f;
    (&_S654)->differential_0 = 0.0f;
    DiffPair_float_0 _S655;
    (&_S655)->primal_1 = _s_diff_ctx_3._S434.vals_1[int(6)];
    (&_S655)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S654, &_S655, _S635.differential_0.vals_0[int(6)]);
    *(&(&_S598)->vals_0[int(5)]) = 0.0f;
    DiffPair_float_0 _S656;
    (&_S656)->primal_1 = 0.0f;
    (&_S656)->differential_0 = 0.0f;
    DiffPair_float_0 _S657;
    (&_S657)->primal_1 = _s_diff_ctx_3._S434.vals_1[int(5)];
    (&_S657)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S656, &_S657, _S635.differential_0.vals_0[int(5)]);
    *(&(&_S598)->vals_0[int(4)]) = 0.0f;
    DiffPair_float_0 _S658;
    (&_S658)->primal_1 = 0.0f;
    (&_S658)->differential_0 = 0.0f;
    DiffPair_float_0 _S659;
    (&_S659)->primal_1 = _s_diff_ctx_3._S434.vals_1[int(4)];
    (&_S659)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S658, &_S659, _S635.differential_0.vals_0[int(4)]);
    *(&(&_S598)->vals_0[int(3)]) = 0.0f;
    DiffPair_float_0 _S660;
    (&_S660)->primal_1 = 0.0f;
    (&_S660)->differential_0 = 0.0f;
    DiffPair_float_0 _S661;
    (&_S661)->primal_1 = _s_diff_ctx_3._S434.vals_1[int(3)];
    (&_S661)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S660, &_S661, _S635.differential_0.vals_0[int(3)]);
    *(&(&_S598)->vals_0[int(2)]) = 0.0f;
    DiffPair_float_0 _S662;
    (&_S662)->primal_1 = 0.0f;
    (&_S662)->differential_0 = 0.0f;
    DiffPair_float_0 _S663;
    (&_S663)->primal_1 = _s_diff_ctx_3._S434.vals_1[int(2)];
    (&_S663)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S662, &_S663, _S635.differential_0.vals_0[int(2)]);
    *(&(&_S598)->vals_0[int(1)]) = 0.0f;
    DiffPair_float_0 _S664;
    (&_S664)->primal_1 = 0.0f;
    (&_S664)->differential_0 = 0.0f;
    DiffPair_float_0 _S665;
    (&_S665)->primal_1 = _s_diff_ctx_3._S434.vals_1[int(1)];
    (&_S665)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S664, &_S665, _S635.differential_0.vals_0[int(1)]);
    *(&(&_S598)->vals_0[int(0)]) = 0.0f;
    DiffPair_float_0 _S666;
    (&_S666)->primal_1 = 0.0f;
    (&_S666)->differential_0 = 0.0f;
    DiffPair_float_0 _S667;
    (&_S667)->primal_1 = _s_diff_ctx_3._S434.vals_1[int(0)];
    (&_S667)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S666, &_S667, _S635.differential_0.vals_0[int(0)]);
    FixedArray<float, 16>  _S668;
    *(&_S668[int(0)]) = 0.0f;
    *(&_S668[int(1)]) = 0.0f;
    *(&_S668[int(2)]) = 0.0f;
    *(&_S668[int(3)]) = 0.0f;
    *(&_S668[int(4)]) = 0.0f;
    *(&_S668[int(5)]) = 0.0f;
    *(&_S668[int(6)]) = 0.0f;
    *(&_S668[int(7)]) = 0.0f;
    *(&_S668[int(8)]) = 0.0f;
    *(&_S668[int(9)]) = 0.0f;
    *(&_S668[int(10)]) = 0.0f;
    *(&_S668[int(11)]) = 0.0f;
    *(&_S668[int(12)]) = 0.0f;
    *(&_S668[int(13)]) = 0.0f;
    *(&_S668[int(14)]) = 0.0f;
    *(&_S668[int(15)]) = 0.0f;
    *(&_S668[int(15)]) = _S637.differential_0;
    *(&_S668[int(14)]) = _S639.differential_0;
    *(&_S668[int(13)]) = _S641.differential_0;
    *(&_S668[int(12)]) = _S643.differential_0;
    *(&_S668[int(11)]) = _S645.differential_0;
    *(&_S668[int(10)]) = _S647.differential_0;
    *(&_S668[int(9)]) = _S649.differential_0;
    *(&_S668[int(8)]) = _S651.differential_0;
    *(&_S668[int(7)]) = _S653.differential_0;
    *(&_S668[int(6)]) = _S655.differential_0;
    *(&_S668[int(5)]) = _S657.differential_0;
    *(&_S668[int(4)]) = _S659.differential_0;
    *(&_S668[int(3)]) = _S661.differential_0;
    *(&_S668[int(2)]) = _S663.differential_0;
    *(&_S668[int(1)]) = _S665.differential_0;
    *(&_S668[int(0)]) = _S667.differential_0;
    s_diff_Feature_0 _S669 = _S632;
    (&_S669)->vals_0 = _S668;
    s_diff_Feature_0 _S670 = Feature_x24_syn_dadd_0(_S598, _S669);
    DiffPair_Feature_0 _S671;
    (&_S671)->primal_1 = _S595;
    (&_S671)->differential_0 = _S632;
    s_bwd_prop_Linear_eval_0(this_30.layers_0[int(1)], &_S671, _S670, _S593);
    _S598 = _S671.differential_0;
    *(&(&_S598)->vals_0[int(15)]) = 0.0f;
    DiffPair_float_0 _S672;
    (&_S672)->primal_1 = 0.0f;
    (&_S672)->differential_0 = 0.0f;
    DiffPair_float_0 _S673;
    (&_S673)->primal_1 = _s_diff_ctx_3._S433.vals_1[int(15)];
    (&_S673)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S672, &_S673, _S671.differential_0.vals_0[int(15)]);
    *(&(&_S598)->vals_0[int(14)]) = 0.0f;
    DiffPair_float_0 _S674;
    (&_S674)->primal_1 = 0.0f;
    (&_S674)->differential_0 = 0.0f;
    DiffPair_float_0 _S675;
    (&_S675)->primal_1 = _s_diff_ctx_3._S433.vals_1[int(14)];
    (&_S675)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S674, &_S675, _S671.differential_0.vals_0[int(14)]);
    *(&(&_S598)->vals_0[int(13)]) = 0.0f;
    DiffPair_float_0 _S676;
    (&_S676)->primal_1 = 0.0f;
    (&_S676)->differential_0 = 0.0f;
    DiffPair_float_0 _S677;
    (&_S677)->primal_1 = _s_diff_ctx_3._S433.vals_1[int(13)];
    (&_S677)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S676, &_S677, _S671.differential_0.vals_0[int(13)]);
    *(&(&_S598)->vals_0[int(12)]) = 0.0f;
    DiffPair_float_0 _S678;
    (&_S678)->primal_1 = 0.0f;
    (&_S678)->differential_0 = 0.0f;
    DiffPair_float_0 _S679;
    (&_S679)->primal_1 = _s_diff_ctx_3._S433.vals_1[int(12)];
    (&_S679)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S678, &_S679, _S671.differential_0.vals_0[int(12)]);
    *(&(&_S598)->vals_0[int(11)]) = 0.0f;
    DiffPair_float_0 _S680;
    (&_S680)->primal_1 = 0.0f;
    (&_S680)->differential_0 = 0.0f;
    DiffPair_float_0 _S681;
    (&_S681)->primal_1 = _s_diff_ctx_3._S433.vals_1[int(11)];
    (&_S681)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S680, &_S681, _S671.differential_0.vals_0[int(11)]);
    *(&(&_S598)->vals_0[int(10)]) = 0.0f;
    DiffPair_float_0 _S682;
    (&_S682)->primal_1 = 0.0f;
    (&_S682)->differential_0 = 0.0f;
    DiffPair_float_0 _S683;
    (&_S683)->primal_1 = _s_diff_ctx_3._S433.vals_1[int(10)];
    (&_S683)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S682, &_S683, _S671.differential_0.vals_0[int(10)]);
    *(&(&_S598)->vals_0[int(9)]) = 0.0f;
    DiffPair_float_0 _S684;
    (&_S684)->primal_1 = 0.0f;
    (&_S684)->differential_0 = 0.0f;
    DiffPair_float_0 _S685;
    (&_S685)->primal_1 = _s_diff_ctx_3._S433.vals_1[int(9)];
    (&_S685)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S684, &_S685, _S671.differential_0.vals_0[int(9)]);
    *(&(&_S598)->vals_0[int(8)]) = 0.0f;
    DiffPair_float_0 _S686;
    (&_S686)->primal_1 = 0.0f;
    (&_S686)->differential_0 = 0.0f;
    DiffPair_float_0 _S687;
    (&_S687)->primal_1 = _s_diff_ctx_3._S433.vals_1[int(8)];
    (&_S687)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S686, &_S687, _S671.differential_0.vals_0[int(8)]);
    *(&(&_S598)->vals_0[int(7)]) = 0.0f;
    DiffPair_float_0 _S688;
    (&_S688)->primal_1 = 0.0f;
    (&_S688)->differential_0 = 0.0f;
    DiffPair_float_0 _S689;
    (&_S689)->primal_1 = _s_diff_ctx_3._S433.vals_1[int(7)];
    (&_S689)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S688, &_S689, _S671.differential_0.vals_0[int(7)]);
    *(&(&_S598)->vals_0[int(6)]) = 0.0f;
    DiffPair_float_0 _S690;
    (&_S690)->primal_1 = 0.0f;
    (&_S690)->differential_0 = 0.0f;
    DiffPair_float_0 _S691;
    (&_S691)->primal_1 = _s_diff_ctx_3._S433.vals_1[int(6)];
    (&_S691)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S690, &_S691, _S671.differential_0.vals_0[int(6)]);
    *(&(&_S598)->vals_0[int(5)]) = 0.0f;
    DiffPair_float_0 _S692;
    (&_S692)->primal_1 = 0.0f;
    (&_S692)->differential_0 = 0.0f;
    DiffPair_float_0 _S693;
    (&_S693)->primal_1 = _s_diff_ctx_3._S433.vals_1[int(5)];
    (&_S693)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S692, &_S693, _S671.differential_0.vals_0[int(5)]);
    *(&(&_S598)->vals_0[int(4)]) = 0.0f;
    DiffPair_float_0 _S694;
    (&_S694)->primal_1 = 0.0f;
    (&_S694)->differential_0 = 0.0f;
    DiffPair_float_0 _S695;
    (&_S695)->primal_1 = _s_diff_ctx_3._S433.vals_1[int(4)];
    (&_S695)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S694, &_S695, _S671.differential_0.vals_0[int(4)]);
    *(&(&_S598)->vals_0[int(3)]) = 0.0f;
    DiffPair_float_0 _S696;
    (&_S696)->primal_1 = 0.0f;
    (&_S696)->differential_0 = 0.0f;
    DiffPair_float_0 _S697;
    (&_S697)->primal_1 = _s_diff_ctx_3._S433.vals_1[int(3)];
    (&_S697)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S696, &_S697, _S671.differential_0.vals_0[int(3)]);
    *(&(&_S598)->vals_0[int(2)]) = 0.0f;
    DiffPair_float_0 _S698;
    (&_S698)->primal_1 = 0.0f;
    (&_S698)->differential_0 = 0.0f;
    DiffPair_float_0 _S699;
    (&_S699)->primal_1 = _s_diff_ctx_3._S433.vals_1[int(2)];
    (&_S699)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S698, &_S699, _S671.differential_0.vals_0[int(2)]);
    *(&(&_S598)->vals_0[int(1)]) = 0.0f;
    DiffPair_float_0 _S700;
    (&_S700)->primal_1 = 0.0f;
    (&_S700)->differential_0 = 0.0f;
    DiffPair_float_0 _S701;
    (&_S701)->primal_1 = _s_diff_ctx_3._S433.vals_1[int(1)];
    (&_S701)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S700, &_S701, _S671.differential_0.vals_0[int(1)]);
    *(&(&_S598)->vals_0[int(0)]) = 0.0f;
    DiffPair_float_0 _S702;
    (&_S702)->primal_1 = 0.0f;
    (&_S702)->differential_0 = 0.0f;
    DiffPair_float_0 _S703;
    (&_S703)->primal_1 = _s_diff_ctx_3._S433.vals_1[int(0)];
    (&_S703)->differential_0 = 0.0f;
    s_bwd_prop_max_0(&_S702, &_S703, _S671.differential_0.vals_0[int(0)]);
    FixedArray<float, 16>  _S704;
    *(&_S704[int(0)]) = 0.0f;
    *(&_S704[int(1)]) = 0.0f;
    *(&_S704[int(2)]) = 0.0f;
    *(&_S704[int(3)]) = 0.0f;
    *(&_S704[int(4)]) = 0.0f;
    *(&_S704[int(5)]) = 0.0f;
    *(&_S704[int(6)]) = 0.0f;
    *(&_S704[int(7)]) = 0.0f;
    *(&_S704[int(8)]) = 0.0f;
    *(&_S704[int(9)]) = 0.0f;
    *(&_S704[int(10)]) = 0.0f;
    *(&_S704[int(11)]) = 0.0f;
    *(&_S704[int(12)]) = 0.0f;
    *(&_S704[int(13)]) = 0.0f;
    *(&_S704[int(14)]) = 0.0f;
    *(&_S704[int(15)]) = 0.0f;
    *(&_S704[int(15)]) = _S673.differential_0;
    *(&_S704[int(14)]) = _S675.differential_0;
    *(&_S704[int(13)]) = _S677.differential_0;
    *(&_S704[int(12)]) = _S679.differential_0;
    *(&_S704[int(11)]) = _S681.differential_0;
    *(&_S704[int(10)]) = _S683.differential_0;
    *(&_S704[int(9)]) = _S685.differential_0;
    *(&_S704[int(8)]) = _S687.differential_0;
    *(&_S704[int(7)]) = _S689.differential_0;
    *(&_S704[int(6)]) = _S691.differential_0;
    *(&_S704[int(5)]) = _S693.differential_0;
    *(&_S704[int(4)]) = _S695.differential_0;
    *(&_S704[int(3)]) = _S697.differential_0;
    *(&_S704[int(2)]) = _S699.differential_0;
    *(&_S704[int(1)]) = _S701.differential_0;
    *(&_S704[int(0)]) = _S703.differential_0;
    s_diff_Feature_0 _S705 = _S632;
    (&_S705)->vals_0 = _S704;
    s_diff_Feature_0 _S706 = Feature_x24_syn_dadd_0(_S598, _S705);
    DiffPair_Feature_0 _S707;
    (&_S707)->primal_1 = (*dpin_feature_1).primal_1;
    (&_S707)->differential_0 = _S632;
    s_bwd_prop_Linear_eval_0(this_30.layers_0[int(0)], &_S707, _S706, _S593);
    dpin_feature_1->primal_1 = (*dpin_feature_1).primal_1;
    dpin_feature_1->differential_0 = _S707.differential_0;
    return;
}

__device__ void s_bwd_prop_DiffTensorView_load_0(DiffTensorView_0 _S708, uint3  _S709, float _S710)
{
    DiffTensorView_load_backward_0(_S708, _S709, _S710);
    return;
}

__device__ void s_bwd_prop_computeInterpolatedFeature_0(DiffTensorView_0 featureGrid_3, uint2  frameDim_2, s_diff_Feature_0 _s_dOut_1, s_bwd_prop_computeInterpolatedFeature_Intermediates_0 _s_diff_ctx_4)
{
    uint dim0_1 = _s_diff_ctx_4._S375 - 1U;
    uint dim1_2 = _s_diff_ctx_4._S376 - 1U;
    uint _S711 = frameDim_2.x;
    uint _S712 = _S711 / dim0_1;
    uint _S713 = frameDim_2.y;
    uint _S714 = _S713 / dim1_2;
    uint2  pixelId_2 = uint2 {(((threadIdx)) + ((blockIdx)) * ((blockDim))).x, (((threadIdx)) + ((blockIdx)) * ((blockDim))).y};
    uint2  cellId_2 = pixelId_2 / make_uint2 (_S712, _S714);
    uint _S715 = _S711 / dim0_1;
    uint _S716 = _S713 / dim1_2;
    uint2  cellSize_2 = make_uint2 (_S715, _S716);
    uint2  cellId_3 = pixelId_2 / cellSize_2;
    float2  _S717 = make_float2 ((float)pixelId_2.x, (float)pixelId_2.y);
    float2  _S718 = make_float2 ((float)cellSize_2.x, (float)cellSize_2.y);
    float2  _S719 = make_float2 ((float)cellId_3.x, (float)cellId_3.y);
    float2  weights_5 = _S717 / _S718 - _S719;
    uint _S720 = cellId_3.x;
    uint _S721 = cellId_3.y;
    float _S722 = weights_5.x;
    float _S723 = 1.0f - _S722;
    float _S724 = weights_5.y;
    float _S725 = 1.0f - _S724;
    uint _S726 = _S720 + 1U;
    uint _S727 = _S721 + 1U;
    uint3  _S728 = make_uint3 (_S720, _S721, 0U);
    uint3  _S729 = make_uint3 (_S726, _S721, 0U);
    uint3  _S730 = make_uint3 (_S720, _S727, 0U);
    uint3  _S731 = make_uint3 (_S726, _S727, 0U);
    uint3  _S732 = make_uint3 (_S720, _S721, 1U);
    uint3  _S733 = make_uint3 (_S726, _S721, 1U);
    uint3  _S734 = make_uint3 (_S720, _S727, 1U);
    uint3  _S735 = make_uint3 (_S726, _S727, 1U);
    uint3  _S736 = make_uint3 (_S720, _S721, 2U);
    uint3  _S737 = make_uint3 (_S726, _S721, 2U);
    uint3  _S738 = make_uint3 (_S720, _S727, 2U);
    uint3  _S739 = make_uint3 (_S726, _S727, 2U);
    uint3  _S740 = make_uint3 (_S720, _S721, 3U);
    uint3  _S741 = make_uint3 (_S726, _S721, 3U);
    uint3  _S742 = make_uint3 (_S720, _S727, 3U);
    uint3  _S743 = make_uint3 (_S726, _S727, 3U);
    uint3  _S744 = make_uint3 (_S720, _S721, 4U);
    uint3  _S745 = make_uint3 (_S726, _S721, 4U);
    uint3  _S746 = make_uint3 (_S720, _S727, 4U);
    uint3  _S747 = make_uint3 (_S726, _S727, 4U);
    uint3  _S748 = make_uint3 (_S720, _S721, 5U);
    uint3  _S749 = make_uint3 (_S726, _S721, 5U);
    uint3  _S750 = make_uint3 (_S720, _S727, 5U);
    uint3  _S751 = make_uint3 (_S726, _S727, 5U);
    uint3  _S752 = make_uint3 (_S720, _S721, 6U);
    uint3  _S753 = make_uint3 (_S726, _S721, 6U);
    uint3  _S754 = make_uint3 (_S720, _S727, 6U);
    uint3  _S755 = make_uint3 (_S726, _S727, 6U);
    uint3  _S756 = make_uint3 (_S720, _S721, 7U);
    uint3  _S757 = make_uint3 (_S726, _S721, 7U);
    uint3  _S758 = make_uint3 (_S720, _S727, 7U);
    uint3  _S759 = make_uint3 (_S726, _S727, 7U);
    uint3  _S760 = make_uint3 (_S720, _S721, 8U);
    uint3  _S761 = make_uint3 (_S726, _S721, 8U);
    uint3  _S762 = make_uint3 (_S720, _S727, 8U);
    uint3  _S763 = make_uint3 (_S726, _S727, 8U);
    uint3  _S764 = make_uint3 (_S720, _S721, 9U);
    uint3  _S765 = make_uint3 (_S726, _S721, 9U);
    uint3  _S766 = make_uint3 (_S720, _S727, 9U);
    uint3  _S767 = make_uint3 (_S726, _S727, 9U);
    uint3  _S768 = make_uint3 (_S720, _S721, 10U);
    uint3  _S769 = make_uint3 (_S726, _S721, 10U);
    uint3  _S770 = make_uint3 (_S720, _S727, 10U);
    uint3  _S771 = make_uint3 (_S726, _S727, 10U);
    uint3  _S772 = make_uint3 (_S720, _S721, 11U);
    uint3  _S773 = make_uint3 (_S726, _S721, 11U);
    uint3  _S774 = make_uint3 (_S720, _S727, 11U);
    uint3  _S775 = make_uint3 (_S726, _S727, 11U);
    uint3  _S776 = make_uint3 (_S720, _S721, 12U);
    uint3  _S777 = make_uint3 (_S726, _S721, 12U);
    uint3  _S778 = make_uint3 (_S720, _S727, 12U);
    uint3  _S779 = make_uint3 (_S726, _S727, 12U);
    uint3  _S780 = make_uint3 (_S720, _S721, 13U);
    uint3  _S781 = make_uint3 (_S726, _S721, 13U);
    uint3  _S782 = make_uint3 (_S720, _S727, 13U);
    float _S783 = _S724 * _s_dOut_1.vals_0[int(13)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, make_uint3 (_S726, _S727, 13U), _S722 * _S783);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S782, _S723 * _S783);
    float _S784 = _S725 * _s_dOut_1.vals_0[int(13)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S781, _S722 * _S784);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S780, _S723 * _S784);
    float _S785 = _S724 * _s_dOut_1.vals_0[int(12)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S779, _S722 * _S785);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S778, _S723 * _S785);
    float _S786 = _S725 * _s_dOut_1.vals_0[int(12)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S777, _S722 * _S786);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S776, _S723 * _S786);
    float _S787 = _S724 * _s_dOut_1.vals_0[int(11)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S775, _S722 * _S787);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S774, _S723 * _S787);
    float _S788 = _S725 * _s_dOut_1.vals_0[int(11)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S773, _S722 * _S788);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S772, _S723 * _S788);
    float _S789 = _S724 * _s_dOut_1.vals_0[int(10)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S771, _S722 * _S789);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S770, _S723 * _S789);
    float _S790 = _S725 * _s_dOut_1.vals_0[int(10)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S769, _S722 * _S790);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S768, _S723 * _S790);
    float _S791 = _S724 * _s_dOut_1.vals_0[int(9)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S767, _S722 * _S791);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S766, _S723 * _S791);
    float _S792 = _S725 * _s_dOut_1.vals_0[int(9)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S765, _S722 * _S792);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S764, _S723 * _S792);
    float _S793 = _S724 * _s_dOut_1.vals_0[int(8)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S763, _S722 * _S793);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S762, _S723 * _S793);
    float _S794 = _S725 * _s_dOut_1.vals_0[int(8)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S761, _S722 * _S794);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S760, _S723 * _S794);
    float _S795 = _S724 * _s_dOut_1.vals_0[int(7)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S759, _S722 * _S795);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S758, _S723 * _S795);
    float _S796 = _S725 * _s_dOut_1.vals_0[int(7)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S757, _S722 * _S796);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S756, _S723 * _S796);
    float _S797 = _S724 * _s_dOut_1.vals_0[int(6)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S755, _S722 * _S797);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S754, _S723 * _S797);
    float _S798 = _S725 * _s_dOut_1.vals_0[int(6)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S753, _S722 * _S798);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S752, _S723 * _S798);
    float _S799 = _S724 * _s_dOut_1.vals_0[int(5)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S751, _S722 * _S799);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S750, _S723 * _S799);
    float _S800 = _S725 * _s_dOut_1.vals_0[int(5)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S749, _S722 * _S800);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S748, _S723 * _S800);
    float _S801 = _S724 * _s_dOut_1.vals_0[int(4)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S747, _S722 * _S801);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S746, _S723 * _S801);
    float _S802 = _S725 * _s_dOut_1.vals_0[int(4)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S745, _S722 * _S802);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S744, _S723 * _S802);
    float _S803 = _S724 * _s_dOut_1.vals_0[int(3)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S743, _S722 * _S803);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S742, _S723 * _S803);
    float _S804 = _S725 * _s_dOut_1.vals_0[int(3)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S741, _S722 * _S804);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S740, _S723 * _S804);
    float _S805 = _S724 * _s_dOut_1.vals_0[int(2)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S739, _S722 * _S805);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S738, _S723 * _S805);
    float _S806 = _S725 * _s_dOut_1.vals_0[int(2)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S737, _S722 * _S806);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S736, _S723 * _S806);
    float _S807 = _S724 * _s_dOut_1.vals_0[int(1)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S735, _S722 * _S807);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S734, _S723 * _S807);
    float _S808 = _S725 * _s_dOut_1.vals_0[int(1)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S733, _S722 * _S808);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S732, _S723 * _S808);
    float _S809 = _S724 * _s_dOut_1.vals_0[int(0)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S731, _S722 * _S809);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S730, _S723 * _S809);
    float _S810 = _S725 * _s_dOut_1.vals_0[int(0)];
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S729, _S722 * _S810);
    s_bwd_prop_DiffTensorView_load_0(featureGrid_3, _S728, _S723 * _S810);
    return;
}

__device__ void s_bwd_prop_renderImage_0(MLP_0 mlp_1, DiffTensorView_0 featureGrid_4, DiffTensorView_0 imageOutput_1, s_bwd_prop_renderImage_Intermediates_0 _s_diff_ctx_5, uint _S811)
{
    uint3  dispatchIdx_1 = ((threadIdx)) + ((blockIdx)) * ((blockDim));
    uint _S812 = dispatchIdx_1.x;
    uint _S813 = dispatchIdx_1.y;
    bool _S814 = !(_S812 >= _s_diff_ctx_5._S440 || _S813 >= _s_diff_ctx_5._S441);
    uint _S815 = __ballot_sync(_S811, _S814);
    uint _S816;
    if(_S814)
    {
        uint2  _S817 = make_uint2 (_s_diff_ctx_5._S440, _s_diff_ctx_5._S441);
        s_bwd_prop_computeInterpolatedFeature_Intermediates_0 _S818 = _s_diff_ctx_5._S436;
        Feature_0 _S819 = s_bwd_primal_computeInterpolatedFeature_0(featureGrid_4, _S817, &_S818);
        s_bwd_prop_MLP_eval_Intermediates_0 _S820 = _s_diff_ctx_5._S438;
        Feature_0 _S821 = s_bwd_primal_MLP_eval_0(mlp_1, _S819, &_S820);
        s_primal_ctx_DiffTensorView_storeOnce_0(imageOutput_1, make_uint3 (_S812, _S813, 0U), _S821.vals_1[int(0)]);
        s_primal_ctx_DiffTensorView_storeOnce_0(imageOutput_1, make_uint3 (_S812, _S813, 1U), _S821.vals_1[int(1)]);
        s_primal_ctx_DiffTensorView_storeOnce_0(imageOutput_1, make_uint3 (_S812, _S813, 2U), _S821.vals_1[int(2)]);
        uint _S822 = __ballot_sync(_S811, true);
        _S816 = _S822;
    }
    else
    {
        uint _S823 = __ballot_sync(_S811, true);
        _S816 = _S823;
    }
    uint2  _S824 = make_uint2 (0U);
    uint3  _S825 = make_uint3 (0U);
    uint _S826 = __ballot_sync(_S816, _S814);
    float _S827;
    float _S828;
    float _S829;
    uint3  _S830;
    uint3  _S831;
    uint3  _S832;
    uint2  _S833;
    if(_S814)
    {
        uint2  _S834 = make_uint2 (_s_diff_ctx_5._S440, _s_diff_ctx_5._S441);
        uint3  _S835 = make_uint3 (_S812, _S813, 0U);
        uint3  _S836 = make_uint3 (_S812, _S813, 1U);
        uint3  _S837 = make_uint3 (_S812, _S813, 2U);
        uint _S838 = __ballot_sync(_S816, true);
        _S827 = _s_diff_ctx_5._S439.vals_1[int(2)];
        _S830 = _S837;
        _S828 = _s_diff_ctx_5._S439.vals_1[int(1)];
        _S831 = _S836;
        _S829 = _s_diff_ctx_5._S439.vals_1[int(0)];
        _S832 = _S835;
        _S833 = _S834;
        _S816 = _S838;
    }
    else
    {
        uint _S839 = __ballot_sync(_S816, true);
        _S827 = 0.0f;
        _S830 = _S825;
        _S828 = 0.0f;
        _S831 = _S825;
        _S829 = 0.0f;
        _S832 = _S825;
        _S833 = _S824;
        _S816 = _S839;
    }
    uint _S840 = __ballot_sync(_S816, _S814);
    if(_S814)
    {
        DiffPair_float_0 _S841;
        (&_S841)->primal_1 = _S827;
        (&_S841)->differential_0 = 0.0f;
        s_bwd_prop_DiffTensorView_storeOnce_0(imageOutput_1, _S830, &_S841);
        DiffPair_float_0 _S842;
        (&_S842)->primal_1 = _S828;
        (&_S842)->differential_0 = 0.0f;
        s_bwd_prop_DiffTensorView_storeOnce_0(imageOutput_1, _S831, &_S842);
        DiffPair_float_0 _S843;
        (&_S843)->primal_1 = _S829;
        (&_S843)->differential_0 = 0.0f;
        s_bwd_prop_DiffTensorView_storeOnce_0(imageOutput_1, _S832, &_S843);
        FixedArray<float, 16>  _S844;
        *(&_S844[int(0)]) = 0.0f;
        *(&_S844[int(1)]) = 0.0f;
        *(&_S844[int(2)]) = 0.0f;
        *(&_S844[int(3)]) = 0.0f;
        *(&_S844[int(4)]) = 0.0f;
        *(&_S844[int(5)]) = 0.0f;
        *(&_S844[int(6)]) = 0.0f;
        *(&_S844[int(7)]) = 0.0f;
        *(&_S844[int(8)]) = 0.0f;
        *(&_S844[int(9)]) = 0.0f;
        *(&_S844[int(10)]) = 0.0f;
        *(&_S844[int(11)]) = 0.0f;
        *(&_S844[int(12)]) = 0.0f;
        *(&_S844[int(13)]) = 0.0f;
        *(&_S844[int(14)]) = 0.0f;
        *(&_S844[int(15)]) = 0.0f;
        *(&_S844[int(2)]) = _S841.differential_0;
        *(&_S844[int(1)]) = _S842.differential_0;
        *(&_S844[int(0)]) = _S843.differential_0;
        s_diff_Feature_0 _S845 = Feature_x24_syn_dzero_0();
        s_diff_Feature_0 _S846 = _S845;
        (&_S846)->vals_0 = _S844;
        DiffPair_Feature_0 _S847;
        (&_S847)->primal_1 = _s_diff_ctx_5._S437;
        (&_S847)->differential_0 = _S845;
        s_bwd_prop_MLP_eval_0(mlp_1, &_S847, _S846, _s_diff_ctx_5._S438, _S840);
        s_bwd_prop_computeInterpolatedFeature_0(featureGrid_4, _S833, _S847.differential_0, _s_diff_ctx_5._S436);
        uint _S848 = __ballot_sync(_S816, true);
    }
    return;
}

__device__ void s_bwd_renderImage_0(MLP_0 _S849, DiffTensorView_0 _S850, DiffTensorView_0 _S851, uint _S852)
{
    s_bwd_prop_renderImage_Intermediates_0 _S853;
    s_bwd_primal_renderImage_0(_S849, _S850, _S851, &_S853);
    s_bwd_prop_renderImage_0(_S849, _S850, _S851, _S853, _S852);
    return;
}

extern "C" {
__global__ void __kernel__renderImage_bwd_diff(MLP_0 _S854, DiffTensorView_0 _S855, DiffTensorView_0 _S856)
{
    uint _S857 = __ballot_sync(4294967295U, true);
    s_bwd_renderImage_0(_S854, _S855, _S856, _S857);
    return;
}

}
__device__ DiffPair_Feature_0 s_fwd_computeInterpolatedFeature_0(DiffTensorView_0 featureGrid_5, uint2  frameDim_3)
{
    uint dim1_3 = DiffTensorView_size_0(featureGrid_5, 1U) - 1U;
    uint _S858 = frameDim_3.x / (DiffTensorView_size_0(featureGrid_5, 0U) - 1U);
    uint _S859 = frameDim_3.y / dim1_3;
    uint2  cellSize_3 = make_uint2 (_S858, _S859);
    uint2  pixelId_3 = uint2 {(((threadIdx)) + ((blockIdx)) * ((blockDim))).x, (((threadIdx)) + ((blockIdx)) * ((blockDim))).y};
    uint2  cellId_4 = pixelId_3 / cellSize_3;
    float2  _S860 = make_float2 ((float)pixelId_3.x, (float)pixelId_3.y);
    float2  _S861 = make_float2 ((float)cellSize_3.x, (float)cellSize_3.y);
    float2  _S862 = make_float2 ((float)cellId_4.x, (float)cellId_4.y);
    float2  weights_6 = _S860 / _S861 - _S862;
    FixedArray<float, 16>  _S863 = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    uint _S864 = cellId_4.x;
    uint _S865 = cellId_4.y;
    float _S866 = weights_6.x;
    float _S867 = 1.0f - _S866;
    float _S868 = weights_6.y;
    float _S869 = 1.0f - _S868;
    uint _S870 = _S864 + 1U;
    uint _S871 = _S865 + 1U;
    uint3  _S872 = make_uint3 (_S864, _S865, 1U);
    uint3  _S873 = make_uint3 (_S870, _S865, 1U);
    uint3  _S874 = make_uint3 (_S864, _S871, 1U);
    uint3  _S875 = make_uint3 (_S870, _S871, 1U);
    uint3  _S876 = make_uint3 (_S864, _S865, 2U);
    uint3  _S877 = make_uint3 (_S870, _S865, 2U);
    uint3  _S878 = make_uint3 (_S864, _S871, 2U);
    uint3  _S879 = make_uint3 (_S870, _S871, 2U);
    uint3  _S880 = make_uint3 (_S864, _S865, 3U);
    uint3  _S881 = make_uint3 (_S870, _S865, 3U);
    uint3  _S882 = make_uint3 (_S864, _S871, 3U);
    uint3  _S883 = make_uint3 (_S870, _S871, 3U);
    uint3  _S884 = make_uint3 (_S864, _S865, 4U);
    uint3  _S885 = make_uint3 (_S870, _S865, 4U);
    uint3  _S886 = make_uint3 (_S864, _S871, 4U);
    uint3  _S887 = make_uint3 (_S870, _S871, 4U);
    uint3  _S888 = make_uint3 (_S864, _S865, 5U);
    uint3  _S889 = make_uint3 (_S870, _S865, 5U);
    uint3  _S890 = make_uint3 (_S864, _S871, 5U);
    uint3  _S891 = make_uint3 (_S870, _S871, 5U);
    uint3  _S892 = make_uint3 (_S864, _S865, 6U);
    uint3  _S893 = make_uint3 (_S870, _S865, 6U);
    uint3  _S894 = make_uint3 (_S864, _S871, 6U);
    uint3  _S895 = make_uint3 (_S870, _S871, 6U);
    uint3  _S896 = make_uint3 (_S864, _S865, 7U);
    uint3  _S897 = make_uint3 (_S870, _S865, 7U);
    uint3  _S898 = make_uint3 (_S864, _S871, 7U);
    uint3  _S899 = make_uint3 (_S870, _S871, 7U);
    uint3  _S900 = make_uint3 (_S864, _S865, 8U);
    uint3  _S901 = make_uint3 (_S870, _S865, 8U);
    uint3  _S902 = make_uint3 (_S864, _S871, 8U);
    uint3  _S903 = make_uint3 (_S870, _S871, 8U);
    uint3  _S904 = make_uint3 (_S864, _S865, 9U);
    uint3  _S905 = make_uint3 (_S870, _S865, 9U);
    uint3  _S906 = make_uint3 (_S864, _S871, 9U);
    uint3  _S907 = make_uint3 (_S870, _S871, 9U);
    uint3  _S908 = make_uint3 (_S864, _S865, 10U);
    uint3  _S909 = make_uint3 (_S870, _S865, 10U);
    uint3  _S910 = make_uint3 (_S864, _S871, 10U);
    uint3  _S911 = make_uint3 (_S870, _S871, 10U);
    uint3  _S912 = make_uint3 (_S864, _S865, 11U);
    uint3  _S913 = make_uint3 (_S870, _S865, 11U);
    uint3  _S914 = make_uint3 (_S864, _S871, 11U);
    uint3  _S915 = make_uint3 (_S870, _S871, 11U);
    uint3  _S916 = make_uint3 (_S864, _S865, 12U);
    uint3  _S917 = make_uint3 (_S870, _S865, 12U);
    uint3  _S918 = make_uint3 (_S864, _S871, 12U);
    uint3  _S919 = make_uint3 (_S870, _S871, 12U);
    uint3  _S920 = make_uint3 (_S864, _S865, 13U);
    uint3  _S921 = make_uint3 (_S870, _S865, 13U);
    uint3  _S922 = make_uint3 (_S864, _S871, 13U);
    uint3  _S923 = make_uint3 (_S870, _S871, 13U);
    DiffPair_float_0 _S924 = DiffTensorView_load_forward_0(featureGrid_5, make_uint3 (_S864, _S865, 0U));
    DiffPair_float_0 _S925 = DiffTensorView_load_forward_0(featureGrid_5, make_uint3 (_S870, _S865, 0U));
    DiffPair_float_0 _S926 = DiffTensorView_load_forward_0(featureGrid_5, make_uint3 (_S864, _S871, 0U));
    DiffPair_float_0 _S927 = DiffTensorView_load_forward_0(featureGrid_5, make_uint3 (_S870, _S871, 0U));
    float _S928 = _S924.primal_1 * _S867 * _S869 + _S925.primal_1 * _S866 * _S869 + _S926.primal_1 * _S867 * _S868 + _S927.primal_1 * _S866 * _S868;
    float _S929 = _S924.differential_0 * _S867 * _S869 + _S925.differential_0 * _S866 * _S869 + _S926.differential_0 * _S867 * _S868 + _S927.differential_0 * _S866 * _S868;
    Feature_0 _S930;
    (&_S930)->vals_1 = _S863;
    *(&(&_S930)->vals_1[int(0)]) = _S928;
    s_diff_Feature_0 _S931;
    (&_S931)->vals_0 = _S863;
    *(&(&_S931)->vals_0[int(0)]) = _S929;
    DiffPair_float_0 _S932 = DiffTensorView_load_forward_0(featureGrid_5, _S872);
    DiffPair_float_0 _S933 = DiffTensorView_load_forward_0(featureGrid_5, _S873);
    DiffPair_float_0 _S934 = DiffTensorView_load_forward_0(featureGrid_5, _S874);
    DiffPair_float_0 _S935 = DiffTensorView_load_forward_0(featureGrid_5, _S875);
    float _S936 = _S932.differential_0 * _S867 * _S869 + _S933.differential_0 * _S866 * _S869 + _S934.differential_0 * _S867 * _S868 + _S935.differential_0 * _S866 * _S868;
    *(&(&_S930)->vals_1[int(1)]) = _S932.primal_1 * _S867 * _S869 + _S933.primal_1 * _S866 * _S869 + _S934.primal_1 * _S867 * _S868 + _S935.primal_1 * _S866 * _S868;
    *(&(&_S931)->vals_0[int(1)]) = _S936;
    DiffPair_float_0 _S937 = DiffTensorView_load_forward_0(featureGrid_5, _S876);
    DiffPair_float_0 _S938 = DiffTensorView_load_forward_0(featureGrid_5, _S877);
    DiffPair_float_0 _S939 = DiffTensorView_load_forward_0(featureGrid_5, _S878);
    DiffPair_float_0 _S940 = DiffTensorView_load_forward_0(featureGrid_5, _S879);
    float _S941 = _S937.differential_0 * _S867 * _S869 + _S938.differential_0 * _S866 * _S869 + _S939.differential_0 * _S867 * _S868 + _S940.differential_0 * _S866 * _S868;
    *(&(&_S930)->vals_1[int(2)]) = _S937.primal_1 * _S867 * _S869 + _S938.primal_1 * _S866 * _S869 + _S939.primal_1 * _S867 * _S868 + _S940.primal_1 * _S866 * _S868;
    *(&(&_S931)->vals_0[int(2)]) = _S941;
    DiffPair_float_0 _S942 = DiffTensorView_load_forward_0(featureGrid_5, _S880);
    DiffPair_float_0 _S943 = DiffTensorView_load_forward_0(featureGrid_5, _S881);
    DiffPair_float_0 _S944 = DiffTensorView_load_forward_0(featureGrid_5, _S882);
    DiffPair_float_0 _S945 = DiffTensorView_load_forward_0(featureGrid_5, _S883);
    float _S946 = _S942.differential_0 * _S867 * _S869 + _S943.differential_0 * _S866 * _S869 + _S944.differential_0 * _S867 * _S868 + _S945.differential_0 * _S866 * _S868;
    *(&(&_S930)->vals_1[int(3)]) = _S942.primal_1 * _S867 * _S869 + _S943.primal_1 * _S866 * _S869 + _S944.primal_1 * _S867 * _S868 + _S945.primal_1 * _S866 * _S868;
    *(&(&_S931)->vals_0[int(3)]) = _S946;
    DiffPair_float_0 _S947 = DiffTensorView_load_forward_0(featureGrid_5, _S884);
    DiffPair_float_0 _S948 = DiffTensorView_load_forward_0(featureGrid_5, _S885);
    DiffPair_float_0 _S949 = DiffTensorView_load_forward_0(featureGrid_5, _S886);
    DiffPair_float_0 _S950 = DiffTensorView_load_forward_0(featureGrid_5, _S887);
    float _S951 = _S947.differential_0 * _S867 * _S869 + _S948.differential_0 * _S866 * _S869 + _S949.differential_0 * _S867 * _S868 + _S950.differential_0 * _S866 * _S868;
    *(&(&_S930)->vals_1[int(4)]) = _S947.primal_1 * _S867 * _S869 + _S948.primal_1 * _S866 * _S869 + _S949.primal_1 * _S867 * _S868 + _S950.primal_1 * _S866 * _S868;
    *(&(&_S931)->vals_0[int(4)]) = _S951;
    DiffPair_float_0 _S952 = DiffTensorView_load_forward_0(featureGrid_5, _S888);
    DiffPair_float_0 _S953 = DiffTensorView_load_forward_0(featureGrid_5, _S889);
    DiffPair_float_0 _S954 = DiffTensorView_load_forward_0(featureGrid_5, _S890);
    DiffPair_float_0 _S955 = DiffTensorView_load_forward_0(featureGrid_5, _S891);
    float _S956 = _S952.differential_0 * _S867 * _S869 + _S953.differential_0 * _S866 * _S869 + _S954.differential_0 * _S867 * _S868 + _S955.differential_0 * _S866 * _S868;
    *(&(&_S930)->vals_1[int(5)]) = _S952.primal_1 * _S867 * _S869 + _S953.primal_1 * _S866 * _S869 + _S954.primal_1 * _S867 * _S868 + _S955.primal_1 * _S866 * _S868;
    *(&(&_S931)->vals_0[int(5)]) = _S956;
    DiffPair_float_0 _S957 = DiffTensorView_load_forward_0(featureGrid_5, _S892);
    DiffPair_float_0 _S958 = DiffTensorView_load_forward_0(featureGrid_5, _S893);
    DiffPair_float_0 _S959 = DiffTensorView_load_forward_0(featureGrid_5, _S894);
    DiffPair_float_0 _S960 = DiffTensorView_load_forward_0(featureGrid_5, _S895);
    float _S961 = _S957.differential_0 * _S867 * _S869 + _S958.differential_0 * _S866 * _S869 + _S959.differential_0 * _S867 * _S868 + _S960.differential_0 * _S866 * _S868;
    *(&(&_S930)->vals_1[int(6)]) = _S957.primal_1 * _S867 * _S869 + _S958.primal_1 * _S866 * _S869 + _S959.primal_1 * _S867 * _S868 + _S960.primal_1 * _S866 * _S868;
    *(&(&_S931)->vals_0[int(6)]) = _S961;
    DiffPair_float_0 _S962 = DiffTensorView_load_forward_0(featureGrid_5, _S896);
    DiffPair_float_0 _S963 = DiffTensorView_load_forward_0(featureGrid_5, _S897);
    DiffPair_float_0 _S964 = DiffTensorView_load_forward_0(featureGrid_5, _S898);
    DiffPair_float_0 _S965 = DiffTensorView_load_forward_0(featureGrid_5, _S899);
    float _S966 = _S962.differential_0 * _S867 * _S869 + _S963.differential_0 * _S866 * _S869 + _S964.differential_0 * _S867 * _S868 + _S965.differential_0 * _S866 * _S868;
    *(&(&_S930)->vals_1[int(7)]) = _S962.primal_1 * _S867 * _S869 + _S963.primal_1 * _S866 * _S869 + _S964.primal_1 * _S867 * _S868 + _S965.primal_1 * _S866 * _S868;
    *(&(&_S931)->vals_0[int(7)]) = _S966;
    DiffPair_float_0 _S967 = DiffTensorView_load_forward_0(featureGrid_5, _S900);
    DiffPair_float_0 _S968 = DiffTensorView_load_forward_0(featureGrid_5, _S901);
    DiffPair_float_0 _S969 = DiffTensorView_load_forward_0(featureGrid_5, _S902);
    DiffPair_float_0 _S970 = DiffTensorView_load_forward_0(featureGrid_5, _S903);
    float _S971 = _S967.differential_0 * _S867 * _S869 + _S968.differential_0 * _S866 * _S869 + _S969.differential_0 * _S867 * _S868 + _S970.differential_0 * _S866 * _S868;
    *(&(&_S930)->vals_1[int(8)]) = _S967.primal_1 * _S867 * _S869 + _S968.primal_1 * _S866 * _S869 + _S969.primal_1 * _S867 * _S868 + _S970.primal_1 * _S866 * _S868;
    *(&(&_S931)->vals_0[int(8)]) = _S971;
    DiffPair_float_0 _S972 = DiffTensorView_load_forward_0(featureGrid_5, _S904);
    DiffPair_float_0 _S973 = DiffTensorView_load_forward_0(featureGrid_5, _S905);
    DiffPair_float_0 _S974 = DiffTensorView_load_forward_0(featureGrid_5, _S906);
    DiffPair_float_0 _S975 = DiffTensorView_load_forward_0(featureGrid_5, _S907);
    float _S976 = _S972.differential_0 * _S867 * _S869 + _S973.differential_0 * _S866 * _S869 + _S974.differential_0 * _S867 * _S868 + _S975.differential_0 * _S866 * _S868;
    *(&(&_S930)->vals_1[int(9)]) = _S972.primal_1 * _S867 * _S869 + _S973.primal_1 * _S866 * _S869 + _S974.primal_1 * _S867 * _S868 + _S975.primal_1 * _S866 * _S868;
    *(&(&_S931)->vals_0[int(9)]) = _S976;
    DiffPair_float_0 _S977 = DiffTensorView_load_forward_0(featureGrid_5, _S908);
    DiffPair_float_0 _S978 = DiffTensorView_load_forward_0(featureGrid_5, _S909);
    DiffPair_float_0 _S979 = DiffTensorView_load_forward_0(featureGrid_5, _S910);
    DiffPair_float_0 _S980 = DiffTensorView_load_forward_0(featureGrid_5, _S911);
    float _S981 = _S977.differential_0 * _S867 * _S869 + _S978.differential_0 * _S866 * _S869 + _S979.differential_0 * _S867 * _S868 + _S980.differential_0 * _S866 * _S868;
    *(&(&_S930)->vals_1[int(10)]) = _S977.primal_1 * _S867 * _S869 + _S978.primal_1 * _S866 * _S869 + _S979.primal_1 * _S867 * _S868 + _S980.primal_1 * _S866 * _S868;
    *(&(&_S931)->vals_0[int(10)]) = _S981;
    DiffPair_float_0 _S982 = DiffTensorView_load_forward_0(featureGrid_5, _S912);
    DiffPair_float_0 _S983 = DiffTensorView_load_forward_0(featureGrid_5, _S913);
    DiffPair_float_0 _S984 = DiffTensorView_load_forward_0(featureGrid_5, _S914);
    DiffPair_float_0 _S985 = DiffTensorView_load_forward_0(featureGrid_5, _S915);
    float _S986 = _S982.differential_0 * _S867 * _S869 + _S983.differential_0 * _S866 * _S869 + _S984.differential_0 * _S867 * _S868 + _S985.differential_0 * _S866 * _S868;
    *(&(&_S930)->vals_1[int(11)]) = _S982.primal_1 * _S867 * _S869 + _S983.primal_1 * _S866 * _S869 + _S984.primal_1 * _S867 * _S868 + _S985.primal_1 * _S866 * _S868;
    *(&(&_S931)->vals_0[int(11)]) = _S986;
    DiffPair_float_0 _S987 = DiffTensorView_load_forward_0(featureGrid_5, _S916);
    DiffPair_float_0 _S988 = DiffTensorView_load_forward_0(featureGrid_5, _S917);
    DiffPair_float_0 _S989 = DiffTensorView_load_forward_0(featureGrid_5, _S918);
    DiffPair_float_0 _S990 = DiffTensorView_load_forward_0(featureGrid_5, _S919);
    float _S991 = _S987.differential_0 * _S867 * _S869 + _S988.differential_0 * _S866 * _S869 + _S989.differential_0 * _S867 * _S868 + _S990.differential_0 * _S866 * _S868;
    *(&(&_S930)->vals_1[int(12)]) = _S987.primal_1 * _S867 * _S869 + _S988.primal_1 * _S866 * _S869 + _S989.primal_1 * _S867 * _S868 + _S990.primal_1 * _S866 * _S868;
    *(&(&_S931)->vals_0[int(12)]) = _S991;
    DiffPair_float_0 _S992 = DiffTensorView_load_forward_0(featureGrid_5, _S920);
    DiffPair_float_0 _S993 = DiffTensorView_load_forward_0(featureGrid_5, _S921);
    DiffPair_float_0 _S994 = DiffTensorView_load_forward_0(featureGrid_5, _S922);
    DiffPair_float_0 _S995 = DiffTensorView_load_forward_0(featureGrid_5, _S923);
    float _S996 = _S992.differential_0 * _S867 * _S869 + _S993.differential_0 * _S866 * _S869 + _S994.differential_0 * _S867 * _S868 + _S995.differential_0 * _S866 * _S868;
    *(&(&_S930)->vals_1[int(13)]) = _S992.primal_1 * _S867 * _S869 + _S993.primal_1 * _S866 * _S869 + _S994.primal_1 * _S867 * _S868 + _S995.primal_1 * _S866 * _S868;
    *(&(&_S931)->vals_0[int(13)]) = _S996;
    *(&(&_S930)->vals_1[14U]) = _S866;
    *(&(&_S931)->vals_0[14U]) = 0.0f;
    *(&(&_S930)->vals_1[15U]) = _S868;
    *(&(&_S931)->vals_0[15U]) = 0.0f;
    DiffPair_Feature_0 _S997 = { _S930, _S931 };
    return _S997;
}

__device__ DiffPair_Feature_0 s_fwd_Linear_eval_0(Linear_0 this_31, DiffPair_Feature_0 dpin_feature_2)
{
    float * inPtr_4 = Linear_moveInputsToSharedMem_0(this_31, dpin_feature_2.primal_1.vals_1);
    float * wtPtr_5 = Linear_moveWeightsToSharedMem_0(this_31);
    float * outPtr_4 = Linear_outBufferForCurrentWarp_0(this_31);
    float _S998 = *inPtr_4;
    float _S999 = *wtPtr_5;
    float _S1000 = *outPtr_4;
    _inline_matmul_1(&_S998, &_S999, &_S1000);
    *outPtr_4 = _S1000;
    *wtPtr_5 = _S999;
    *inPtr_4 = _S998;
    FixedArray<float, 16>  _S1001 = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    float * _S1002 = ((this_31.bias_0.primal_0).data_ptr<float>());
    FixedArray<float, 16>  _S1003;
    *(&_S1003[int(0)]) = 0.0f;
    *(&_S1003[int(1)]) = 0.0f;
    *(&_S1003[int(2)]) = 0.0f;
    *(&_S1003[int(3)]) = 0.0f;
    *(&_S1003[int(4)]) = 0.0f;
    *(&_S1003[int(5)]) = 0.0f;
    *(&_S1003[int(6)]) = 0.0f;
    *(&_S1003[int(7)]) = 0.0f;
    *(&_S1003[int(8)]) = 0.0f;
    *(&_S1003[int(9)]) = 0.0f;
    *(&_S1003[int(10)]) = 0.0f;
    *(&_S1003[int(11)]) = 0.0f;
    *(&_S1003[int(12)]) = 0.0f;
    *(&_S1003[int(13)]) = 0.0f;
    *(&_S1003[int(14)]) = 0.0f;
    *(&_S1003[int(15)]) = 0.0f;
    float _S1004 = *_S1002;
    Linear_moveOutputsToLocalArray_0(this_31, &_S1003, &_S1004);
    *_S1002 = _S1004;
    Feature_0 _S1005 = { _S1003 };
    s_diff_Feature_0 _S1006 = { _S1001 };
    DiffPair_Feature_0 _S1007 = { _S1005, _S1006 };
    return _S1007;
}

__device__ DiffPair_Feature_0 s_fwd_MLP_eval_0(MLP_0 this_32, DiffPair_Feature_0 dpin_feature_3)
{
    DiffPair_Feature_0 _S1008 = { dpin_feature_3.primal_1, dpin_feature_3.differential_0 };
    DiffPair_float_0 _S1009 = { 0.0f, 0.0f };
    DiffPair_Feature_0 _S1010 = s_fwd_Linear_eval_0(this_32.layers_0[int(0)], _S1008);
    DiffPair_float_0 _S1011 = { _S1010.primal_1.vals_1[int(0)], _S1010.differential_0.vals_0[int(0)] };
    DiffPair_float_0 _S1012 = _d_max_1(_S1009, _S1011);
    Feature_0 _S1013 = _S1010.primal_1;
    *(&(&_S1013)->vals_1[int(0)]) = _S1012.primal_1;
    s_diff_Feature_0 _S1014 = _S1010.differential_0;
    *(&(&_S1014)->vals_0[int(0)]) = _S1012.differential_0;
    DiffPair_float_0 _S1015 = { _S1010.primal_1.vals_1[int(1)], _S1010.differential_0.vals_0[int(1)] };
    DiffPair_float_0 _S1016 = _d_max_1(_S1009, _S1015);
    *(&(&_S1013)->vals_1[int(1)]) = _S1016.primal_1;
    *(&(&_S1014)->vals_0[int(1)]) = _S1016.differential_0;
    DiffPair_float_0 _S1017 = { _S1010.primal_1.vals_1[int(2)], _S1010.differential_0.vals_0[int(2)] };
    DiffPair_float_0 _S1018 = _d_max_1(_S1009, _S1017);
    *(&(&_S1013)->vals_1[int(2)]) = _S1018.primal_1;
    *(&(&_S1014)->vals_0[int(2)]) = _S1018.differential_0;
    DiffPair_float_0 _S1019 = { _S1010.primal_1.vals_1[int(3)], _S1010.differential_0.vals_0[int(3)] };
    DiffPair_float_0 _S1020 = _d_max_1(_S1009, _S1019);
    *(&(&_S1013)->vals_1[int(3)]) = _S1020.primal_1;
    *(&(&_S1014)->vals_0[int(3)]) = _S1020.differential_0;
    DiffPair_float_0 _S1021 = { _S1010.primal_1.vals_1[int(4)], _S1010.differential_0.vals_0[int(4)] };
    DiffPair_float_0 _S1022 = _d_max_1(_S1009, _S1021);
    *(&(&_S1013)->vals_1[int(4)]) = _S1022.primal_1;
    *(&(&_S1014)->vals_0[int(4)]) = _S1022.differential_0;
    DiffPair_float_0 _S1023 = { _S1010.primal_1.vals_1[int(5)], _S1010.differential_0.vals_0[int(5)] };
    DiffPair_float_0 _S1024 = _d_max_1(_S1009, _S1023);
    *(&(&_S1013)->vals_1[int(5)]) = _S1024.primal_1;
    *(&(&_S1014)->vals_0[int(5)]) = _S1024.differential_0;
    DiffPair_float_0 _S1025 = { _S1010.primal_1.vals_1[int(6)], _S1010.differential_0.vals_0[int(6)] };
    DiffPair_float_0 _S1026 = _d_max_1(_S1009, _S1025);
    *(&(&_S1013)->vals_1[int(6)]) = _S1026.primal_1;
    *(&(&_S1014)->vals_0[int(6)]) = _S1026.differential_0;
    DiffPair_float_0 _S1027 = { _S1010.primal_1.vals_1[int(7)], _S1010.differential_0.vals_0[int(7)] };
    DiffPair_float_0 _S1028 = _d_max_1(_S1009, _S1027);
    *(&(&_S1013)->vals_1[int(7)]) = _S1028.primal_1;
    *(&(&_S1014)->vals_0[int(7)]) = _S1028.differential_0;
    DiffPair_float_0 _S1029 = { _S1010.primal_1.vals_1[int(8)], _S1010.differential_0.vals_0[int(8)] };
    DiffPair_float_0 _S1030 = _d_max_1(_S1009, _S1029);
    *(&(&_S1013)->vals_1[int(8)]) = _S1030.primal_1;
    *(&(&_S1014)->vals_0[int(8)]) = _S1030.differential_0;
    DiffPair_float_0 _S1031 = { _S1010.primal_1.vals_1[int(9)], _S1010.differential_0.vals_0[int(9)] };
    DiffPair_float_0 _S1032 = _d_max_1(_S1009, _S1031);
    *(&(&_S1013)->vals_1[int(9)]) = _S1032.primal_1;
    *(&(&_S1014)->vals_0[int(9)]) = _S1032.differential_0;
    DiffPair_float_0 _S1033 = { _S1010.primal_1.vals_1[int(10)], _S1010.differential_0.vals_0[int(10)] };
    DiffPair_float_0 _S1034 = _d_max_1(_S1009, _S1033);
    *(&(&_S1013)->vals_1[int(10)]) = _S1034.primal_1;
    *(&(&_S1014)->vals_0[int(10)]) = _S1034.differential_0;
    DiffPair_float_0 _S1035 = { _S1010.primal_1.vals_1[int(11)], _S1010.differential_0.vals_0[int(11)] };
    DiffPair_float_0 _S1036 = _d_max_1(_S1009, _S1035);
    *(&(&_S1013)->vals_1[int(11)]) = _S1036.primal_1;
    *(&(&_S1014)->vals_0[int(11)]) = _S1036.differential_0;
    DiffPair_float_0 _S1037 = { _S1010.primal_1.vals_1[int(12)], _S1010.differential_0.vals_0[int(12)] };
    DiffPair_float_0 _S1038 = _d_max_1(_S1009, _S1037);
    *(&(&_S1013)->vals_1[int(12)]) = _S1038.primal_1;
    *(&(&_S1014)->vals_0[int(12)]) = _S1038.differential_0;
    DiffPair_float_0 _S1039 = { _S1010.primal_1.vals_1[int(13)], _S1010.differential_0.vals_0[int(13)] };
    DiffPair_float_0 _S1040 = _d_max_1(_S1009, _S1039);
    *(&(&_S1013)->vals_1[int(13)]) = _S1040.primal_1;
    *(&(&_S1014)->vals_0[int(13)]) = _S1040.differential_0;
    DiffPair_float_0 _S1041 = { _S1010.primal_1.vals_1[int(14)], _S1010.differential_0.vals_0[int(14)] };
    DiffPair_float_0 _S1042 = _d_max_1(_S1009, _S1041);
    *(&(&_S1013)->vals_1[int(14)]) = _S1042.primal_1;
    *(&(&_S1014)->vals_0[int(14)]) = _S1042.differential_0;
    DiffPair_float_0 _S1043 = { _S1010.primal_1.vals_1[int(15)], _S1010.differential_0.vals_0[int(15)] };
    DiffPair_float_0 _S1044 = _d_max_1(_S1009, _S1043);
    *(&(&_S1013)->vals_1[int(15)]) = _S1044.primal_1;
    *(&(&_S1014)->vals_0[int(15)]) = _S1044.differential_0;
    DiffPair_Feature_0 _S1045 = { _S1013, _S1014 };
    DiffPair_Feature_0 _S1046 = s_fwd_Linear_eval_0(this_32.layers_0[int(1)], _S1045);
    DiffPair_float_0 _S1047 = { _S1046.primal_1.vals_1[int(0)], _S1046.differential_0.vals_0[int(0)] };
    DiffPair_float_0 _S1048 = _d_max_1(_S1009, _S1047);
    _S1013 = _S1046.primal_1;
    *(&(&_S1013)->vals_1[int(0)]) = _S1048.primal_1;
    _S1014 = _S1046.differential_0;
    *(&(&_S1014)->vals_0[int(0)]) = _S1048.differential_0;
    DiffPair_float_0 _S1049 = { _S1046.primal_1.vals_1[int(1)], _S1046.differential_0.vals_0[int(1)] };
    DiffPair_float_0 _S1050 = _d_max_1(_S1009, _S1049);
    *(&(&_S1013)->vals_1[int(1)]) = _S1050.primal_1;
    *(&(&_S1014)->vals_0[int(1)]) = _S1050.differential_0;
    DiffPair_float_0 _S1051 = { _S1046.primal_1.vals_1[int(2)], _S1046.differential_0.vals_0[int(2)] };
    DiffPair_float_0 _S1052 = _d_max_1(_S1009, _S1051);
    *(&(&_S1013)->vals_1[int(2)]) = _S1052.primal_1;
    *(&(&_S1014)->vals_0[int(2)]) = _S1052.differential_0;
    DiffPair_float_0 _S1053 = { _S1046.primal_1.vals_1[int(3)], _S1046.differential_0.vals_0[int(3)] };
    DiffPair_float_0 _S1054 = _d_max_1(_S1009, _S1053);
    *(&(&_S1013)->vals_1[int(3)]) = _S1054.primal_1;
    *(&(&_S1014)->vals_0[int(3)]) = _S1054.differential_0;
    DiffPair_float_0 _S1055 = { _S1046.primal_1.vals_1[int(4)], _S1046.differential_0.vals_0[int(4)] };
    DiffPair_float_0 _S1056 = _d_max_1(_S1009, _S1055);
    *(&(&_S1013)->vals_1[int(4)]) = _S1056.primal_1;
    *(&(&_S1014)->vals_0[int(4)]) = _S1056.differential_0;
    DiffPair_float_0 _S1057 = { _S1046.primal_1.vals_1[int(5)], _S1046.differential_0.vals_0[int(5)] };
    DiffPair_float_0 _S1058 = _d_max_1(_S1009, _S1057);
    *(&(&_S1013)->vals_1[int(5)]) = _S1058.primal_1;
    *(&(&_S1014)->vals_0[int(5)]) = _S1058.differential_0;
    DiffPair_float_0 _S1059 = { _S1046.primal_1.vals_1[int(6)], _S1046.differential_0.vals_0[int(6)] };
    DiffPair_float_0 _S1060 = _d_max_1(_S1009, _S1059);
    *(&(&_S1013)->vals_1[int(6)]) = _S1060.primal_1;
    *(&(&_S1014)->vals_0[int(6)]) = _S1060.differential_0;
    DiffPair_float_0 _S1061 = { _S1046.primal_1.vals_1[int(7)], _S1046.differential_0.vals_0[int(7)] };
    DiffPair_float_0 _S1062 = _d_max_1(_S1009, _S1061);
    *(&(&_S1013)->vals_1[int(7)]) = _S1062.primal_1;
    *(&(&_S1014)->vals_0[int(7)]) = _S1062.differential_0;
    DiffPair_float_0 _S1063 = { _S1046.primal_1.vals_1[int(8)], _S1046.differential_0.vals_0[int(8)] };
    DiffPair_float_0 _S1064 = _d_max_1(_S1009, _S1063);
    *(&(&_S1013)->vals_1[int(8)]) = _S1064.primal_1;
    *(&(&_S1014)->vals_0[int(8)]) = _S1064.differential_0;
    DiffPair_float_0 _S1065 = { _S1046.primal_1.vals_1[int(9)], _S1046.differential_0.vals_0[int(9)] };
    DiffPair_float_0 _S1066 = _d_max_1(_S1009, _S1065);
    *(&(&_S1013)->vals_1[int(9)]) = _S1066.primal_1;
    *(&(&_S1014)->vals_0[int(9)]) = _S1066.differential_0;
    DiffPair_float_0 _S1067 = { _S1046.primal_1.vals_1[int(10)], _S1046.differential_0.vals_0[int(10)] };
    DiffPair_float_0 _S1068 = _d_max_1(_S1009, _S1067);
    *(&(&_S1013)->vals_1[int(10)]) = _S1068.primal_1;
    *(&(&_S1014)->vals_0[int(10)]) = _S1068.differential_0;
    DiffPair_float_0 _S1069 = { _S1046.primal_1.vals_1[int(11)], _S1046.differential_0.vals_0[int(11)] };
    DiffPair_float_0 _S1070 = _d_max_1(_S1009, _S1069);
    *(&(&_S1013)->vals_1[int(11)]) = _S1070.primal_1;
    *(&(&_S1014)->vals_0[int(11)]) = _S1070.differential_0;
    DiffPair_float_0 _S1071 = { _S1046.primal_1.vals_1[int(12)], _S1046.differential_0.vals_0[int(12)] };
    DiffPair_float_0 _S1072 = _d_max_1(_S1009, _S1071);
    *(&(&_S1013)->vals_1[int(12)]) = _S1072.primal_1;
    *(&(&_S1014)->vals_0[int(12)]) = _S1072.differential_0;
    DiffPair_float_0 _S1073 = { _S1046.primal_1.vals_1[int(13)], _S1046.differential_0.vals_0[int(13)] };
    DiffPair_float_0 _S1074 = _d_max_1(_S1009, _S1073);
    *(&(&_S1013)->vals_1[int(13)]) = _S1074.primal_1;
    *(&(&_S1014)->vals_0[int(13)]) = _S1074.differential_0;
    DiffPair_float_0 _S1075 = { _S1046.primal_1.vals_1[int(14)], _S1046.differential_0.vals_0[int(14)] };
    DiffPair_float_0 _S1076 = _d_max_1(_S1009, _S1075);
    *(&(&_S1013)->vals_1[int(14)]) = _S1076.primal_1;
    *(&(&_S1014)->vals_0[int(14)]) = _S1076.differential_0;
    DiffPair_float_0 _S1077 = { _S1046.primal_1.vals_1[int(15)], _S1046.differential_0.vals_0[int(15)] };
    DiffPair_float_0 _S1078 = _d_max_1(_S1009, _S1077);
    *(&(&_S1013)->vals_1[int(15)]) = _S1078.primal_1;
    *(&(&_S1014)->vals_0[int(15)]) = _S1078.differential_0;
    DiffPair_Feature_0 _S1079 = { _S1013, _S1014 };
    DiffPair_Feature_0 _S1080 = s_fwd_Linear_eval_0(this_32.layers_0[int(2)], _S1079);
    DiffPair_float_0 _S1081 = { _S1080.primal_1.vals_1[int(0)], _S1080.differential_0.vals_0[int(0)] };
    DiffPair_float_0 _S1082 = _d_max_1(_S1009, _S1081);
    _S1013 = _S1080.primal_1;
    *(&(&_S1013)->vals_1[int(0)]) = _S1082.primal_1;
    _S1014 = _S1080.differential_0;
    *(&(&_S1014)->vals_0[int(0)]) = _S1082.differential_0;
    DiffPair_float_0 _S1083 = { _S1080.primal_1.vals_1[int(1)], _S1080.differential_0.vals_0[int(1)] };
    DiffPair_float_0 _S1084 = _d_max_1(_S1009, _S1083);
    *(&(&_S1013)->vals_1[int(1)]) = _S1084.primal_1;
    *(&(&_S1014)->vals_0[int(1)]) = _S1084.differential_0;
    DiffPair_float_0 _S1085 = { _S1080.primal_1.vals_1[int(2)], _S1080.differential_0.vals_0[int(2)] };
    DiffPair_float_0 _S1086 = _d_max_1(_S1009, _S1085);
    *(&(&_S1013)->vals_1[int(2)]) = _S1086.primal_1;
    *(&(&_S1014)->vals_0[int(2)]) = _S1086.differential_0;
    DiffPair_float_0 _S1087 = { _S1080.primal_1.vals_1[int(3)], _S1080.differential_0.vals_0[int(3)] };
    DiffPair_float_0 _S1088 = _d_max_1(_S1009, _S1087);
    *(&(&_S1013)->vals_1[int(3)]) = _S1088.primal_1;
    *(&(&_S1014)->vals_0[int(3)]) = _S1088.differential_0;
    DiffPair_float_0 _S1089 = { _S1080.primal_1.vals_1[int(4)], _S1080.differential_0.vals_0[int(4)] };
    DiffPair_float_0 _S1090 = _d_max_1(_S1009, _S1089);
    *(&(&_S1013)->vals_1[int(4)]) = _S1090.primal_1;
    *(&(&_S1014)->vals_0[int(4)]) = _S1090.differential_0;
    DiffPair_float_0 _S1091 = { _S1080.primal_1.vals_1[int(5)], _S1080.differential_0.vals_0[int(5)] };
    DiffPair_float_0 _S1092 = _d_max_1(_S1009, _S1091);
    *(&(&_S1013)->vals_1[int(5)]) = _S1092.primal_1;
    *(&(&_S1014)->vals_0[int(5)]) = _S1092.differential_0;
    DiffPair_float_0 _S1093 = { _S1080.primal_1.vals_1[int(6)], _S1080.differential_0.vals_0[int(6)] };
    DiffPair_float_0 _S1094 = _d_max_1(_S1009, _S1093);
    *(&(&_S1013)->vals_1[int(6)]) = _S1094.primal_1;
    *(&(&_S1014)->vals_0[int(6)]) = _S1094.differential_0;
    DiffPair_float_0 _S1095 = { _S1080.primal_1.vals_1[int(7)], _S1080.differential_0.vals_0[int(7)] };
    DiffPair_float_0 _S1096 = _d_max_1(_S1009, _S1095);
    *(&(&_S1013)->vals_1[int(7)]) = _S1096.primal_1;
    *(&(&_S1014)->vals_0[int(7)]) = _S1096.differential_0;
    DiffPair_float_0 _S1097 = { _S1080.primal_1.vals_1[int(8)], _S1080.differential_0.vals_0[int(8)] };
    DiffPair_float_0 _S1098 = _d_max_1(_S1009, _S1097);
    *(&(&_S1013)->vals_1[int(8)]) = _S1098.primal_1;
    *(&(&_S1014)->vals_0[int(8)]) = _S1098.differential_0;
    DiffPair_float_0 _S1099 = { _S1080.primal_1.vals_1[int(9)], _S1080.differential_0.vals_0[int(9)] };
    DiffPair_float_0 _S1100 = _d_max_1(_S1009, _S1099);
    *(&(&_S1013)->vals_1[int(9)]) = _S1100.primal_1;
    *(&(&_S1014)->vals_0[int(9)]) = _S1100.differential_0;
    DiffPair_float_0 _S1101 = { _S1080.primal_1.vals_1[int(10)], _S1080.differential_0.vals_0[int(10)] };
    DiffPair_float_0 _S1102 = _d_max_1(_S1009, _S1101);
    *(&(&_S1013)->vals_1[int(10)]) = _S1102.primal_1;
    *(&(&_S1014)->vals_0[int(10)]) = _S1102.differential_0;
    DiffPair_float_0 _S1103 = { _S1080.primal_1.vals_1[int(11)], _S1080.differential_0.vals_0[int(11)] };
    DiffPair_float_0 _S1104 = _d_max_1(_S1009, _S1103);
    *(&(&_S1013)->vals_1[int(11)]) = _S1104.primal_1;
    *(&(&_S1014)->vals_0[int(11)]) = _S1104.differential_0;
    DiffPair_float_0 _S1105 = { _S1080.primal_1.vals_1[int(12)], _S1080.differential_0.vals_0[int(12)] };
    DiffPair_float_0 _S1106 = _d_max_1(_S1009, _S1105);
    *(&(&_S1013)->vals_1[int(12)]) = _S1106.primal_1;
    *(&(&_S1014)->vals_0[int(12)]) = _S1106.differential_0;
    DiffPair_float_0 _S1107 = { _S1080.primal_1.vals_1[int(13)], _S1080.differential_0.vals_0[int(13)] };
    DiffPair_float_0 _S1108 = _d_max_1(_S1009, _S1107);
    *(&(&_S1013)->vals_1[int(13)]) = _S1108.primal_1;
    *(&(&_S1014)->vals_0[int(13)]) = _S1108.differential_0;
    DiffPair_float_0 _S1109 = { _S1080.primal_1.vals_1[int(14)], _S1080.differential_0.vals_0[int(14)] };
    DiffPair_float_0 _S1110 = _d_max_1(_S1009, _S1109);
    *(&(&_S1013)->vals_1[int(14)]) = _S1110.primal_1;
    *(&(&_S1014)->vals_0[int(14)]) = _S1110.differential_0;
    DiffPair_float_0 _S1111 = { _S1080.primal_1.vals_1[int(15)], _S1080.differential_0.vals_0[int(15)] };
    DiffPair_float_0 _S1112 = _d_max_1(_S1009, _S1111);
    *(&(&_S1013)->vals_1[int(15)]) = _S1112.primal_1;
    *(&(&_S1014)->vals_0[int(15)]) = _S1112.differential_0;
    DiffPair_Feature_0 _S1113 = { _S1013, _S1014 };
    return _S1113;
}

__device__ void s_fwd_renderImage_0(MLP_0 mlp_2, DiffTensorView_0 featureGrid_6, DiffTensorView_0 imageOutput_2)
{
    uint3  dispatchIdx_2 = ((threadIdx)) + ((blockIdx)) * ((blockDim));
    uint _S1114 = dispatchIdx_2.x;
    uint _S1115 = DiffTensorView_size_0(imageOutput_2, 0U);
    uint _S1116 = dispatchIdx_2.y;
    uint _S1117 = DiffTensorView_size_0(imageOutput_2, 1U);
    if(_S1114 >= _S1115 || _S1116 >= _S1117)
    {
        return;
    }
    DiffPair_Feature_0 _S1118 = s_fwd_computeInterpolatedFeature_0(featureGrid_6, make_uint2 (_S1115, _S1117));
    DiffPair_Feature_0 _S1119 = { _S1118.primal_1, _S1118.differential_0 };
    DiffPair_Feature_0 _S1120 = s_fwd_MLP_eval_0(mlp_2, _S1119);
    DiffPair_float_0 _S1121 = { _S1120.primal_1.vals_1[int(0)], _S1120.differential_0.vals_0[int(0)] };
    DiffTensorView_storeOnce_forward_0(imageOutput_2, make_uint3 (_S1114, _S1116, 0U), _S1121);
    DiffPair_float_0 _S1122 = { _S1120.primal_1.vals_1[int(1)], _S1120.differential_0.vals_0[int(1)] };
    DiffTensorView_storeOnce_forward_0(imageOutput_2, make_uint3 (_S1114, _S1116, 1U), _S1122);
    DiffPair_float_0 _S1123 = { _S1120.primal_1.vals_1[int(2)], _S1120.differential_0.vals_0[int(2)] };
    DiffTensorView_storeOnce_forward_0(imageOutput_2, make_uint3 (_S1114, _S1116, 2U), _S1123);
    return;
}

extern "C" {
__global__ void __kernel__renderImage_fwd_diff(MLP_0 _S1124, DiffTensorView_0 _S1125, DiffTensorView_0 _S1126)
{
    s_fwd_renderImage_0(_S1124, _S1125, _S1126);
    return;
}

}
extern "C" {
__global__ void __kernel__renderImage(MLP_0 mlp_3, DiffTensorView_0 featureGrid_7, DiffTensorView_0 imageOutput_3)
{
    uint3  dispatchIdx_3 = ((threadIdx)) + ((blockIdx)) * ((blockDim));
    uint _S1127 = dispatchIdx_3.x;
    uint _S1128 = DiffTensorView_size_0(imageOutput_3, 0U);
    uint _S1129 = dispatchIdx_3.y;
    uint _S1130 = DiffTensorView_size_0(imageOutput_3, 1U);
    if(_S1127 >= _S1128 || _S1129 >= _S1130)
    {
        return;
    }
    Feature_0 feature_1 = computeInterpolatedFeature_0(featureGrid_7, make_uint2 (_S1128, _S1130));
    Feature_0 output_2 = MLP_eval_0(mlp_3, feature_1);
    DiffTensorView_storeOnce_0(imageOutput_3, make_uint3 (_S1127, _S1129, 0U), output_2.vals_1[int(0)]);
    DiffTensorView_storeOnce_0(imageOutput_3, make_uint3 (_S1127, _S1129, 1U), output_2.vals_1[int(1)]);
    DiffTensorView_storeOnce_0(imageOutput_3, make_uint3 (_S1127, _S1129, 2U), output_2.vals_1[int(2)]);
    return;
}

}
