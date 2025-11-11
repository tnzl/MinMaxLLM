#pragma once

/**
 * @file base_op.h
 * @brief Base class for all operators with unified backend abstraction
 */

/**
 * @enum OpBackend
 * @brief Unified backend enumeration for all operators
 * 
 * This enum defines the available backends that can be used across
 * all operator types. Each operator maps these backends to their
 * specific implementation types.
 */
enum class OpBackend
{
    NAIVE,  ///< Reference implementation without SIMD optimizations
    AVX2,   ///< AVX2 SIMD optimized implementation
    AVX512  ///< AVX-512 SIMD optimized implementation (future)
};

/**
 * @class BaseOp
 * @brief Base class for all operators
 * 
 * Provides common infrastructure for all operators:
 * - Unified backend abstraction (OpBackend)
 * - Virtual prepare() for weight prefetching and buffer allocation
 * - Pure virtual run() with operator-specific signatures
 * 
 * Derived classes should:
 * - Override run() with their specific signature
 * - Override prepare() if they need initialization
 * - Define their own ImplType enum for implementation selection
 * - Implement resolve_impl() to map OpBackend to their ImplType
 */
class BaseOp
{
protected:
    OpBackend backend_; ///< Selected backend for this operator instance

public:
    /**
     * @brief Construct a BaseOp with specified backend
     * @param backend Backend to use (default: AVX2)
     */
    explicit BaseOp(OpBackend backend = OpBackend::AVX2) : backend_(backend) {}

    /**
     * @brief Virtual destructor for proper cleanup of derived classes
     */
    virtual ~BaseOp() = default;

    /**
     * @brief Prepare operator for execution
     * 
     * Override this method to:
     * - Prefetch weights into cache
     * - Allocate intermediate buffers
     * - Perform any one-time setup
     * 
     * Default implementation does nothing.
     */
    virtual void prepare() {}

    /**
     * @brief Execute the operator
     * 
     * Pure virtual method - each operator defines its own signature.
     * Examples:
     * - LinearOp: run(Tensor &input, Tensor &output)
     * - SelfAttention: run(Tensor &input, size_t token_idx, Tensor &output)
     */
    // Note: Cannot declare pure virtual with no signature in C++.
    // Each derived class will define its own run() overloads.
};

