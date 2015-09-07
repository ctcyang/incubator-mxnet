/*!
 *  Copyright (c) 2015 by Contributors
 * \file narray.h
 * \brief narray interface that dynamically schedules operations
 */
#ifndef MXNET_NARRAY_H_
#define MXNET_NARRAY_H_

#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/io.h>
#include <dmlc/type_traits.h>
#include <dmlc/registry.h>
#include <memory>
#include "./base.h"
#include "./context.h"
#include "./storage.h"
#include "./context.h"
#include "./engine.h"
// check c++11
#if DMLC_USE_CXX11 == 0
#error "cxx11 was required for narray module"
#endif

namespace mxnet {
/*!
 * \brief ndarray interface
 */
class NArray {
 public:
  /*! \brief default cosntructor */
  NArray() {}
  /*!
   * \brief constructing a new dynamic NArray
   * \param shape the shape of array
   * \param ctx context of NArray
   * \param delay_alloc whether delay the allocation
   */
  NArray(const TShape &shape, Context ctx,
         bool delay_alloc = false)
      : ptr_(std::make_shared<Chunk>(shape.Size(), ctx, delay_alloc)), shape_(shape), offset_(0) {
  }
  /*!
   * \brief constructing a static NArray that shares data with TBlob
   *  Use with caution: allocate ONLY ONE NArray for each TBlob,
   *  make sure the memory region is available through out the life of NArray
   * \param data the memory content of static data
   * \param dev_id the device id this tensor sits at
   */
  NArray(const TBlob &data, int dev_id)
      : ptr_(std::make_shared<Chunk>(data, dev_id)), shape_(data.shape_), offset_(0) {
  }
  /*!
   * \return the shape of current NArray
   */
  inline const TShape &shape() const {
    return shape_;
  }
  /*!
   * \return the data TBlob
   */
  inline TBlob data() const {
    return TBlob(static_cast<real_t*>(ptr_->shandle.dptr) + offset_, \
                                      shape_, ptr_->shandle.ctx.dev_mask);
  }
  /*!
   * \return the context of NArray, this function is only valid when the NArray is not empty
   */
  inline Context ctx() const {
    return ptr_->shandle.ctx;
  }
  /*! \return whether this narray is not initialized */
  inline bool is_none() const {
    return ptr_.get() == nullptr;
  }
  /*! \brief wait until the result of the NArray is computed */
  inline void Wait() const {
    if (is_none()) return;
    Engine::Get()->WaitForVar(ptr_->var);
  }
  /*! \return the associated variable of the narray.*/
  inline Engine::VarHandle var() const {
    return ptr_->var;
  }
  /*!
   * \brief save the content into binary stream
   * \param strm the output stream
   */
  void Save(dmlc::Stream *strm) const;
  /*!
   * \brief load the content from binary stream
   * \param strm the output stream
   * \return whether the load is successful
   */
  bool Load(dmlc::Stream *strm);
  /*!
   * \brief set all the elements in narray to be scalar
   * \param scalar the scalar to set
   * \return reference of self
   */
  NArray &operator=(real_t scalar);
  /*!
   * \brief elementwise add to current space
   *  this mutate the current NArray
   * \param src the data to add
   * \return reference of self
   */
  NArray &operator+=(const NArray &src);
  /*!
   * \brief elementwise add to current space
   *  this mutate the current NArray
   * \param src the data to add
   * \return reference of self
   */
  NArray &operator+=(const real_t &src);
  /*!
   * \brief elementwise subtract from current narray
   * this mutate the current NArray
   * \param src the data to substract
   * \return reference of self
   */
  NArray &operator-=(const NArray &src);
  /*!
   * \brief elementwise subtract from current narray
   * this mutate the current NArray
   * \param src the data to substract
   * \return reference of self
   */
  NArray &operator-=(const real_t &src);
  /*!
   * \brief elementwise multiplication to current narray
   *  this mutate the current NArray
   * \param src the data to substract
   * \return reference of self
   */
  NArray &operator*=(const NArray &src);
  /*!
   * \brief elementwise multiplication to current narray
   *  this mutate the current NArray
   * \param src the data to substract
   * \return reference of self
   */
  NArray &operator*=(const real_t &src);
  /*!
   * \brief elementwise division from current narray
   *  this mutate the current NArray
   * \param src the data to substract
   * \return reference of self
   */
  NArray &operator/=(const NArray &src);
  /*!
   * \brief elementwise division from current narray
   *  this mutate the current NArray
   * \param src the data to substract
   * \return reference of self
   */
  NArray &operator/=(const real_t &src);
  /*!
   * \brief return transpose of current NArray
   * \return a new transposed NArray
   */
  NArray T() const;
  /*!
   * \brief return a new copy this NArray
   * \param ctx the new context of this NArray
   * \return the new copy
   */
  NArray Copy(Context ctx) const;
  /*!
   * \brief Slice a NArray
   * \param begin begin index in first dim
   * \param end end index in first dim
   * \return sliced NArray
   */
  inline NArray Slice(index_t begin, index_t end) const {
    NArray ret = *this;
    CHECK(!is_none()) << "NArray is not initialized";
    CHECK_GE(shape_[0], end) << "Slice end index out of range";
    size_t length = 1;
    if (shape_.ndim() == 1) {
      ret.offset_ = begin;
    } else {
      for (index_t i = 1; i < shape_.ndim(); ++i) {
        length *= shape_[i];
      }
      ret.offset_ = begin * length;
    }
    ret.shape_[0] = end - begin;
    return ret;
  }
  /*!
   * \brief Get an reshaped NArray
   * \param shape new shape
   * \return NArray in new shape
   */
  inline NArray Reshape(const TShape &shape) const {
    CHECK_GE(shape_.Size(), shape.Size())
        << "NArray.Reshape: target shape size is different from current shape";
    NArray ret = *this;
    ret.shape_ = shape;
    return ret;
  }

 private:
  /*! \brief the real data chunk that backs NArray */
  struct Chunk {
    /*! \brief storage handlefrom storage engine */
    Storage::Handle shandle;
    /*! \brief variable from engine */
    Engine::VarHandle var;
    /*!
     * \brief if this is true, this means the data do not come
     * from Storage, and do not need to be freed
     */
    bool static_data;
    /*! \brief whether allocation is delayed */
    bool delay_alloc;
    /*! \brief default cosntructor */
    Chunk() : static_data(true), delay_alloc(false) {
      var  = Engine::Get()->NewVariable();
    }
    /*! \brief construct from static data */
    Chunk(const TBlob &data, int dev_id)
        : static_data(true),
          delay_alloc(false) {
      var = Engine::Get()->NewVariable();
      shandle.ctx = Context(data.dev_mask_, dev_id);
      shandle.dptr = data.dptr_;
      shandle.size = data.shape_.Size() * sizeof(real_t);
    }
    /*! \brief construct a new chunk */
    Chunk(uint64_t size, Context ctx, bool delay_alloc_)
        : static_data(false), delay_alloc(true) {
      var = Engine::Get()->NewVariable();
      shandle.size = size * sizeof(real_t);
      shandle.ctx = ctx;
      if (!delay_alloc_) this->CheckAndAlloc();
    }
    /*! \brief check if delay alloc is on, do alloc if not yet done */
    inline void CheckAndAlloc(void) {
      if (delay_alloc) {
        shandle = Storage::Get()->Alloc(shandle.size, shandle.ctx);
        delay_alloc = false;
      }
    }
    /*! \brief destructor */
    ~Chunk() {
      if (static_data) {
        Engine::Get()->DeleteVariable([](RunContext s) {}, shandle.ctx, var);
      } else {
        CHECK(!delay_alloc) << "deleted before allocation";
        Storage::Handle h = this->shandle;
        Engine::Get()->DeleteVariable([h](RunContext s) {
            Storage::Get()->Free(h);
          }, shandle.ctx, var);
      }
    }
  };
  /*! \brief internal data of NArray */
  std::shared_ptr<Chunk> ptr_;
  /*! \brief shape of current NArray */
  TShape shape_;
  /*! \brief offset in chunk */
  size_t offset_;

  // add friend to helper functions
  friend void CopyFromTo(const NArray &from, NArray *to);
  template<typename OP>
  friend void BinaryOp(const NArray &lhs, const NArray &rhs, NArray *out);
  template<typename OP>
  friend void UnaryOp(const NArray &lhs, const NArray &rhs, NArray *out);
  template<typename OP, bool reverse>
  friend void ScalarOp(const NArray &lhs, const real_t &rhs, NArray *out);
  friend void SetValueOp(const real_t &rhs, NArray *out);
};

/*!
 * \brief issue an copy operation from one NArray to another
 *  the two narray can sit on different devices
 *  this operation will be scheduled by the engine
 *
 *  NOTE: this function name explicitly marks the order of from and to
 *     due to different possible convention carried by copy function
 * \param from the narray we want to copy data from
 * \param to the target narray
 */
void CopyFromTo(const NArray &from, NArray *to);

/*!
 * \brief elementwise add
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result narray
 */
NArray operator+(const NArray &lhs, const NArray &rhs);
/*!
 * \brief elementwise add
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result narray
 */
NArray operator+(const NArray &lhs, const real_t &rhs);
/*!
 * \brief elementwise substraction
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result narray
 */
NArray operator-(const NArray &lhs, const NArray &rhs);
/*!
 * \brief elementwise substraction
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result narray
 */
NArray operator-(const NArray &lhs, const real_t &rhs);
/*!
 * \brief elementwise multiplication
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result narray
 */
NArray operator*(const NArray &lhs, const NArray &rhs);\
/*!
 * \brief elementwise multiplication
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result narray
 */
NArray operator*(const NArray &lhs, const real_t &rhs);
/*!
 * \brief elementwise division
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result narray
 */
NArray operator/(const NArray &lhs, const NArray &rhs);
/*!
 * \brief elementwise division
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result narray
 */
NArray operator/(const NArray &lhs, const real_t &rhs);

//--------------------------------------------------------------
// The following part are API Registration of NArray functions.
//--------------------------------------------------------------
/*! \brief definition of NArray function */
typedef std::function<void (NArray **used_vars,
                            real_t *scalars,
                            NArray **mutate_vars)> NArrayAPIFunction;
/*! \brief mask information on how functions can be exposed */
enum NArrayFunctionTypeMask {
  /*! \brief all the use_vars should go before scalar */
  kNArrayArgBeforeScalar = 1,
  /*! \brief all the scalar should go before use_vars */
  kScalarArgBeforeNArray = 1 << 1,
  /*!
   * \brief whether this function allows the handles in the target to
   *  be empty NArray that are not yet initialized, and will initialize
   *  them when the function is invoked.
   *
   *  most function should support this, except copy between different
   *  devices, which requires the NArray to be pre-initialized with context
   */
  kAcceptEmptyMutateTarget = 1 << 2
};
/*! \brief Registry entry for NArrayFunction */
struct NArrayFunctionReg
    : public dmlc::FunctionRegEntryBase<NArrayFunctionReg,
                                        NArrayAPIFunction> {
  /*! \brief number of variable used by this function */
  unsigned num_use_vars;
  /*! \brief number of variable mutated by this function */
  unsigned num_mutate_vars;
  /*! \brief number of scalars used by this function */
  unsigned num_scalars;
  /*! \brief information on how function should be called from API */
  int type_mask;
  /*!
   * \brief constructor
   */
  NArrayFunctionReg()
      : num_use_vars(0),
        num_mutate_vars(0),
        num_scalars(0),
        type_mask(0) {}
  /*!
   * \brief set the function body to a NArray setvalue function
   *  this will also auto set the parameters correctly
   * \param fsetvalue function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NArrayFunctionReg &set_function(void fsetvalue(const real_t &rhs,
                                                        NArray *out)) {
    body = [fsetvalue] (NArray **used_vars,
                       real_t *s, NArray **mutate_vars) {
      fsetvalue(s[0], mutate_vars[0]);
    };
    num_mutate_vars = 1; num_scalars = 1;
    // type_mask = kNArrayArgBeforeScalar;
    this->add_argument("rhs", "real_t", "Right operand to the function.");
    return *this;
  }
  /*!
   * \brief set the function body to a binary NArray function
   *  this will also auto set the parameters correctly
   * \param fbinary function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NArrayFunctionReg &set_function(void fbinary(const NArray &lhs,
                                                      const NArray &rhs,
                                                      NArray *out)) {
    body = [fbinary] (NArray **used_vars,
                      real_t *s, NArray **mutate_vars) {
      fbinary(*used_vars[0], *used_vars[1], mutate_vars[0]);
    };
    num_use_vars = 2; num_mutate_vars = 1;
    type_mask = kNArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
    this->add_argument("lhs", "NArray", "Left operand to the function.");
    this->add_argument("rhs", "NArray", "Right operand to the function.");
    return *this;
  }
  /*!
   * \brief set the function body to a binary NArray function
   *  this will also auto set the parameters correctly
   * \param fscalar function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NArrayFunctionReg &set_function(void fscalar(const NArray &lhs,
                                                      const real_t &rhs,
                                                      NArray *out)) {
    body = [fscalar] (NArray **used_vars,
                       real_t *s, NArray **mutate_vars) {
      fscalar(*used_vars[0], s[0], mutate_vars[0]);
    };
    num_use_vars = 1; num_mutate_vars = 1; num_scalars = 1;
    type_mask = kNArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
    this->add_argument("lhs", "NArray", "Left operand to the function.");
    this->add_argument("rhs", "real_t", "Right operand to the function.");
    return *this;
  }
  /*!
   * \brief set the function body to a unary NArray function
   *  this will also auto set the parameters correctly
   * \param funary function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NArrayFunctionReg &set_function(void funary(const NArray &src,
                                                     NArray *out)) {
    body = [funary] (NArray **used_vars,
                     real_t *s, NArray **mutate_vars) {
      funary(*used_vars[0], mutate_vars[0]);
    };
    num_use_vars = 1; num_mutate_vars = 1;
    type_mask = kNArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
    this->add_argument("src", "NArray", "Source input to the function.");
    return *this;
  }
  /*!
   * \brief set the number of mutate variables
   * \param n number of mutate variablesx
   * \return ref to the registered entry, used to set properties
   */
  inline NArrayFunctionReg &set_num_use_vars(unsigned n) {
    num_use_vars = n; return *this;
  }
  /*!
   * \brief set the number of mutate variables
   * \param n number of mutate variablesx
   * \return ref to the registered entry, used to set properties
   */
  inline NArrayFunctionReg &set_num_mutate_vars(unsigned n) {
    num_mutate_vars = n; return *this;
  }
  /*!
   * \brief set the number of scalar arguments
   * \param n number of scalar arguments
   * \return ref to the registered entry, used to set properties
   */
  inline NArrayFunctionReg &set_num_scalars(unsigned n) {
    num_scalars = n; return *this;
  }
  /*!
   * \brief set type mask
   * \param tmask typemask
   * \return ref to the registered entry, used to set properties
   */
  inline NArrayFunctionReg &set_type_mask(int tmask) {
    type_mask = tmask; return *this;
  }
};  // NArrayFunctionReg

/*!
 * \brief Macro to register NArray function
 *
 * Example: the following code is example to register a plus
 * \code
 *
 * REGISTER_NARRAY_FUN(Plus)
 * .set_function(Plus);
 *
 * \endcode
 */
#define MXNET_REGISTER_NARRAY_FUN(name)                                 \
  DMLC_REGISTRY_REGISTER(::mxnet::NArrayFunctionReg, NArrayFunctionReg, name)

}  // namespace mxnet

namespace dmlc {
/*!\brief traits */
DMLC_DECLARE_TRAITS(has_saveload, mxnet::NArray, true);
}  // namespace dmlc
#endif  // MXNET_NARRAY_H_
