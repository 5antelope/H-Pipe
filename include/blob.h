#include <cstddef>
#include <sstream>
#include <typeinfo>
#include <type_traits>
#include <vector>

namespace hpiper {
/**
 * @brief Blob is a general container that hosts a typed pointer.
 *
 * A Blob hosts a pointer as well as its type, and takes charge of deleting it
 * properly when the blob is deallocated or re-allocated with a new type. A blob
 * could contain ANYTHING, although the most common case is to contain a Tensor.
 */
class Blob {
private:
  template <class T>
  static void Destroy(void* pointer) {
    delete static_cast<T*>(pointer);
  }
  typedef void (*DestroyCall)(void *);
  TypeMeta meta_;
  void* pointer_;
  DestroyCall destroy_;

  DISABLE_COPY_AND_ASSIGN(Blob);

public:
  /**
   * Initializes an empty Blob.
   */
  Blob() : meta_(), pointer_(nullptr) {}
  ~Blob() { Reset(); }

  inline const char* TypeName() const { return meta_.name(); }

  /**
   * Gets the const reference of the stored object. The code checks if the
   * stored object is of the desired type.
   */
  template <class T>
  const T& Get() const {
    return *static_cast<const T*>(pointer_);
  }

  /**
   * Gets a mutable pointer to the stored object. If the current object is not
   * of the right type, a new object is created and the old object is freed.
   * Note that type T should have a default constructor. Otherwise, create the
   * object yourself first, and and use Reset().
   */
  template <class T>
  T* GetMutable() {
    if (IsType<T>()) {
      return static_cast<T*>(pointer_);
    } else {
      return Reset<T>(new T());
    }
  }

  /**
   * Sets the underlying object to the allocated one. The Blob then takes over
   * the ownership of the passed in pointer. If there is already an object in
   * the Blob, the old object is freed.
   */
  template <class T>
  T* Reset(T* allocated) {
    if (pointer_) { destroy_(pointer_); }
    CAFFE_VLOG(1) << "Create new mutable object " << TypeMeta::Name<T>();
    meta_ = TypeMeta::Make<T>();
    pointer_ = static_cast<void*>(allocated);
    destroy_ = &Destroy<T>;
    return allocated;
  }

  /**
   * Resets the Blob to an empty one.
   */
  inline void Reset() {
    if (pointer_) {
      destroy_(pointer_);
      pointer_ = nullptr;
    }
  }

  /**
   * Serializes the current blob, if possible. Note that this serialization uses
   * the registration mechanism and one has to implement specific serialization
   * approaches for specific classes.
   */
  string Serialize(const string& name) const;

};
} // namespace hpiper