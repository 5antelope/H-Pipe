#include <iostream>

#include "include/blob.h"
#include "include/common.h"
#include "include/context.h"
#include "include/tensor.h"
#include "include/caffe2.pb.h"

namespace hpiper {
namespace {

class Foo {};
class Bar {};

int main(int argc, char* argv[]) { {
  Blob blob;

  int* int_unused UNUSED_VARIABLE = blob.GetMutable<int>();
  std::cout << blob.IsType<int>() << std::endl; // true
  std::cout << blob.IsType<Foo>() << std::endl; // false

  Foo* foo_unused UNUSED_VARIABLE = blob.GetMutable<Foo>();
  std::cout << blob.IsType<Foo>() << std::endl; // true
  std::cout << blob.IsType<int>() << std::endl; // false
}



template <typename T> class TensorCPUTest : public ::testing::Test {};
template <typename T> class TensorCPUDeathTest : public ::testing::Test {};
typedef ::testing::Types<char, int, float> TensorTypes;
TYPED_TEST_CASE(TensorCPUTest, TensorTypes);
TYPED_TEST_CASE(TensorCPUDeathTest, TensorTypes);

TYPED_TEST(TensorCPUTest, TensorInitializedEmpty) {
  TensorCPU tensor;
  EXPECT_EQ(tensor.ndim(), 0);
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  tensor.Reshape(dims);
  EXPECT_EQ(tensor.ndim(), 3);
  EXPECT_EQ(tensor.dim(0), 2);
  EXPECT_EQ(tensor.dim(1), 3);
  EXPECT_EQ(tensor.dim(2), 5);
  EXPECT_EQ(tensor.size(), 2 * 3 * 5);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
}

TYPED_TEST(TensorCPUTest, TensorInitializedNonEmpty) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  TensorCPU tensor(dims);
  EXPECT_EQ(tensor.ndim(), 3);
  EXPECT_EQ(tensor.dim(0), 2);
  EXPECT_EQ(tensor.dim(1), 3);
  EXPECT_EQ(tensor.dim(2), 5);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
  dims[0] = 7;
  dims[1] = 11;
  dims[2] = 13;
  dims.push_back(17);
  tensor.Reshape(dims);
  EXPECT_EQ(tensor.ndim(), 4);
  EXPECT_EQ(tensor.dim(0), 7);
  EXPECT_EQ(tensor.dim(1), 11);
  EXPECT_EQ(tensor.dim(2), 13);
  EXPECT_EQ(tensor.dim(3), 17);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
}

TYPED_TEST(TensorCPUTest, TensorShareData) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  TensorCPU tensor(dims);
  TensorCPU other_tensor(dims);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  other_tensor.ShareData(tensor);
  EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
  EXPECT_TRUE(other_tensor.data<TypeParam>() != nullptr);
  EXPECT_EQ(tensor.data<TypeParam>(), other_tensor.data<TypeParam>());
  // Set one value, check the other
  for (int i = 0; i < tensor.size(); ++i) {
    tensor.mutable_data<TypeParam>()[i] = i;
    EXPECT_EQ(other_tensor.data<TypeParam>()[i], i);
  }
}

TYPED_TEST(TensorCPUTest, TensorShareDataCanUseDifferentShapes) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  vector<int> alternate_dims(1);
  alternate_dims[0] = 2 * 3 * 5;
  TensorCPU tensor(dims);
  TensorCPU other_tensor(alternate_dims);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  other_tensor.ShareData(tensor);
  EXPECT_EQ(other_tensor.ndim(), 1);
  EXPECT_EQ(other_tensor.dim(0), alternate_dims[0]);
  EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
  EXPECT_TRUE(other_tensor.data<TypeParam>() != nullptr);
  EXPECT_EQ(tensor.data<TypeParam>(), other_tensor.data<TypeParam>());
  // Set one value, check the other
  for (int i = 0; i < tensor.size(); ++i) {
    tensor.mutable_data<TypeParam>()[i] = i;
    EXPECT_EQ(other_tensor.data<TypeParam>()[i], i);
  }
}


TYPED_TEST(TensorCPUTest, NoLongerSharesAfterReshape) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  TensorCPU tensor(dims);
  TensorCPU other_tensor(dims);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  other_tensor.ShareData(tensor);
  EXPECT_EQ(tensor.data<TypeParam>(), other_tensor.data<TypeParam>());
  auto* old_pointer = other_tensor.data<TypeParam>();

  dims[0] = 7;
  tensor.Reshape(dims);
  EXPECT_EQ(old_pointer, other_tensor.data<TypeParam>());
  EXPECT_NE(old_pointer, tensor.mutable_data<TypeParam>());
}


TYPED_TEST(TensorCPUDeathTest, CannotAccessDataWhenEmpty) {
  TensorCPU tensor;
  EXPECT_EQ(tensor.ndim(), 0);
  ASSERT_DEATH(tensor.data<TypeParam>(), "");
}

}  // namespace
}  // namespace hpiper

