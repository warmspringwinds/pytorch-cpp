/*
Example shows how to define an architecture, visualize it later on
using std tools, and get a forward pass from that model.
*/

#include "ATen/ATen.h"
#include "ATen/Type.h"
#include <sstream>
#include "H5Cpp.h"

using namespace at;

int main()
{

  
  hsize_t dims[2];
  float *signal;

  // This file was saved by converting pytorch's tensor to numpy
  // and writing the numpy array into the hdf5 file
  H5::H5File file = H5::H5File("float_ten_by_ten.h5", H5F_ACC_RDONLY);

  H5::DataSet signal_dset = file.openDataSet("dataset_1");

  
  H5::DataSpace signal_dspace = signal_dset.getSpace();

  signal_dspace.getSimpleExtentDims(dims, NULL);

  std::cout << dims[0] << " " << dims[1] << std::endl;

  signal = new float[(int)(dims[0])*(int)(dims[1])];

  H5::DataSpace signal_mspace(2, dims);


  signal_dset.read(signal, H5::PredType::NATIVE_FLOAT, signal_mspace, 
       signal_dspace);


  auto f = CPU(kFloat).tensorFromBlob(signal, {10, 10});

  std::cout << f << std::endl;

  file.close();


   // Overall output:

  //    10 10
  //  1  1  1  1  1  1  1  1  1  1
  //  1  1  1  1  1  1  1  1  1  1
  //  1  1  1  1  1  1  1  1  1  1
  //  1  1  1  1  1  1  1  1  1  1
  //  1  1  1  1  1  1  1  1  1  1
  //  1  1  1  1  1  1  1  1  1  1
  //  1  1  1  1  1  1  1  1  1  1
  //  1  1  1  1  1  1  1  1  1  1
  //  1  1  1  1  1  1  1  1  1  1
  //  1  1  1  1  1  1  1  1  1  1
  // [ CPUFloatTensor{10,10} ]


   return 0;
}