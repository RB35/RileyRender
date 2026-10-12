# Riley Render
A simple program that takes a canvas of shapes and convert them into a raster image. SYCL and AdaptiveCpp are used for cross-platform parallel computing

## Compile commands
`g++ -O3 -march=native rr_sequential.cpp -o rr_seq`

`acpp -O3 -march=native rr_sycl.cpp -o rr_sycl`

`acpp -O3 -march=native rr_sycl_USM.cpp -o rr_sycl_USM`

`acpp --acpp-targets="omp" -O3 -march=native rr_sycl.cpp -o rr_sycl_cpu`

`acpp -O3 -march=native rr_sycl_aa.cpp -o rr_sycl_aa`
