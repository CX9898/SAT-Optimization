cd models
nvcc -Xcompiler -fPIC -shared -lcusparse -lcurand -I./sddmm/include \
  --relocatable-device-code=true \
  -gencode=arch=compute_80,code=sm_89 \
  -o block_attn.so \
  block_attn.cu \
  sddmm/src/sddmm.cu \
  sddmm/src/sddmmKernel.cu \
  sddmm/src/cudaUtil.cu \
  sddmm/src/parallelAlgorithm.cu \
  sddmm/src/rowReordering.cu \
  sddmm/src/colReordering.cu \
  sddmm/src/util.cpp \
  sddmm/src/host.cpp \
  sddmm/src/Matrix.cpp \
  sddmm/src/ReBELL.cpp
nvcc -Xcompiler -fPIC -shared -lcusparse -o block_attn_mask.so block_attn_mask.cu
nvcc -Xcompiler -fPIC -shared -lcublas -o attn.so attn.cu
nvcc -Xcompiler -fPIC -shared -lcublas -o attn_mask.so attn_mask.cu
cd ..