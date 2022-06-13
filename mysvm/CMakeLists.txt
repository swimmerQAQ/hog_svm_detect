aux_source_directory(. svm_src)

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

add_library(libsvm STATIC ${svm_src})