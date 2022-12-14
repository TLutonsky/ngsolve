find_path (MUMPS_DIR include/zmumps_c.h HINTS ENV MUMPS_DIR DOC "Mumps Directory")
if(EXISTS ${MUMPS_DIR}/include/zmumps_c.h)
    set(MUMPS_FOUND YES)
    set(MUMPS_INCLUDES ${MUMPS_DIR})
    find_path (MUMPS_INCLUDE_DIR mumps_compat.h HINTS "${MUMPS_DIR}" PATH_SUFFIXES include NO_DEFAULT_PATH)
    list(APPEND MUMPS_INCLUDES ${MUMPS_INCLUDE_DIR})
    find_library(LIB_MUMPS_COMMON mumps_common PATHS ${MUMPS_DIR}/lib)
    find_library(LIB_MUMPS_D dmumps PATHS ${MUMPS_DIR}/lib)
    find_library(LIB_MUMPS_Z zmumps PATHS ${MUMPS_DIR}/lib)
    find_library(LIB_PORD pord PATHS ${MUMPS_DIR}/lib)
    find_library(LIB_PARMETIS parmetis HINTS ${PARMETIS_DIR}/lib REQUIRED)
    find_library(LIB_METIS metis HINTS ${PARMETIS_DIR}/lib REQUIRED)
    
    if (NOT USE_MKL)
        find_library(LIB_SCALAPACK scalapack HINTS ${SCALAPACK_DIR}/lib REQUIRED)
    endif()
    
    set(MUMPS_LIBRARIES ${LIB_MUMPS_D} ${LIB_MUMPS_Z} ${LIB_MUMPS_COMMON} ${LIB_PARMETIS} ${LIB_METIS} ${LIB_SCALAPACK})

    if (LIB_PORD)
       list(APPEND MUMPS_LIBRARIES ${LIB_PORD})
    endif()
    
endif(EXISTS ${MUMPS_DIR}/include/zmumps_c.h)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MUMPS DEFAULT_MSG MUMPS_LIBRARIES MUMPS_INCLUDES)
