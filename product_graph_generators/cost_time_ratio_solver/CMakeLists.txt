target_sources(${PROJECT_NAME} 
				PRIVATE
					cost_time_ratio_solver.cpp
					cost_time_ratio_solver.hpp
	)
if(WITH_CUDA)
	target_sources(${PROJECT_NAME} PRIVATE cost_time_ratio_solver_cuda.cpp  ctr_bnb_subroutine.cu  ctr_bnb_subroutine.cuh)
endif()