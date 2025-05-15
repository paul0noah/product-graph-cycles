target_sources(${PROJECT_NAME} 
				PRIVATE
					dijkstra_solver.cpp
					dijkstra_solver.hpp
	)
if(WITH_CUDA)
	target_sources(${PROJECT_NAME} PRIVATE dijkstra_solver_cuda.cpp  dijkstra_bnb_subroutine.cu  dijkstra_bnb_subroutine.cuh)
endif()