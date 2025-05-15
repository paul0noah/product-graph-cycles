add_subdirectory(constraints)
add_subdirectory(product_spaces)
add_subdirectory(energy)
add_subdirectory(cost_time_ratio_solver)
add_subdirectory(dijkstra_solver)

target_sources(${PROJECT_NAME} 
				PRIVATE
					product_graph_generators.cpp
					product_graph_generators.hpp
	)
