include(${CMAKE_SOURCE_DIR}/cmake/AddCliffordUnitTest.cmake)

add_cliff_ut(
	NAME LinearLayout
	SRCS LinearLayoutTest.cpp
	LIBS CliffTools
)
