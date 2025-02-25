#include <apex_api.hpp>

inline auto now = std::chrono::high_resolution_clock::now;

inline double diff(const std::chrono::high_resolution_clock::time_point &start_time)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now() - start_time).count();
}
