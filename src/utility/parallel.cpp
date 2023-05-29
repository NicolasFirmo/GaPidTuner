#include "parallel.hpp"

namespace par {

const thread_id_t nThreads = std::thread::hardware_concurrency();

} // namespace par
