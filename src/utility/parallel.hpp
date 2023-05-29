#pragma once

namespace par {

using thread_id_t = decltype(std::thread::hardware_concurrency());

extern const thread_id_t nThreads;

template<typename Callable, typename... Args>
requires std::is_invocable_v<Callable, thread_id_t, Args...>
void parallelProcess(Callable&& callable, Args&&... args)
{
	std::vector<std::thread> threads;
	threads.reserve(nThreads);

	for (thread_id_t id = 0; id < nThreads; id++)
		threads.emplace_back(std::thread{callable, id, args...});

	for (auto& thread : threads)
		thread.join();
}

} // namespace par
