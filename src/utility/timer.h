#pragma once

#include <chrono>
#include <iostream>
#include <string_view>
#include <type_traits>

class CommonTimer {
public:
	using clock = std::chrono::steady_clock;

protected:
	explicit CommonTimer(std::string_view scopeName);

	clock::time_point tp_;
	std::string_view  scopeName_;
};

CommonTimer::CommonTimer(std::string_view scopeName) : tp_(clock::now()), scopeName_(scopeName) {}

class Timer : public CommonTimer {
public:
	explicit Timer(std::string_view scopeName);
	~Timer();

	Timer(const Timer&)			   = default;
	Timer& operator=(const Timer&) = default;
	Timer(Timer&&)				   = default;
	Timer& operator=(Timer&&)	   = default;
};

Timer::Timer(std::string_view scopeName) : CommonTimer(scopeName) {}

Timer::~Timer()
{
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - tp_).count();
	std::clog << scopeName_ << " took: " << duration << "us\n";
}

class AccTimer : public CommonTimer {
public:
	explicit AccTimer(std::string_view scopeName);
	~AccTimer();

	void start();
	void accumulate();

	[[nodiscard]] constexpr auto getTotalDuration() const { return totalDuration_.count(); };

	AccTimer(const AccTimer&)			 = default;
	AccTimer& operator=(const AccTimer&) = default;
	AccTimer(AccTimer&&)				 = default;
	AccTimer& operator=(AccTimer&&)		 = default;

private:
	std::chrono::microseconds totalDuration_{0};
};

AccTimer::AccTimer(std::string_view scopeName) : CommonTimer(scopeName) {}

AccTimer::~AccTimer()
{
	auto duration = totalDuration_.count();
	std::clog << scopeName_ << " took: " << duration << "us in total\n";
}

void AccTimer::start()
{
	tp_ = clock::now();
}

void AccTimer::accumulate()
{
	totalDuration_ += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - tp_);
	tp_ = clock::now();
}
