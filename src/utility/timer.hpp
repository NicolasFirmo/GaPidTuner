#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string_view>

template<class TimeUnityT = std::chrono::microseconds>
class CommonTimer {
public:
	using TimeUnityType = TimeUnityT;
	using clock			= std::chrono::steady_clock;

	static const char* timeUnityString;

protected:
	explicit CommonTimer(std::string_view scopeName) : tp_(clock::now()), scopeName_(scopeName)
	{
		// static_assert(std::chrono::_Is_duration_v<TimeUnityT>,
		// 			  "Template argument TimeUnityT must be a "
		// 			  "std::chrono::duration type!");
	}

	clock::time_point tp_;
	std::string_view  scopeName_;
};

template<>
const char* CommonTimer<std::chrono::nanoseconds>::timeUnityString = "ns";
template<>
const char* CommonTimer<std::chrono::microseconds>::timeUnityString = "us";
template<>
const char* CommonTimer<std::chrono::milliseconds>::timeUnityString = "ms";
template<>
const char* CommonTimer<std::chrono::seconds>::timeUnityString = "s";
template<>
const char* CommonTimer<std::chrono::minutes>::timeUnityString = "m";
template<>
const char* CommonTimer<std::chrono::hours>::timeUnityString = "h";

template<class TimeUnityT = std::chrono::microseconds>
class Timer : public CommonTimer<TimeUnityT> {
public:
	explicit Timer(std::string_view scopeName) : CommonTimer<TimeUnityT>(scopeName) {}
	~Timer()
	{
		auto duration = std::chrono::duration_cast<TimeUnityT>(Timer::clock::now() - this->tp_).count();
		std::clog << this->scopeName_ << " took: " << std::setw(6) << duration << this->timeUnityString << '\n';
	}

	Timer(const Timer&)				   = default;
	Timer& operator=(const Timer&)	   = default;
	Timer(Timer&&) noexcept			   = default;
	Timer& operator=(Timer&&) noexcept = default;
};

template<class TimeUnityT = std::chrono::microseconds>
class AccTimer : public CommonTimer<TimeUnityT> {
public:
	explicit AccTimer(std::string_view scopeName) : CommonTimer<TimeUnityT>(scopeName) {}
	~AccTimer()
	{
		auto duration = totalDuration_.count();
		std::clog << this->scopeName_ << " took: " << std::setw(6) << duration << this->timeUnityString
				  << " on total\n";
	}

	void startCount() { this->tp_ = AccTimer::clock::now(); }

	void accumulateCount()
	{
		totalDuration_ += std::chrono::duration_cast<TimeUnityT>(AccTimer::clock::now() - this->tp_);
		this->tp_ = AccTimer::clock::now();
	}

	[[nodiscard]] constexpr auto getTotalDuration() const { return totalDuration_.count(); };

	AccTimer(const AccTimer&)				 = default;
	AccTimer& operator=(const AccTimer&)	 = default;
	AccTimer(AccTimer&&) noexcept			 = default;
	AccTimer& operator=(AccTimer&&) noexcept = default;

private:
	TimeUnityT totalDuration_{0};
};
