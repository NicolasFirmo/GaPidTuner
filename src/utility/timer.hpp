#pragma once
#include <chrono>
#include <iostream>
#include <string_view>
#include <type_traits>
#include <iomanip>

template <class TimeUnityT = std::chrono::microseconds>
class CommonTimer {
public:
	using TimeUnityType = TimeUnityT;
	static const char *timeUnityString;

protected:
	CommonTimer(std::string_view scopeName) : tp_(clock::now()), scopeName_(scopeName) {
		// static_assert(std::chrono::_Is_duration_v<TimeUnityT>,
		// 			  "Template argument TimeUnityT must be a "
		// 			  "std::chrono::duration type!");
	}

	using clock = std::chrono::steady_clock;
	clock::time_point tp_;
	const std::string_view scopeName_;
};

template <>
const char *CommonTimer<std::chrono::nanoseconds>::timeUnityString = "ns";
template <>
const char *CommonTimer<std::chrono::microseconds>::timeUnityString = "us";
template <>
const char *CommonTimer<std::chrono::milliseconds>::timeUnityString = "ms";
template <>
const char *CommonTimer<std::chrono::seconds>::timeUnityString = "s";
template <>
const char *CommonTimer<std::chrono::minutes>::timeUnityString = "m";
template <>
const char *CommonTimer<std::chrono::hours>::timeUnityString = "h";

template <class TimeUnityT = std::chrono::microseconds>
class Timer : public CommonTimer<TimeUnityT> {
public:
	Timer(std::string_view scopeName) : CommonTimer<TimeUnityT>(scopeName) {}
	~Timer() {
		auto duration = std::chrono::duration_cast<TimeUnityT>(
			Timer::clock::now() - this->tp_).count();
		std::clog << this->scopeName_ << " took: " << std::setw(6) << duration << this->timeUnityString << '\n';
	}
};

template <class TimeUnityT = std::chrono::microseconds>
class AccTimer : public CommonTimer<TimeUnityT> {
public:
	AccTimer(std::string_view scopeName) : CommonTimer<TimeUnityT>(scopeName) {}
	~AccTimer() {
		auto duration = totalDuration_.count();
		std::clog << this->scopeName_ << " took: " << std::setw(6) << duration << this->timeUnityString
				  << " on total\n";
	}

	void startCount() { this->tp_ = AccTimer::clock::now(); }
	void accumulateCount() {
		totalDuration_ += std::chrono::duration_cast<TimeUnityT>(AccTimer::clock::now() - this->tp_);
	}

private:
	TimeUnityT totalDuration_{0};
};