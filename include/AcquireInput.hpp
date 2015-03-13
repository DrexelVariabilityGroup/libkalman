#ifndef ACQUIREINPUT_HPP
#define ACQUIREINPUT_HPP

#include <iostream>
#include <limits>
#include <string>

using namespace std;

template<typename InType> void AcquireInput(ostream& Os, istream& Is, const string& Prompt, const string& FailString, InType& Result) {
	do {
		Os << Prompt.c_str();
		if (Is.fail()) {
			Is.clear();
			Is.ignore(numeric_limits<streamsize>::max(), '\n');
			}
		Is >> Result;
		if (Is.fail()) {
			Os << FailString.c_str();
			}
		} while(Is.fail());
	}

template<typename InType> InType AcquireInput(ostream& Os, istream& Is, const string& Prompt, const string& FailString) {
	InType temp;
	AcquireInput(Os,Is,Prompt,FailString,temp);
	return temp;
	}

#endif