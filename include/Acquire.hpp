#ifndef ACQUIRE_HPP
#define ACQUIRE_HPP

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <sstream>
#include <boost/system/error_code.hpp>
#include <boost/system/system_error.hpp>
#include <boost/system/linux_error.hpp>
#include <boost/filesystem.hpp>
#include <boost/io/detail/quoted_manip.hpp>

using namespace std;
using namespace boost::filesystem;

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

void AcquireDirectory(ostream& Os, istream& Is, const string& Prompt, const string& FailString, string& Result); /*{
	bool validPath = false, satisfied = false;
	string userInput, homeDir = getenv("HOME"), workingDir = getenv("PWD"), rootDir = "/";
	path inputPath, rootPath = path(rootDir);
	do {
		validPath = false, satisfied = false;
		do {
			cout << Prompt;
			getline(cin,userInput);
			//cout << "getLine: " << userInput << endl;
			for (int i = 0; i < userInput.length(); ++i) {
				if (!userInput.compare(i,1,"~")) {
					//cout << "Found ~" << endl;
					userInput.replace(i,1,homeDir);
					//cout << "Parsed to " << userInput << endl;
					}
				}
			if (userInput.compare(0,1,"/")) {
				//cout << "Relative path!" << endl;
				try {
					inputPath = canonical(path(userInput), current_path());
					//cout << "Valid path! " << inputPath << endl;
					validPath = true;
					} catch (filesystem_error) {
					cout << "Invalid path!" << endl;
					}
				} else {
				//cout << "Absolute path!" << endl;
				try {
					inputPath = canonical(path(userInput), rootPath);
					//cout << "Valid path! " << inputPath << endl;
					validPath = true;
					} catch (filesystem_error) {
					cout << "Invalid path!" << endl;
					}
				}
			} while (!validPath);
		if (is_regular_file(inputPath)) {
			cout << "Error! " << inputPath << "is a file!" << endl;
			continue;
			} else {
			Result = inputPath.string();
			satisfied = true;
			}
		} while (!satisfied);
	}*/

#endif
