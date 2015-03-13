#include <mathimf.h>
#include <mkl.h>
#include <mkl_types.h>
#include <omp.h>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include "AcquireInput.hpp"
#include "Kalman.hpp"
#include "MCMC.hpp"

#define TIME_LNLIKE
#define TIME_MCMC

using namespace std;

int main() {
	cout.clear();
	cout << endl;
	cout << "Program: testMethod" << endl;
	cout << "Purpose: Program to test Kalman filtering method (end-to-end test)." << endl;
	cout << "Author: Vishal Kasliwal" << endl;
	cout << "Institution: Drexel university, Department of Physics" << endl;
	cout << "Email: vpk24@drexel.edu" << endl;
	cout << endl;

	int nthreads = 0;
	AcquireInput(cout,cin,"Number of threads to use: ","Invalid value.\n",nthreads);
	//omp_set_dynamic(0);
	omp_set_num_threads(nthreads);
	int threadNum = omp_get_thread_num();

	cout << "Create a test light curve with known parameters - make an ARMA light curve with p AR and q MA co-efficients." << endl;
	int pMaster = 0, qMaster = 0;
	AcquireInput(cout,cin,"Number of AR coefficients p: ","Invalid value.\n",pMaster);
	qMaster = pMaster;
	while (qMaster >= pMaster) {
		cout << "The number of MA coefficients (q) must be less than the number of AR coefficients (p) if the system is to \n correspond to a C-ARMA process." << endl;
		cout << "Please select q < " << pMaster << endl;
		AcquireInput(cout,cin,"Number of MA coefficients q: ","Invalid value.\n",qMaster);
		}
	cout << "Creating DLM with " << pMaster << " AR components and " << qMaster << " MA components." << endl;
	DLM SystemMaster = DLM();
	SystemMaster.allocDLM(pMaster, qMaster);
	cout << "Allocated " << SystemMaster.allocated << " bytes for the DLM!" << endl;

	double* ThetaMaster = static_cast<double*>(_mm_malloc((pMaster+qMaster+1)*sizeof(double),64));
	#pragma omp parallel for simd default(none) shared(pMaster,qMaster,ThetaMaster)
	for (int i = 0; i < pMaster+qMaster+1; i++) {
		ThetaMaster[i] = 0.0;
		}
	cout << "Set the values of the DLM parameters." << endl;

	int goodYN = 0;
	while (goodYN == 0) {
		while (ThetaMaster[0] <= 0.0) {
			cout << "Set the standard deviation of the disturbances (sigma_dist) such that sigma_dist > 0.0" << endl;
			AcquireInput(cout,cin,"Set the value of sigma_dist: ","Invalid value.\n",ThetaMaster[0]);
			}

		string inStr;
		for (int i = 1; i < 1+pMaster; i++) {
			cout << "Set the value of phi_" << i;
			AcquireInput(cout,cin,": ","Invalid value.\n",ThetaMaster[i]);
			}

		for (int i = 1+pMaster; i < 1+pMaster+qMaster; i++) {
			cout << "Set the value of theta_" << i-pMaster;
			AcquireInput(cout,cin,": ","Invalid value.\n",ThetaMaster[i]);
			}

		//printf("testMethod - threadNum: %d; Address of SystemMaster: %p\n",threadNum,&SystemMaster);
		SystemMaster.setDLM(ThetaMaster);
		cout << endl;
		cout << "Checking to see if the system is stable, invertible, not-redundant, and reasonable..." << endl;
		goodYN = SystemMaster.checkARMAParams(ThetaMaster);
		cout << "System parameters are ";
		if (goodYN == 0) {
			cout << "bad!" << endl;
			cout << "Redo!" << endl;
			cout << endl;
			} else {
			cout << "good!" << endl;
			cout << endl;
			}
		}
	cout << "System is set to use the following parameters..." << endl;
	cout << "sigma_dist: " << ThetaMaster[0] << endl;
	for (int i = 1; i < 1+pMaster; i++) {
		cout << "phi_" << i << ": " << ThetaMaster[i] << endl;
		}
	for (int i = 1+pMaster; i < 1+pMaster+qMaster; i++) {
		cout << "theta_" << i << ": " << ThetaMaster[i] << endl;
		}
	cout << endl;

	bool setSeedsYN = 0;
	unsigned int burnSeed = 1311890535, distSeed = 2603023340, noiseSeed = 2410288857;
	AcquireInput(cout,cin,"Supply seeds for light curve? 1/0: ","Invalid value.\n",setSeedsYN);
	if (setSeedsYN) {
		burnSeed = 0, distSeed = 0, noiseSeed = 0;
		do {
			cout << "All seeds must be strictly-positive integers (i.e. of type unsigned int - 10 digits max)." << endl;
			AcquireInput(cout,cin,"Burn-in phase disturbance seed: ","Invalid value.\n",burnSeed);
			} while (burnSeed <= 0);
		do {
			cout << "All seeds must be strictly-positive integers (i.e. of type unsigned int - 10 digits max)." << endl;
			AcquireInput(cout,cin,"Observation phase disturbance seed: ","Invalid value.\n",distSeed);
			} while (distSeed <= 0);
		do {
			cout << "All seeds must be strictly-positive integers (i.e. of type unsigned int - 10 digits max)." << endl;
			AcquireInput(cout,cin,"Observation phase noise seed: ","Invalid value.\n",noiseSeed);
			} while (noiseSeed <= 0);
		}
	cout << endl;

	int numBurn = 0;
	AcquireInput(cout,cin,"Number of burn-in steps for system: ","Invalid value.\n",numBurn);
	double* burnRand = static_cast<double*>(_mm_malloc(numBurn*sizeof(double),64));
	for (int i = 0; i < numBurn; i++) {
		burnRand[i] = 0.0;
		}
	cout << "Burning..." << endl;
	SystemMaster.burnSystem(numBurn, burnSeed, burnRand);
	_mm_free(burnRand);
	cout << "Burn phase complete!" << endl;
	cout << endl;

	int numObs = 0;
	AcquireInput(cout,cin,"Number of observations: ","Invalid value.\n",numObs);
	double noiseSigma = 0.0;
	do {
		cout << "Set the standard deviation of the noise (sigma_noise) such that sigma_noise > 0.0" << endl;
		AcquireInput(cout,cin,"Observation noise sigma_noise: ","Invalid value.\n",noiseSigma);
		} while (noiseSigma <= 0.0);
	double* distRand = static_cast<double*>(_mm_malloc(numObs*sizeof(double),64));
	double* noiseRand = static_cast<double*>(_mm_malloc(numObs*sizeof(double),64));
	double* y = static_cast<double*>(_mm_malloc(numObs*sizeof(double),64));
	double* yerr = static_cast<double*>(_mm_malloc(numObs*sizeof(double),64));
	for (int i = 0; i < numObs; i++) {
		distRand[i] = 0.0;
		noiseRand[i] = 0.0;
		y[i] = 0.0;
		yerr[i] = noiseSigma;
		}
	cout << "Observing..." << endl;
	SystemMaster.observeSystem(numObs, distSeed, noiseSeed, distRand, noiseRand, noiseSigma, y);

	cout << "Writing y" << endl;
	string yPath = "/home/exarkun/Desktop/y.dat";
	ofstream yFile;
	yFile.open(yPath);
	yFile.precision(16);
	for (int i = 0; i < numObs-1; i++) {
		yFile << noshowpos << scientific << y[i] << " " << yerr[i] << endl;
		}
	yFile << noshowpos << scientific << y[numObs-1] << " " << yerr[numObs-1] << endl;
	yFile.close();
	_mm_free(distRand);
	_mm_free(noiseRand);

	cout << "Computing LnLike..." << endl;
	SystemMaster.resetState();

	#ifdef TIME_LNLIKE
	#pragma omp barrier
	double timeBegLnLike = 0.0;
	double timeEndLnLike = 0.0;
	double timeTotLnLike = 0.0;
	timeBegLnLike = dtime();
	#endif

	double LnLike = SystemMaster.computeLnLike(numObs, y, yerr);

	#ifdef TIME_LNLIKE
	#pragma omp barrier
	timeEndLnLike = dtime();
	timeTotLnLike = timeEndLnLike - timeBegLnLike;
	cout << "Time taken: " << timeTotLnLike << endl;
	#endif

	cout << "LnLike: " << LnLike << endl;
	cout << endl;
	//printf("testMethod - Address of SystemMaster: %p\n",&SystemMaster);
	SystemMaster.deallocDLM();
	cout << endl;
	cout << endl;

	cout << "Starting MCMC Phase" << endl;
	cout << endl;

	int pMax = 0, qMax = 0;
	do {
		AcquireInput(cout,cin,"Maximum number of AR coefficients to test: ","Invalid value.\n",pMax);
		} while (pMax <= 0);
	qMax = pMax - 1;
	cout << endl;

	int nwalkers = 2*(pMax+qMax), nsteps = 0;
	do {
		AcquireInput(cout,cin,"Number of walkers to use: ","Invalid value.\n",nwalkers);
		} while (nwalkers < 2*(pMax+qMax));
	do {
		AcquireInput(cout,cin,"Number of steps to take: ","Invalid value.\n",nsteps);
		} while (nsteps <= 0);

	setSeedsYN = 0;
	unsigned int zSSeed = 2229588325, walkerSeed = 3767076656, moveSeed = 2867335446, initSeed = 3684614774;
	AcquireInput(cout,cin,"Supply seeds for MCMC? 1/0: ","Invalid value.\n",setSeedsYN);
	if (setSeedsYN) {
		zSSeed = 0, walkerSeed = 0, moveSeed = 0, initSeed = 0;
		do {
			cout << "All seeds must be strictly-positive integers (i.e. of type unsigned int - 10 digits max)." << endl;
			AcquireInput(cout,cin,"Zs Seed: ","Invalid value.\n",zSSeed);
			} while (zSSeed <= 0);
		do {
			cout << "All seeds must be strictly-positive integers (i.e. of type unsigned int - 10 digits max)." << endl;
			AcquireInput(cout,cin,"Walker choice seed: ","Invalid value.\n",walkerSeed);
			} while (walkerSeed <= 0);
		do {
			cout << "All seeds must be strictly-positive integers (i.e. of type unsigned int - 10 digits max)." << endl;
			AcquireInput(cout,cin,"Bernoulli move seed: ","Invalid value.\n",moveSeed);
			} while (moveSeed <= 0);
		do {
			cout << "All seeds must be strictly-positive integers (i.e. of type unsigned int - 10 digits max)." << endl;
			AcquireInput(cout,cin,"Initital positions seed: ","Invalid value.\n",initSeed);
			} while (initSeed <= 0);
		}
	cout << endl;

	int ndims = 1;

	LnLikeData Data;
	Data.numPts = numObs;
	Data.y = y;
	Data.yerr = yerr;

	//DLM* Systems = static_cast<DLM*>(_mm_malloc(nthreads*sizeof(DLM),64));

	LnLikeArgs Args;
	Args.numThreads = nthreads;
	Args.Data = Data;
	Args.Systems = nullptr;

	void* p2Args = nullptr;

	double* initPos = nullptr;
	VSLStreamStatePtr initStream;
	string myPath;
	ostringstream convertP, convertQ;

	DLM Systems[nthreads];

	for (int p = pMax; p > 0; --p) {
		for (int q = p-1; q > -1; q--) {

			cout << endl;
			cout << "Running MCMC for p = " << p << " and q = " << q << endl;
			int threadNum = omp_get_thread_num();
			//printf("testMethod - threadNum: %d\n",threadNum);

			ndims = p+q+1;

			//DLM Systems[nthreads];
			for (int tNum = 0; tNum < nthreads; tNum++) {
				//printf("testMethod - threadNum: %d; Address of Systems[%d]: %p\n",threadNum,tNum,&Systems[tNum]);
				//Systems[tNum] = DLM();
				Systems[tNum].allocDLM(p,q);
				cout << "Allocated " << Systems[tNum].allocated << " bytes for Systems[" << tNum << "]!" << endl;
				}

			Args.Systems = Systems;

			p2Args = &Args;

			EnsembleSampler newEnsemble = EnsembleSampler(ndims, nwalkers, nsteps, nthreads, 2.0, calcLnLike, p2Args, zSSeed, walkerSeed, moveSeed);

			initPos = static_cast<double*>(_mm_malloc(nwalkers*ndims*sizeof(double),64));
			vslNewStream(&initStream, VSL_BRNG_SFMT19937, initSeed);
			vslSkipAheadStream(initStream, nwalkers*(pMax+qMax+1)*(p*qMax+q));
			vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, initStream, nwalkers*ndims, initPos, 0.0, 1e-3);
			vslDeleteStream(&initStream);

			cout << "Running MCMC..." << endl;

			#ifdef TIME_MCMC
			#pragma omp barrier
			double timeBegMCMC = 0.0;
			double timeEndMCMC = 0.0;
			double timeTotMCMC = 0.0;
			timeBegMCMC = dtime();
			#endif
		
			newEnsemble.runMCMC(initPos);
		
			#ifdef TIME_MCMC
			#pragma omp barrier
			timeEndMCMC = dtime();
			timeTotMCMC = timeEndMCMC - timeBegMCMC;
			cout << "Time taken: " << timeTotMCMC/(60.0) << " (min)"<< endl;
			#endif

			printf("MCMC done. Writing result to ");
			convertP << p;
			convertQ << q;
			myPath = "/home/exarkun/Desktop/mcmcOut_" + convertP.str() + "_" + convertQ.str() + ".dat";
			cout << myPath << endl;
			newEnsemble.writeChain(myPath,1);
			convertP.str("");
			convertQ.str("");
			printf("Result written!\n");
			fflush(0);

			//cout << "Deallocating " << allocated << " bytes from Systems..." << endl;
			for (int tNum = 0; tNum < nthreads; tNum++) {
				//printf("testMethod - threadNum: %d; Address of Systems[%d]: %p\n",threadNum,tNum,&Systems[tNum]);
				Systems[tNum].deallocDLM();
				}
			//allocated = 0;
			//Args.Systems = nullptr;
			//p2Args = nullptr;
			_mm_free(initPos);
			//cout << endl;
			}
		}

	cout << endl;
	cout << "Deleting Systems..." << endl;
	cout << "Program exiting...Have a nice day!" << endl; 

	//_mm_free(Systems);
	_mm_free(y);
	_mm_free(yerr);
	_mm_free(ThetaMaster);
	}