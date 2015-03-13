CPPC = icpc

IDIR = include/
SRCDIR = src
ODIR = src/obj
BOOSTLIB = ~/Documents/Research/code/trunk/Boost/boost_1_57_0/

VERFLAGS = -gxx-name=g++-4.8 -std=c++11 -g -Wall

#CPPFLAGS = -std=c++11 -O3 -xHost -ip -parallel -funroll-loops -fno-alias -fno-fnalias -fargument-noalias

#CPPFLAGS = -std=c++11 -O3 -xHost -ip -parallel -funroll-loops -fno-alias -fno-fnalias -fargument-noalias -no-ansi-alias

CPPFLAGS = -O3 -ip -parallel -funroll-loops -fno-alias -fno-fnalias -fargument-noalias -fstrict-aliasing -ansi-alias -fno-stack-protector-all
#-opt-streaming-stores always

OFFLOAD_FLAGS =
#OFFLOAD_FLAGS = -offload=optional

#MKL Flags.
MKLFLAGS = -DMKL_ILP64 -I$(MKLROOT)/include
#-mkl=sequential
#-offload-attribute-target=mic
#MKLFLAGS = -DMKL_ILP64 -I$(MKLROOT)/include -offload-option,mic,compiler,"$(MKLROOT)/lib/mic/libmkl_intel_ilp64.a $(MKLROOT)/lib/mic/libmkl_intel_thread.a $(MKLROOT)/lib/mic/libmkl_core.a" -offload-attribute-target=mic

#MKL link line. 
#MKLLINKLINE = -lpthread -lm
#MKLLINKLINE = -Wl,--start-group  $(MKLROOT)/lib/intel64/libmkl_intel_ilp64.a $(MKLROOT)/lib/intel64/libmkl_intel_thread.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm
MKL_LIBS=-L$(MKLROOT)/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -lpthread -lm
MKL_MIC_LIBS=-L$(MKLROOT)/lib/mic -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core

NLOPTLIBS = -lnlopt
OMPFLAGS = -openmp -openmp-simd

REPORTFLAG = -qopt-report=0 -qopt-report-phase=vec -qopt-report-file=stdout -openmp-report=0
#-guide
# -opt-report-phase=offload

#FPFLAGS = -fp-model strict -fp-model extended -fimf-arch-consistency=true -fimf-precision=high -no-fma 
# enable <name> floating point model variation
#     except[-]  - enable/disable floating point semantics
#     extended   - enables intermediates in 80-bit precision
#     fast       - allows value-unsafe optimizations
#     precise    - allows value-safe optimizations
#     source     - enables intermediates in source precision
#     strict     - enables -fp-model precise -fp-model except and disables floating point multiply add

_DEPENDENCIES = Constants.hpp Utilities.hpp AcquireInput.hpp Universe.hpp Spherical.hpp Obj.hpp Kepler.hpp Kalman.hpp MCMC.hpp
#PRH.hpp DLAPACKE.hpp
DEPENDENCIES = $(patsubst %,$(IDIR)/%,$(_DEPENDENCIES))

_OBJECTS = Constants.o Utilities.o Universe.o Spherical.o Obj.o Kepler.o Kalman.o MCMC.o
# PRH.o DLAPACKE.o
OBJECTS = $(patsubst %,$(ODIR)/%,$(_OBJECTS))

EXEC3 = testKalmanCPP
EXEC1 = testPoint
EXEC2 = testMethod
EXT = .cpp

all: $(EXEC1) $(EXEC2)
# $(EXEC3)

$(EXEC1): $(OBJECTS) $(patsub %,$(EXEC1)%,$(EXT))
	$(CPPC) $(VERFLAGS) -xHost $(CPPFLAGS) $(FPFLAG) $(MKLFLAGS) $(OMPFLAGS) -I $(IDIR)  $(REPORTFLAG) $^ $(SRCDIR)/$(EXEC1)$(EXT) $(OMPFLAGS) $(MKL_LIBS) -o $@

$(EXEC2): $(OBJECTS) $(patsub %,$(EXEC2)%,$(EXT))
	$(CPPC) $(VERFLAGS) -xHost $(CPPFLAGS) $(FPFLAG) $(MKLFLAGS) $(OMPFLAGS) -I $(IDIR)  $(REPORTFLAG) $^ $(SRCDIR)/$(EXEC2)$(EXT) $(OMPFLAGS) $(MKL_LIBS) -o $@

$(EXEC3): $(OBJECTS) $(patsub %,$(EXEC3)%,$(EXT))
	$(CPPC) $(VERFLAGS) -xHost $(CPPFLAGS) $(FPFLAG) $(MKLFLAGS) $(OMPFLAGS) -I $(IDIR)  $(REPORTFLAG) $^ $(SRCDIR)/$(EXEC3)$(EXT) $(OMPFLAGS) $(MKL_LIBS) -o $@

$(ODIR)/Universe.o: $(SRCDIR)/Universe.cpp $(IDIR)/Universe.hpp
	$(CPPC) -c $(VERFLAGS) -xHost $(CPPFLAGS) $(OFFLOAD_FLAGS) $(FPFLAGS) -I $(IDIR) -I $(BOOSTLIB) $< -o $@

$(ODIR)/Spherical.o: $(SRCDIR)/Spherical.cpp $(IDIR)/Spherical.hpp
	$(CPPC) -c $(VERFLAGS) $(CPPFLAGS) $(OFFLOAD_FLAGS) $(FPFLAGS) -I $(IDIR) -I $(BOOSTLIB) $< -o $@

$(ODIR)/Kalman.o: $(SRCDIR)/Kalman.cpp $(IDIR)/Kalman.hpp
	$(CPPC) -c $(VERFLAGS) $(CPPFLAGS) $(OMPFLAGS) $(FPFLAGS) $(REPORTFLAG) -I $(IDIR) $< -o $@

$(ODIR)/MCMC.o: $(SRCDIR)/MCMC.cpp $(IDIR)/MCMC.hpp
	$(CPPC) -c $(VERFLAGS) $(CPPFLAGS) $(OMPFLAGS) $(FPFLAGS) $(REPORTFLAG) -I $(IDIR) $< -o $@

$(ODIR)/%.o: $(SRCDIR)/%.cpp $(DEPENDENCIES)
	$(CPPC) -c $(VERFLAGS) -xHost $(CPPFLAGS) $(FPFLAGS) $(OMPFLAGS) -I $(IDIR) $< -o $@

.PHONY: clean
.PHONY: cleanExec
clean:
	rm -f $(ODIR)/*.o *~ $(EXEC) $(SRCDIR)/*~ $(IDIR)*~
	rm $(EXEC1)
	rm $(EXEC2)
	rm $(EXEC3)

clean$(EXEC1):
	rm $(EXEC1)

clean$(EXEC2):
	rm $(EXEC2)

clean$(EXEC3):
	rm $(EXEC3)