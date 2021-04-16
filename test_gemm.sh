export TRANSA=N
export TRANSB=N

export ALPHA=1.0 
export BETA=0.0   

export MMIN=103
export MMAX=227
export MSTEP=31

export NMIN=103
export NMAX=227
export NSTEP=31

export KMIN=103
export KMAX=227
export KSTEP=31

"""
export MMIN=1000
export MMAX=3000 
export MSTEP=1000

export NMIN=1000
export NMAX=3000
export NSTEP=1000

export KMIN=1000
export KMAX=3000
export KSTEP=1000
"""

export MCMIN=20   
export MCMAX=30   
export MCSTEP=5   

export NCMIN=20   
export NCMAX=30   
export NCSTEP=5   

export KCMIN=20  
export KCMAX=30  
export KCSTEP=5  

export VISUAL=0
export TIMIN=1.0 

export TEST=T

./test_gemm.x $TRANSA $TRANSB $ALPHA $BETA $MMIN $MMAX $MSTEP $NMIN $NMAX $NSTEP $KMIN $KMAX $KSTEP $MCMIN $MCMAX $MCSTEP $NCMIN $NCMAX $NCSTEP $KCMIN $KCMAX $KCSTEP $VISUAL $TIMIN $TEST
