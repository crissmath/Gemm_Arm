void gemm_blis( char, char, int, int, int, float, float *, int, float *, int, float, float *, int, 
                float *, float *, int, int, int);
void gemm_macrokernel( int, int, int, float, float *, int, float *, int, float, float *, int );
void gemm_microkernel_reference_4x4( int, float, float *, float *, float, float *, int );
void gemm_microkernel_reference_register_4x4( int, float, float *, float *, float, float *, int );
void gemm_base( int, int, int, float, float *, int, float *, int, float, float *, int );
void pack_A( char, int, int, float *, int, float * );
void pack_B( char, int, int, float *, int, float * );

//void gemm_microkernel_neon_4x4( int, int, int, float, float *, float *, float, float *, int );

