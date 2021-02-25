using MAT
using Plots
using MATLAB
using Dates
const USE_GPU = false  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
using TimerOutputs
ti=TimerOutput();
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2);
else
    @init_parallel_stencil(Threads, Float64, 2);
end
include("./seismic2D_function3.jl");
## load image
nx=200;
nz=200;
vp=@ones(nx,nz)*2000;
##
# dimensions
dt=10^-3/2;
dx=10;
dz=10;
nt=1000;

X=(1:nx)*dx;
Z=-(1:nz)*dz;

# PML layers
lp=20;

# PML coefficient, usually 2
nPML=2;

# Theoretical coefficient, more PML layers, less R
# Empirical values
# lp=[10,20,30,40]
# R=[.1,.01,.001,.0001]
Rc=.0001;

# generate empty density
rho=@ones(nx,nz)*1;
#rho=rho.*A;
# Lame constants for solid
mu=rho.*(vp/sqrt(3)).^2;
lambda=rho.*vp.^2;
##
mutable struct C2
    C11
    C13
    C33
    C55
    rho
end
C=C2(lambda+2*mu,lambda,lambda+2*mu,mu,rho);

# source
# magnitude
M=2.7;

s_s1=Int(round(nx/2));
s_s3=Int(round(nz/2));
##
# point interval in time steps
plot_interval=100;

p2= @__FILE__;
if isdir(chop(p2,head=0,tail=3))==0
    mkdir(chop(p2,head=0,tail=3))
end;
##for source_code=1
@time begin
    source_code=1;
    # source locations
    s1=s_s1[source_code];
    s3=s_s3[source_code];

    # source frequency [Hz]
    freq=5;

    # source signal
    singles=rickerWave(freq,dt,nt,M);

    # give source signal to x direction
    src1=zeros(Float32,nt,1);
    src1=1*repeat(singles,1,length(s3));

    # give source signal to z direction
    src3=copy(src1);
    src3=1*repeat(singles,1,length(s3));

    # receiver locations [m]
    r1=3;
    r3=3;

    s1t=dx .*s1;
    s3t=maximum(Z)-dz .*s3;
    r1t=dx .*r1;
    r3t=maximum(Z)-dz .*r3;

    # source type. 'D' for directional source. 'P' for P-source.
    source_type='P';

    # plot source
    plot_source=1;

    # path for this source
    path=string(chop(p2,head=0,tail=3),"/source_code_",(source_code),"/");

    # save wavefield
    save_wavefield=0;
    ## pass parameters to solver
    v1,v3,R1,R3,P=VTI_2D(dt,dx,dz,nt,nx,nz,
    X,Z,
    r1,r3,
    s1,s3,src1,src3,source_type,
    r1t,r3t,
    s1t,s3t,
    lp,nPML,Rc,
    C,
    plot_interval,plot_source,
    path,
    save_wavefield);
end
##
A=zeros(5,4);
B=[1 2 3 4; 4 4 4 4; 0 1 5 6;8 8 8 8;3 4 5 6];
@parallel function f(A,B)
    @inn(A)=@d_xi(B);
    return nothing
end

@time @parallel f(A,B)
##
A2=@zeros(4,4);
@parallel function h(A,B)
    @all(A)=@inn_x(B);
    return nothing
end
@parallel h(A2,B)
##
A2=[A;0 0 0 0];
C=zeros(5,4);
@parallel function g(A,B)
    @all(A)=@inn_x(B);
    return nothing
end

@time @parallel g(C,A2)
