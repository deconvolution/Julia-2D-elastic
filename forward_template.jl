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
include("./seismic2D_function.jl");
## load image
vp=@ones(4000,4000)*2000;
nx,nz=size(vp);

##
# dimensions
dt=10^-3;
dx=10;
dz=10;
nt=100;
nx=size(vp,1);
nz=size(vp,2);

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

s_s1=[200];
s_s3=ones(Int32,size(s_s1))*200;
##
# point interval in time steps
plot_interval=0;

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
    source_type='D';

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
## end
    #=
    write to gif
    sources=[path 'pic/'];
    delaytime=.2;
    filename=['animation'];
    gifmaker(filename,delaytime,sources);
    # write recordings
    ## recording saving
    if ~exist([path '/rec/'],'dir')
    mkdir([path '/rec/'])
end

DATA=rec_conversion(nt,nx,nz,dt,dx,dz,s1,s3,r1,r3);
parsave([path '/rec/simu_info.mat'],DATA);

DATA=R1;
parsave([path '/rec/tR1.mat'],DATA);
DATA=R3;
parsave([path '/rec/tR3.mat'],DATA);
DATA=P;
parsave([path '/rec/tP.mat'],DATA);
%% source saving
%% recording saving
if ~exist([path '/source/'],'dir')
mkdir([path '/source/'])
end
DATA=src1;
parsave([path '/source/src1.mat'],DATA);
DATA=src3;
parsave([path '/source/src3.mat'],DATA);
DATA=source_type;
parsave([path '/source/source_type.mat'],DATA);
end
toc;
=#
##
