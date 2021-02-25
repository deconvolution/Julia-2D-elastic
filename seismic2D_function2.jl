function meshgrid(x,y)
    x2=zeros(length(x),length(y));
    y2=x2;
    x2=repeat(x,1,length(y));
    y2=repeat(reshape(y,1,length(y)),length(x),1);
    return x2,y2
end

function write2mat(path,var)
    file=matopen(path,"w");
    write(file,"data",data);
    close(file);
    return 0
end

function rickerWave(freq,dt,ns,M)
    ## calculate scale
    E=10 .^(5.24+1.44 .*M);
    s=sqrt(E.*freq/.299);

    t=dt:dt:dt*ns;
    t0=1 ./freq;
    t=t .-t0;
    ricker=s .*(1 .-2*pi^2*freq .^2*t .^2).*exp.(-pi^2*freq^2 .*t .^2);
    ricker=ricker;
    ricker=Float32.(ricker);
    return ricker
end
##
@parallel function compute_sigma(dt,dx,dz,C11,C13,C33,C55,v1,v3,beta,
    sigmas11,sigmas13,sigmas33,p)

    @inn(sigmas11)=dt*.5*((@all(C11)-@all(C13)) .*@d_xi(v1)/dx+
    (@all(C13)-@all(C33)) .*@d_yi(v3)/dz)+
    @inn(sigmas11)-
    dt*@inn(beta) .*@inn(sigmas11);

    @inn(sigmas33)=dt*.5*((@all(C33)-@all(C13)) .*@d_yi(v3)/dz+
    (@all(C13)-@all(C11)) .*@d_xi(v1)/dx)+
    @inn(sigmas33)-
    dt*@inn(beta).*@inn(sigmas33);

    @all(sigmas13)=dt*(@all(C55) .*(@d_ya(v1)/dz+@d_xa(v3)/dx))+
    @all(sigmas13)-
    dt*@all(beta).*@all(sigmas13);

    # p
    @inn(p)=-dt*((@all(C11)+@all(C33))*.5 .*@d_xi(v1)/dx+
    (@all(C13)+@all(C33))*.5 .*@d_yi(v3)/dz)+
    @all(p)-
    dt*@all(beta).*@all(p);

    return nothing
end
##
@parallel function compute_v(dt,dx,dz,rho,v1,v3,beta,
    sigmas11,sigmas33,sigmas13,p)

    @inn(v1)=dt ./@all(rho) .*((@d_xa(sigmas11)-@d_xa(p))/dx);

    @inn(v1)=dt ./@inn(rho) .*(@d_yi(sigmas13)/dz)+
    @inn(v1)-
    dt*@all(beta) .*@inn(v1);

    @inn(v3)=dt ./@all(rho) .*((@d_yi(sigmas33)-@d_yi(p))/dz);

    @inn(v3)=dt ./@all(rho) .*@d_xi(sigmas13)/dx+
    @inn(v3)-
    dt*@all(beta) .*@inn(v3);

    return nothing
end

@timeit ti "VTI_2D" function VTI_2D(dt,dx,dz,nt,nx,nz,
    X,Z,
    r1,r3,
    s1,s3,src1,src3,source_type,
    r1t,r3t,
    s1t,s3t,
    lp,nPML,Rc,
    C,
    plot_interval,plot_source,
    path,
    save_wavefield)

    d0=Dates.now();

    #create folder for figures
    if isdir(path)==0
        mkdir(path);
    end

    n_picture=1;
    if save_wavefield==1
        if isdir(string(path,"/forward_wavefield/"))==0;
            mkdir(string(path,"/forward_wavefield/"));
        end
    end

    if plot_interval!=0
        if isdir(string(path,"/forward_pic/"))==0
            mkdir(string(path,"/forward_pic/"))
        end
    end

    # PML
    vmax=sqrt.((C.C33) ./C.rho);
    beta0=(ones(nx,nz) .*vmax .*(nPML+1) .*log(1/Rc)/2/lp/dx);
    beta1=(@zeros(nx,nz));
    beta3=beta1;
    tt=(1:lp)/lp;
    tt2=repeat(reshape(tt,lp,1),1,nz);
    plane_grad1=@zeros(nx,nz);
    plane_grad3=plane_grad1;

    plane_grad1[2:lp+1,:]=reverse(tt2,dims=1);
    plane_grad1[nx-lp:end-1,:]=tt2;
    plane_grad1[1,:]=plane_grad1[2,:];
    plane_grad1[end,:]=plane_grad1[end-1,:];

    tt2=repeat(reshape(tt,1,lp),nx,1);
    plane_grad3[:,2:lp+1]=reverse(tt2,dims=2);
    plane_grad3[:,nz-lp:end-1]=tt2;
    plane_grad3[:,1]=plane_grad3[:,2];
    plane_grad3[:,end]=plane_grad3[:,end-1];

    beta1=beta0.*plane_grad1.^nPML;
    beta3=beta0.*plane_grad3.^nPML;

    IND=unique(findall(f-> f!=0,beta1.*beta3));
    beta=beta1+beta3;
    beta[IND]=beta[IND]/2;

    beta1=beta3=plane_grad1=plane_grad3=vmax=nothing;

    # receiver configuration
    R1=@zeros(nt,length(r1));
    R3=@zeros(nt,length(r3));
    P=@zeros(nt,length(r3));

    # wave vector
    v1=@zeros(nx,nz+1);
    v3=@zeros(nx+1,nz);

    sigmas11=@zeros(nx+1,nz);
    sigmas13=@zeros(nx,nz);
    sigmas33=@zeros(nx,nz+1);
    p=@zeros(nx+1,nz+1);

    l=1;
    @timeit ti "source" if source_type=='D'
        v1[CartesianIndex.(s1,s3)]=v1[CartesianIndex.(s1,s3)]+ .5 ./C.rho[CartesianIndex.(s1,s3)] .*src1[l];
        v3[CartesianIndex.(s1,s3)]=v3[CartesianIndex.(s1,s3)]+ .5 ./C.rho[CartesianIndex.(s1,s3)] .*src3[l];
    end

    @timeit ti "source" if source_type=='P'
         p[CartesianIndex.(s1,s3)]=p[CartesianIndex.(s1,s3)]+src3[l];
    end

    ## save wavefield
    if save_wavefield==1
        data=zeros(nx,nz);
        write2mat(string(path,"/forward_wavefield/v1_",l,".mat"),data);
        data=zeros(nx,nz);
        write2ma(string(path,"/forward_wavefield/v3_",l,".mat"),data);
        data=zeros(nx,nz);
        write2mat(string(path,"/forward_wavefield/sigmas11_",l,".mat"),data);
        data=zeros(nx,nz);
        write2mat(string(path,"/forward_wavefield/sigmas33_",l,".mat"),data);
        data=zeros(nx,nz);
        write2mat(string(path,"/forward_wavefield/sigmas13_",l,".mat"),data);
        data=zeros(nx,nz);
        write2mat(string(path,"/forward_wavefield/p_",l,".mat"),data);

        data=v1;
        write2mat(string(path,"/forward_wavefield/v1_",l+1,".mat"),data);
        data=v3;
        write2mat(string(path,"/forward_wavefield/v3_",l+1,".mat"),data);
        data=sigmas11;
        write2mat(string(path,"/forward_wavefield/sigmas11_",l+1,".mat"),data);
        data=sigmas33;
        write2mat(string(path,"/forward_wavefield/sigmas33_",l+1,".mat"),data);
        data=sigmas13;
        write2mat(string(path,"/forward_wavefield/sigmas13_",l+1,".mat"),data);
        data=p;
        write2mat(string(path,"/forward_wavefield/p_",l+1,".mat"),data);
    end
    ##

    for l=2:nt-1
        @timeit ti "compute_sigma" @parallel compute_sigma(dt,dx,dz,C.C11,C.C13,
        C.C33,C.C55,v1,v3,beta,
        sigmas11,sigmas13,sigmas33,p);

        @timeit ti "compute_v" @parallel compute_v(dt,dx,dz,rho,v1,v3,beta,
            sigmas11,sigmas33,sigmas13,p);

        @timeit ti "source" if source_type=='D'
            v1[CartesianIndex.(s1,s3)]=v1[CartesianIndex.(s1,s3)]+ 1 ./C.rho[CartesianIndex.(s1,s3)] .*src1[l];
            v3[CartesianIndex.(s1,s3)]=v3[CartesianIndex.(s1,s3)]+ 1 ./C.rho[CartesianIndex.(s1,s3)] .*src3[l];
        end

        @timeit ti "source" if source_type=='P'
             p[CartesianIndex.(s1,s3)]=p[CartesianIndex.(s1,s3)]+src3[l];
        end

        # assign recordings
        @timeit ti "receiver" R1[l+1,:].=v1[CartesianIndex.(r1,r3)];
        @timeit ti "receiver" R3[l+1,:].=v3[CartesianIndex.(r1,r3)];
        @timeit ti "receiver" P[l+1,:].=p[CartesianIndex.(r1,r3)];
        # save wavefield
        if save_wavefield==1
            data=v1;
            write2mat(string(path,"/forward_wavefield/v1_",l+1,".mat"),data);
            data=v3;
            write2mat(string(path,"/forward_wavefield/v3_",l+1,".mat"),data);
            data=sigmas11;
            write2mat(string(path,"/forward_wavefield/sigmas11_",l+1,".mat"),data);
            data=sigmas33;
            write2mat(string(path,"/forward_wavefield/sigmas33_",l+1,".mat"),data);
            data=sigmas13;
            write2mat(string(path,"/forward_wavefield/sigmas13_",l+1,".mat"),data);
            data=p;
            write2mat(string(path,"/forward_wavefield/p_",l+1,".mat"),data);
        end

        # plot
        l2=l;
        if plot_interval!=0
            if mod(l,plot_interval)==0 || l==nt-1
                mat"
                hfig=figure('Visible','off');
                set(hfig,'position',[0,0,1200,600]);
                subplot(2,3,1)
                imagesc([min($X,[],'all'),max($X,[],'all')],[max($Z,[],'all'),min($Z,[],'all')],$v3');
                axis on;
                hold on;
                set(gca,'ydir','normal');
                colorbar;
                xlabel({['x [m]']});
                ylabel({['z [m]']});
                title({['t=' num2str($l2*$dt) 's'],['v_3 [m/s]']});
                xlabel('x [m]');
                ylabel('z [m]');
                colorbar;
                hold on;
                ax2=plot($s1t,$s3t,'v','color',[1,0,0]);
                hold on;
                ax4=plot($r1t,$r3t,'^','color',[0,1,1]);

                subplot(2,3,2)
                imagesc([min($X,[],'all'),max($X,[],'all')],[max($Z,[],'all'),min($Z,[],'all')],$sigmas33');
                axis on;
                set(gca,'ydir','normal');
                xlabel('x [m]');
                ylabel('z [m]');
                title('sigmas33 [Pa]');
                colorbar;

                subplot(2,3,3)
                imagesc([min($X,[],'all'),max($X,[],'all')],[max($Z,[],'all'),min($Z,[],'all')],$sigmas13');
                axis on;
                set(gca,'ydir','normal');
                xlabel('x [m]');
                ylabel('z [m]');
                title('sigmas13 [Pa]');
                colorbar;

                subplot(2,3,4)
                imagesc([min($X,[],'all'),max($X,[],'all')],[max($Z,[],'all'),min($Z,[],'all')],$p');
                axis on;
                set(gca,'ydir','normal');
                xlabel('x [m]');
                ylabel('z [m]');
                title('p [Pa]');
                colorbar;

                subplot(2,3,5)
                axis on;
                imagesc([1,length($r1)],[1,($l2+1)]*$dt,$R3(1:$l2+1,:));
                colorbar;
                xlabel('Nr');
                ylabel('t [s]');
                title('R3 [m/s]');
                ylim([1,$nt]*$dt);

                subplot(2,3,6)
                axis on;
                imagesc([min($X,[],'all'),max($X,[],'all')],[max($Z,[],'all'),min($Z,[],'all')],$C.C33');
                set(gca,'ydir','normal');
                xlabel('x [m]');
                ylabel('z [m]');
                title('C33 [Pa]');
                colorbar;
                hold on;
                ax2=plot($s1t,$s3t,'v','color',[1,0,0]);
                hold on;
                ax4=plot($r1t,$r3t,'^','color',[0,1,1]);

                legend([ax2,ax4],...
                'source','receiver',...
                'Location',[0.5,0.02,0.005,0.002],'orientation','horizontal');

                print(gcf,[$path './forward_pic/' num2str($n_picture) '.png'],'-dpng','-r200');
                "
                n_picture=n_picture+1;
            end
        end
        #=
        fprintf('\n time step=%d/%d',l+1,nt);
        fprintf('\n    epalsed time=%.2fs',toc);
        fprintf('\n    n_picture=%d',n_picture);
        d=clock;
        fprintf('\n    current time=%d %d %d %d %d %.0d',d(1),d(2),d(3),d(4),d(5),d(6));
        =#
        d=Dates.now();
        println("\n time step=",l+1,"/",nt);
        println("\n    n_picture=",n_picture);
        println("\n    ",d);
    end
    return v1,v3,R1,R3,P
end
