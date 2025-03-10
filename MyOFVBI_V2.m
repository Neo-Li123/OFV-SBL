% The code aims to achieve off-grid target estimation by sparse Bayesian
% learning in a multi-static (multi-observation) system employing one-bit
% ADCs
% This code apply posterior statistic information to execute grid refine, rather
% than applying first-order Taylor expansion.
% The minimal interval is considered.
% Code date:25/02/28
function [x_est,Result_est]=MyOFVBI_V2(y,A,h,grid_size,bits,Para,Grid,K,Loc_tar,varargin)
% Input:
% y: one bit quantized data
% A: dictionary matrix
% h: quantization threshold
% grid_size: grid size where is assumed to be identical for x or y axis.
% bits: bit depth
% var_n: variance of noise
% block_size: lenght of block in the sparse structure
% max_iters_in : maximum iterations of inner loop
% Output:
% Result_est: Finally estimated result, included  posterior mean and
% covariance matrix of sparse vector, and variance and hermitian matrix of prior
%====================Initial===========================================
var_n=1;
block_size=1;
max_iters_in=2e1;
max_iters=2e2;
PRINT=0;
conver_thres=1e-3;
[num_row,num_col]=size(A);
grid_x=Grid.x;
grid_y=Grid.y;
Interest_ratio=0.8;
num_neigh_del=1;
ratio_dis=0.2;
K_known=false;
for i=1:2:(length(varargin)-1)
    switch lower(varargin{i})
        case 'max_iters'
            max_iters = varargin{i+1};
        case 'max_iters_in'
            max_iters_in = varargin{i+1};
        case 'print'
            PRINT = varargin{i+1};
        case 'var_n'
            var_n=varargin{i+1};
        case 'tol'
            conver_thres=varargin{i+1};
        case 'block_size'
            block_size=varargin{i+1};
        case 'k_known'
            K_known=varargin{i+1};            
        otherwise
            error(['Unrecognized parameter: ''' varargin{i} '''']);
    end
end
% Transform A,D into MN submatrices A_multi
N=num_col/block_size;
M=num_row/block_size;
grid_size_x=ones(N,1)*grid_size;
grid_size_y=ones(N,1)*grid_size;
A_multi=zeros(M,N,block_size);
Mean_x=zeros(N,block_size);
Prod=zeros(num_row,1);
for n=1:block_size
    A_multi(:,:,n)=A((n-1)*M+1:n*M,n:block_size:num_col);
    Mean_x(:,n)=pinv(A_multi(:,:,n))*y((n-1)*M+1:n*M);
end
gamma=1e-1*ones(N,1); % The variance in the sparse prior
y_r=real(y);
y_i=imag(y);
% Mean_x=zeros(N,block_size);
% mean_x=1e-1*ones(num_col,1);
mean_x=reshape(Mean_x,num_col,1);
% mean_x=1e-1*pinv(A)*y;
% Mean_x=reshape(mean_x,N,block_size);
mean_z=Prod-h;
Sigma_y=zeros(M,M,block_size);
Sigma_y_inv=zeros(M,M,block_size);
dmx_report=[];
% x_report=[];
k=6;
min_interval=grid_size/2^k;
GridRef_iters=2^(k-2);
Grid_2D=[grid_x,grid_y];
Loc_add=[-1e4,-1e4];
grid_x_unorder=unique(grid_x);
grid_y_unorder=unique(grid_y);
if PRINT
    fprintf('=============Off grid VBI============================\n');
    fprintf(' The maximum iteration is %d.\n', max_iters);
    fprintf(' The maximum inner iteration is %d.\n', max_iters_in);
    fprintf(' The mimunm grid interval is %d.\n', min_interval);
    fprintf(' The convergent criterion is %d\n :',conver_thres);
    fprintf('=============================================\n');
end

for ii=1:GridRef_iters
    %% ===============EM procedure===========================================
    % tic
    %-------------variational E step------------------------------------
    mean_x_old=mean_x;
    for tt=1:block_size
        Gamma=diag(gamma);
        Sigma_y(:,:,tt)=eye(M)*var_n+A_multi(:,:,tt)*Gamma*A_multi(:,:,tt)';
        if M<N
            Sigma_y_inv(:,:,tt)=inv(Sigma_y(:,:,tt));
            % PYP=A_multi(:,:,tt)'/Sigma_y(:,:,tt)*A_multi(:,:,tt);
            PYP_diag=zeros(N,1);
            for nn=1:N
                PYP_diag(nn)=A_multi(:,nn,tt)'*Sigma_y_inv(:,:,tt)*A_multi(:,nn,tt);
            end
            gpg=gamma.*PYP_diag.*gamma;
            Sigma_diag_tt=gamma-gpg;
            % Sigma(:,:,tt)=Sigma_tt;
            Sigma_diag(:,tt)=real(Sigma_diag_tt);
        else
            PVP_tt=A_multi(:,:,tt)'*A_multi(:,:,tt)/var_n;
            Sigma_tt=inv(diag(gamma.^-1)+PVP_tt);
            Sigma(:,:,tt)=Sigma_tt;
            Sigma_diag(:,tt)=real(diag(Sigma_tt));
        end

    end
    if bits==1
        for jj=1:max_iters_in
            mean_z_old=mean_z;
            for t=1:block_size
                seg_M=(t-1)*M+1:t*M;
                %seg_N=(t-1)*N+1:t*N;
                if M<N
                    Mean_x(:,t)=gamma.*(A_multi(:,:,t)'*Sigma_y_inv(:,:,t)*mean_z(seg_M));
                else
                    Mean_x(:,t)=Sigma(:,:,t)*A_multi(:,:,t)'*mean_z(seg_M)/var_n;
                end
                Prod(seg_M)=A_multi(:,:,t)*Mean_x(:,t);
            end
            z_temp=Prod-h;
            chi_r=y_r.*real(z_temp)*sqrt(2/var_n);
            chi_i=y_i.*imag(z_temp)*sqrt(2/var_n);

            mean_z=sqrt(var_n/2)*(y_r.*logcdf_1der(chi_r)+1j*y_i.*logcdf_1der(chi_i))+Prod;

            dmz=max(abs(mean_z_old-mean_z));
            if dmz<conver_thres
                break;
            end

        end
    else
        mean_z=y;
        for t=1:block_size
            seg_M=(t-1)*M+1:t*M;
            %seg_N=(t-1)*N+1:t*N;
            if M<N
                Mean_x(:,t)=gamma.*(A_multi(:,:,t)'*Sigma_y_inv(:,:,t)*mean_z(seg_M));
            else
                Mean_x(:,t)=Sigma(:,:,t)*A_multi(:,:,t)'*mean_z(seg_M)/var_n;
            end
        end
    end
    mean_x=reshape(Mean_x,[],1);
    % if length(mean_x_old)<length(mean_x)
    %     mean_x_old=[mean_x_old;zeros(length(mean_x)-)]
    % dmx=max(max(abs(mean_x_old-mean_x)));
    %     dmx_conv=norm(mean_x_old-mean_x)/norm(mean_x_old);
    %-------------variational M step------------------------------------
    % update prior noise
    gamma_old=gamma;
    secmom=abs(Mean_x).^2+Sigma_diag;
    gamma=mean(secmom,2);
    dmx=max(max(abs(gamma_old-gamma)));
    % toc

    %% ------------------grid refine--------------------------------------

    [~,Ind_temp]=sort(gamma,'descend');
    Ind_select_2D=sort(Ind_temp(1:floor(N*Interest_ratio)));
    Ind_select_2D(Ind_select_2D==1)=[];Ind_select_2D(Ind_select_2D==N)=[];

    Loc_select_2D=Grid_2D(Ind_select_2D,:);
    grid_x_s_2D=Loc_select_2D(:,1);
    grid_y_s_2D=Loc_select_2D(:,2); % 这一步只能获得1D的坐标，但是无法获得1D的index。
    [~,ind_x_s_1D]=ismember(grid_x_s_2D,grid_x_unorder);
    [~,ind_y_s_1D]=ismember(grid_y_s_2D,grid_y_unorder);

    grid_x_s_1D=grid_x_unorder(ind_x_s_1D);
    grid_y_s_1D=grid_y_unorder(ind_y_s_1D);% 根据上述准则保留2d网格域中坐标信息
    Sigma_del_dd=zeros(M,M,block_size);
    Loc_add_cur=[];

    %-------------------plot gamma--------------------------------------
    figure(199);
    Ind_tar=sort(Ind_temp(1:K));
    Loc_tar_est=Grid_2D(Ind_tar,:);
    plot(Loc_tar(1),Loc_tar(2),'bo' )
    % plot(Loc_tar(3),Loc_tar(4),'b+');
    hold on 
    % plot(Loc_tar_est(1),Loc_tar_est(5),'rx');
    % plot(Loc_tar_est(2),Loc_tar_est(6),'rx');
    % plot(Loc_tar_est(3),Loc_tar_est(7),'rx');
    % plot(Loc_tar_est(4),Loc_tar_est(8),'rx');
    plot(Loc_tar_est(1),Loc_tar_est(2),'rx');
    axis([-50 50 -50 50]);
    grid on;xlabel('X axe');ylabel('Y axe');
    hold off

    N_y_unorder=length(grid_y_unorder);N_x_unorder=length(grid_x_unorder);
    count_effect=0;
    for dd=1:length(Ind_select_2D)
        % tic
        idx_dd=Ind_select_2D(dd);
        idx_x_dd=ind_x_s_1D(dd);
        idx_y_dd=ind_y_s_1D(dd);
        Loc_dd=Grid_2D(idx_dd,:);
        % if norm(Loc_dd-Loc_tar')<=4*min_interval
        %     keyboard;
        % end        
        grid_x_l_dd=grid_x_unorder((idx_x_dd==1)+idx_x_dd-1);
        grid_x_r_dd=grid_x_unorder(-(idx_x_dd==N_x_unorder)+idx_x_dd+1);
        grid_y_l_dd=grid_y_unorder((idx_y_dd==1)+idx_y_dd-1);
        grid_y_r_dd=grid_y_unorder(-(idx_y_dd==N_y_unorder)+idx_y_dd+1);

        ind_region_x=intersect(find(Grid_2D(:,1)>=grid_x_l_dd),find(Grid_2D(:,1)<=grid_x_r_dd));
        ind_region_y=intersect(find(Grid_2D(:,2)>=grid_y_l_dd),find(Grid_2D(:,2)<=grid_y_r_dd));
        ind_region_near=intersect(ind_region_x,ind_region_y);
        region_near_ud=Grid_2D(ind_region_near,:);
        Dist_dd=sqrt(sum(abs(repmat(Loc_dd,length(ind_region_near),1)-region_near_ud).^2,2));
        [Dist_dd_sort,Ind_sort_dd]=sort(Dist_dd);
        Ind_near_max=Ind_sort_dd(end);
        % 取上下界
        Dist_max_ud=Dist_dd(Ind_near_max);

        Dist_all=sqrt(sum(abs(repmat(Loc_dd,N,1)-Grid_2D).^2,2));
        [~,Ind_sort_dd_all]=sort(Dist_all);
        Num_neigh=5;
        region_near_dist=Grid_2D(Ind_sort_dd_all(2:Num_neigh),:);
        Dist_max_all=Dist_all(Ind_sort_dd_all(Num_neigh));% 取最近的6个网格
        if Dist_max_all>Dist_max_ud
            Loc_near_max=region_near_ud(Ind_near_max,:);
            Dist_max=Dist_max_all;
        else
            Loc_near_max=Grid_2D(Ind_sort_dd_all(Num_neigh),:);
            Dist_max=Dist_max_ud;
        end
        ind_near=union(ind_region_near,Ind_sort_dd_all(2:Num_neigh));
        region_near=Grid_2D(ind_near,:);
        if Dist_max<=1*min_interval % 因为希望后面加入的网格可以少于min_interval
            Ind_select_2D(dd)=0;
            continue;
        end
        % 划分被选中网格的搜索范围
        Inter_x=max(region_near(:,1))-min(region_near(:,1));
        Inter_y=max(region_near(:,2))-min(region_near(:,2));
        if abs(Inter_x)>=1.5*min_interval
            Loc_neigh_x=min(region_near(:,1))+Inter_x*([.2 .4 .6 .8]');
        else
            Loc_neigh_x=min(region_near(:,1))+Inter_x*([.25 .5 .75]);
        end
        if abs(Inter_y)>=1.5*min_interval
            Loc_neigh_y=min(region_near(:,2))+Inter_y*([.2 .4 .6 .8]');
        else
            Loc_neigh_y=min(region_near(:,2))+Inter_y*([.25 .5 .75]);
        end
        if ismember(Loc_dd(1),Loc_neigh_x)&&ismember(Loc_dd(2),Loc_neigh_y)
            Loc_neigh_x(Loc_neigh_x==Loc_dd(1))=[];
            Loc_neigh_y(Loc_neigh_y==Loc_dd(2))=[];
        end
        if isempty(Loc_neigh_x)||isempty(Loc_neigh_y)
            Ind_select_2D(dd)=0;            
            continue;
        end        
        % 寻找被选中网格点相邻网格点中能量最高的点，通过最小距离实现
        % Ind_neigh_dist=sort(Ind_sort_dd(1:8));
        Ind_neigh_dist=ind_near;
        Ind_neigh_dist(Ind_neigh_dist==idx_dd)=[];
        [~,Ind_gamma_max]=sort(gamma(Ind_neigh_dist),'descend');
        idx_del=[idx_dd;Ind_neigh_dist(Ind_gamma_max(1:num_neigh_del))];
        varphi_2_dd=zeros(block_size,1);
        q_dd=zeros(block_size,1);

        for bb=1:block_size
            seg_M=(bb-1)*M+1:bb*M;
            A_multi_bb=A_multi(:,:,bb);
            Sigma_del_dd(:,:,bb)=Sigma_y(:,:,bb)-A_multi_bb(:,idx_del)*diag(gamma(idx_del))*A_multi_bb(:,idx_del)';
            varphi_dd=abs(A_multi_bb(:,idx_dd)'*Sigma_del_dd(:,:,bb)*A_multi_bb(:,idx_dd));
            q_dd(bb)=abs(A_multi_bb(:,idx_dd)'*Sigma_del_dd(:,:,bb)*mean_z(seg_M))^2+varphi_dd;
            varphi_2_dd(bb)=varphi_dd^2;
        end
        % toc
        gamma_ref_dd=mean(q_dd./varphi_2_dd);
        if gamma_ref_dd<0
            warning(' off-grid gamma is negative\n');
        end
        % tic
        [grid_est,ML_value]= TypeTwo_ML_2D(gamma_ref_dd,Loc_neigh_x,Loc_neigh_y,Sigma_del_dd,mean_z,Para);
        grid_est_x=grid_est(1);grid_est_y=grid_est(2);
        % toc
        % tic
        % 插入1D网格，需要找到离当前网格最近的网格点
        grid_near_x_unorder=unique(region_near(:,1));
        grid_near_y_unorder=unique(region_near(:,2));
        idx_est_y=find(grid_near_y_unorder==Loc_dd(2));
        idx_est_x=find(grid_near_x_unorder==Loc_dd(1));
        if grid_est_y<Loc_dd(2)
            grid_y_add=Loc_dd(2)-(Loc_dd(2)-grid_near_y_unorder(idx_est_y-1))/2; % 增加坐标的网格
        else
            grid_y_add=Loc_dd(2)+(grid_near_y_unorder(idx_est_y+1)-Loc_dd(2))/2;
        end


        if grid_est_x<Loc_dd(1)
            grid_x_add=Loc_dd(1)-(Loc_dd(1)-grid_near_x_unorder(idx_est_x-1))/2; % 增加坐标的网格
        else
            grid_x_add=Loc_dd(1)+(grid_near_x_unorder(idx_est_x+1)-Loc_dd(1))/2;
        end
        Loc_add_dd=[grid_x_add,grid_y_add];
        % Loc_add_dd=grid_est;
        Lmse=sqrt(sum(abs(repmat(Loc_add_dd,size(ind_near,1),1)-region_near).^2,2));
        if min(Lmse)<min_interval
            % if PRINT
            %     disp(['Added grid is too dense ', num2str(Loc_add_dd)]);
            % end
            Ind_select_2D(dd)=0;            
            continue;
        end
            
        if ismember(Loc_add_dd,Grid_2D,'rows')
            if PRINT
                disp([' Repeated added grid: ', num2str(Loc_add_dd)]);
            end
            Ind_select_2D(dd)=0;
            continue;
        end
        if count_effect>1&&ismember(Loc_add_dd,Loc_add_cur,'rows')
            if PRINT
                disp([' Repeated added grid: ', num2str(Loc_add_dd)]);
            end
            Ind_select_2D(dd)=0;
            continue;
        end
        % if PRINT
        %     disp([' Selected grid:[',num2str(grid_x_unorder(idx_x_dd)),', ', num2str(grid_y_unorder(idx_y_dd)),']==',...
        %         ' New grid: [',num2str(grid_x_add),', ', num2str(grid_y_add),'].']);
        % end
        Loc_add_cur=[Loc_add_cur;Loc_add_dd];
        % toc
        count_effect=count_effect+1;

        % if norm(Loc_dd-Loc_tar')<=4*min_interval
        %     keyboard;
        % end
    end
    % tic
    %==================================估计出所有需要增加的网格后，更新1D的interval，2D的gamma，字典矩阵,网格，长度
    %%  update the grid, interval and prior variance
    % Ind_select=Ind_select+1;
    Loc_add=[Loc_add;Loc_add_cur];
    Ind_select_2D(Ind_select_2D==0)=[];
    if ~isempty(Loc_add_cur)
        gamma(Ind_select_2D)=gamma(Ind_select_2D)/2;
        gamma=[gamma;gamma(Ind_select_2D)];
        %---------------------------Update matrix------------------------------
        A_add=GenerateMtr(Loc_add_cur(:,1),Loc_add_cur(:,2),Para);
        [A_multi]=[A_multi,A_add];
        N=length(gamma);
        Grid_2D=[Grid_2D;Loc_add_cur];
        %------------------------------Discard some ineffective grid-------
        [~,Ind_dis]=sort(gamma);
        Ind_dis=Ind_dis(1:floor(length(gamma)*ratio_dis));% delelt the 20% of minimum  elements
        gamma(Ind_dis)=[];
        Grid_2D(Ind_dis,:)=[];
        A_multi(:,Ind_dis,:)=[];
        grid_x_unorder=unique(Grid_2D(:,1));
        grid_y_unorder=unique(Grid_2D(:,2));
        %-----------------------------Reorder ----------------------------
        num_calorder=1e4*Grid_2D(:,2)+1e-4*Grid_2D(:,1);
        [~,Ind_order]=sort(num_calorder);
        Grid_2D=Grid_2D(Ind_order,:);
        gamma=gamma(Ind_order);
        A_multi=A_multi(:,Ind_order,:);
    end

    % toc
    N=length(gamma);
    Sigma=zeros(N,N,block_size);
    Sigma_diag=zeros(N,block_size);
    Mean_x=zeros(N,block_size);
    [gamma_max,ind_max]=max(gamma);
    if PRINT
        disp([' *Grid Refine Iters: ', num2str(ii),...
            ' Max gamma: ', num2str(max(gamma)),...
            ' Grid: [', num2str(Grid_2D(ind_max,1)),',',num2str(Grid_2D(ind_max,2)) ']']);
    end
end
figure(201)
plot(1:N,gamma,'rx-');hold on
plot(ind_max,gamma_max,'bo');title('网格化更新');hold off;
grid on;
% gamma=1e-1*ones(N,1);
%==============================Refine x===================================
for mm=1:max_iters
    %===============EM procedure===========================================
    %-------------variational E step------------------------------------
    mean_x_old=mean_x;
    for tt=1:block_size
        Gamma=diag(gamma);
        Sigma_y(:,:,tt)=eye(M)*var_n+A_multi(:,:,tt)*Gamma*A_multi(:,:,tt)';
        if M<N
            Sigma_y_inv(:,:,tt)=inv(Sigma_y(:,:,tt));
            % PYP=A_multi(:,:,tt)'/Sigma_y(:,:,tt)*A_multi(:,:,tt);
            PYP_diag=zeros(N,1);
            for nn=1:N
                PYP_diag(nn)=A_multi(:,nn,tt)'*Sigma_y_inv(:,:,tt)*A_multi(:,nn,tt);
            end
            gpg=gamma.*PYP_diag.*gamma;
            Sigma_diag_tt=gamma-gpg;
            % Sigma(:,:,tt)=Sigma_tt;
            Sigma_diag(:,tt)=real(Sigma_diag_tt);
        else
            PVP_tt=A_multi(:,:,tt)'*A_multi(:,:,tt)/var_n;
            Sigma_tt=inv(diag(gamma.^-1)+PVP_tt);
            Sigma(:,:,tt)=Sigma_tt;
            Sigma_diag(:,tt)=real(diag(Sigma_tt));
        end

    end
    if bits==1
        for jj=1:max_iters_in
            mean_z_old=mean_z;
            for t=1:block_size
                seg_M=(t-1)*M+1:t*M;
                %seg_N=(t-1)*N+1:t*N;
                if M<N
                    Mean_x(:,t)=gamma.*(A_multi(:,:,t)'*Sigma_y_inv(:,:,t)*mean_z(seg_M));
                else
                    Mean_x(:,t)=Sigma(:,:,t)*A_multi(:,:,t)'*mean_z(seg_M)/var_n;
                end
                Prod(seg_M)=A_multi(:,:,t)*Mean_x(:,t);
            end
            z_temp=Prod-h;
            chi_r=y_r.*real(z_temp)*sqrt(2/var_n);
            chi_i=y_i.*imag(z_temp)*sqrt(2/var_n);

            mean_z=sqrt(var_n/2)*(y_r.*logcdf_1der(chi_r)+1j*y_i.*logcdf_1der(chi_i))+Prod;

            dmz=max(abs(mean_z_old-mean_z));
            if dmz<conver_thres
                break;
            end

        end
    else
        mean_z=y;
        for t=1:block_size
            seg_M=(t-1)*M+1:t*M;
            %seg_N=(t-1)*N+1:t*N;
            if M<N
                Mean_x(:,t)=gamma.*(A_multi(:,:,t)'*Sigma_y_inv(:,:,t)*mean_z(seg_M));
            else
                Mean_x(:,t)=Sigma(:,:,t)*A_multi(:,:,t)'*mean_z(seg_M)/var_n;
            end
        end
    end
    mean_x=reshape(Mean_x,[],1);
    % if length(mean_x_old)<length(mean_x)
    %     mean_x_old=[mean_x_old;zeros(length(mean_x)-)]
    % dmx=max(max(abs(mean_x_old-mean_x)));
    %     dmx_conv=norm(mean_x_old-mean_x)/norm(mean_x_old);
    %-------------variational M step------------------------------------
    % update prior noise
    gamma_old=gamma;
    secmom=abs(Mean_x).^2+Sigma_diag;
    gamma=mean(secmom,2);
    dmx=max(max(abs(gamma_old-gamma)));
    %===================Judge Convergence===================================
    if PRINT
        disp([' Iters: ', num2str(mm),...
            ' x change: ', num2str(dmx),...
            ' The max gamma value: ', num2str(max(gamma))]);
    end
    % x_report=[x_report,mean_x];
    dmx_report=[dmx_report,dmx];
    if dmx<conver_thres
        fprintf('The off-grid VBI method reach convergence at %d iterations\n',mm);
        break;
    end
end
% 精细化更新会在原有的gamma基础上进行优化，得到更平滑的结果
figure(202)
plot(1:N,gamma,'rx-');hold on
[gamma_max,ind_max]=max(gamma);
plot(ind_max,gamma_max,'bo');title('精细化更新结果');
hold off;
grid on;

[~,~,ind_peak]=FindPeak(gamma,1:N,K);


%% 网格化搜索+进一步更新幅度
    Sigma_del_kk=zeros(M,M,block_size);
    for kk=1:K
        idx_kk=ind_peak(kk);
        Loc_kk=Grid_2D(idx_kk,:);
        Dist_kk=sqrt(sum(abs(repmat(Loc_kk,N,1)-Grid_2D).^2,2));
        [Dist_kk_sort,Ind_temp]=sort(Dist_kk);
        Ind_neigh_kk=Ind_temp(Dist_kk_sort<=3*min_interval);
        region_near_kk=Grid_2D(Ind_neigh_kk,:);
        grid_search_x=linspace(min(region_near_kk(:,1)),max(region_near_kk(:,1)),20);
        grid_search_y=linspace(min(region_near_kk(:,2)),max(region_near_kk(:,2)),20);
        [~,Ind_sort]=sort(gamma(Ind_neigh_kk),'descend');
        idx_del=Ind_neigh_kk(Ind_sort(1:2));
        varphi_2_kk=zeros(block_size,1);
        q_kk=zeros(block_size,1);   
        for bb=1:block_size
            seg_M=(bb-1)*M+1:bb*M;
            A_multi_bb=A_multi(:,:,bb);
            Sigma_del_kk(:,:,bb)=Sigma_y(:,:,bb)-A_multi_bb(:,idx_del)*diag(gamma(idx_del))*A_multi_bb(:,idx_del)';
            varphi_kk=abs(A_multi_bb(:,idx_kk)'*Sigma_del_kk(:,:,bb)*A_multi_bb(:,idx_kk));
            q_kk(bb)=abs(A_multi_bb(:,idx_kk)'*Sigma_del_kk(:,:,bb)*mean_z(seg_M))^2+varphi_kk;
            varphi_2_kk(bb)=varphi_kk^2;
        end
        % toc
        gamma_ref_kk=mean(q_kk./varphi_2_kk);
        if gamma_ref_kk<0
            warning(' off-grid gamma is negative\n');
        end
        % tic
        [grid_est,ML_value]= TypeTwo_ML_2D(gamma_ref_kk,grid_search_x,grid_search_y,Sigma_del_kk,mean_z,Para);
        Grid_2D(idx_kk,:)=grid_est; 
    end
% % if K_known
% %     A_refine=GenerateMtr(Grid_2D(ind_peak,1),Grid_2D(ind_peak,2),Para);
% %     A_multi(:,ind_peak,:)=A_refine;
% %     gamma_refine=min(gamma)*ones(N,1);
% %     gamma_refine(ind_peak)=gamma(ind_peak);
% %     for mm=1:2e1
% %         %===============EM procedure===========================================
% %         %-------------variational E step------------------------------------
% %         mean_x_old=mean_x;
% %         for tt=1:block_size
% %             Gamma=diag(gamma_refine);
% %             Sigma_y(:,:,tt)=eye(M)*var_n+A_multi(:,:,tt)*Gamma*A_multi(:,:,tt)';
% %             if M<N
% %                 Sigma_y_inv(:,:,tt)=inv(Sigma_y(:,:,tt));
% %                 % PYP=A_multi(:,:,tt)'/Sigma_y(:,:,tt)*A_multi(:,:,tt);
% %                 PYP_diag=zeros(N,1);
% %                 for nn=1:N
% %                     PYP_diag(nn)=A_multi(:,nn,tt)'*Sigma_y_inv(:,:,tt)*A_multi(:,nn,tt);
% %                 end
% %                 gpg=gamma_refine.*PYP_diag.*gamma_refine;
% %                 Sigma_diag_tt=gamma_refine-gpg;
% %                 % Sigma(:,:,tt)=Sigma_tt;
% %                 Sigma_diag(:,tt)=real(Sigma_diag_tt);
% %             else
% %                 PVP_tt=A_multi(:,:,tt)'*A_multi(:,:,tt)/var_n;
% %                 Sigma_tt=inv(diag(gamma_refine.^-1)+PVP_tt);
% %                 Sigma(:,:,tt)=Sigma_tt;
% %                 Sigma_diag(:,tt)=real(diag(Sigma_tt));
% %             end
% % 
% %         end
% %         if bits==1
% %             for jj=1:max_iters_in
% %                 mean_z_old=mean_z;
% %                 for t=1:block_size
% %                     seg_M=(t-1)*M+1:t*M;
% %                     %seg_N=(t-1)*N+1:t*N;
% %                     if M<N
% %                         Mean_x(:,t)=gamma_refine.*(A_multi(:,:,t)'*Sigma_y_inv(:,:,t)*mean_z(seg_M));
% %                     else
% %                         Mean_x(:,t)=Sigma(:,:,t)*A_multi(:,:,t)'*mean_z(seg_M)/var_n;
% %                     end
% %                     Prod(seg_M)=A_multi(:,:,t)*Mean_x(:,t);
% %                 end
% %                 z_temp=Prod-h;
% %                 chi_r=y_r.*real(z_temp)*sqrt(2/var_n);
% %                 chi_i=y_i.*imag(z_temp)*sqrt(2/var_n);
% % 
% %                 mean_z=sqrt(var_n/2)*(y_r.*logcdf_1der(chi_r)+1j*y_i.*logcdf_1der(chi_i))+Prod;
% % 
% %                 dmz=max(abs(mean_z_old-mean_z));
% %                 if dmz<conver_thres
% %                     break;
% %                 end
% % 
% %             end
% %         else
% %             mean_z=y;
% %             for t=1:block_size
% %                 seg_M=(t-1)*M+1:t*M;
% %                 %seg_N=(t-1)*N+1:t*N;
% %                 if M<N
% %                     Mean_x(:,t)=gamma_refine.*(A_multi(:,:,t)'*Sigma_y_inv(:,:,t)*mean_z(seg_M));
% %                 else
% %                     Mean_x(:,t)=Sigma(:,:,t)*A_multi(:,:,t)'*mean_z(seg_M)/var_n;
% %                 end
% %             end
% %         end
% %         mean_x=reshape(Mean_x,[],1);
% %         % if length(mean_x_old)<length(mean_x)
% %         %     mean_x_old=[mean_x_old;zeros(length(mean_x)-)]
% %         % dmx=max(max(abs(mean_x_old-mean_x)));
% %         %     dmx_conv=norm(mean_x_old-mean_x)/norm(mean_x_old);
% %         %-------------variational M step------------------------------------
% %         % update prior noise
% %         gamma_refine_old=gamma_refine;
% %         secmom=abs(Mean_x).^2+Sigma_diag;
% %         gamma_refine=mean(secmom,2);
% %         dmx=max(max(abs(gamma_refine_old-gamma_refine)));
% %         %===================Judge Convergence===================================
% %         if PRINT
% %             disp([' Iters: ', num2str(mm),...
% %                 ' x change: ', num2str(dmx),...
% %                 ' The max gamma value: ', num2str(max(gamma_refine))]);
% %         end
% %         % x_report=[x_report,mean_x];
% %         dmx_report=[dmx_report,dmx];
% %         if dmx<conver_thres
% %             fprintf('The refine step reach convergence at %d iterations\n',mm);
% %             break;
% %         end
% %     end 
% %     gamma=gamma_refine;
% % end
Loc_est=Grid_2D(ind_peak,:);
x_est=Mean_x(ind_peak,:).';
Result_est.Loc=Loc_est';
Result_est.gamma=gamma;
Result_est.cov=Sigma;
Result_est.report=dmx_report;
Result_est.Grid=Grid_2D;
% Result_est.x_report=x_report;
% Result_est.bias=[b_x';b_y'];

figure(200);
plot(Loc_est(1),Loc_est(2),'rx',Loc_tar(1),Loc_tar(2),'bo');
axis([Loc_tar(1)-10,Loc_tar(1)+10,Loc_tar(2)-10,Loc_tar(2)+10]);
grid on;xlabel('X axe');ylabel('Y axe');
end

function [grid_est,ML_value]= TypeTwo_ML_2D(gamma_ref_dd,grid_neigh_x,grid_neigh_y,Sigma_del_dd,mean_z,Para)
[Ng]=length(grid_neigh_x);
[Mg]=length(grid_neigh_y);
[M,~,BS]=size(Sigma_del_dd);
Func_set=zeros(Mg,Ng);
for gm=1:Mg
    for gn=1:Ng
        A_multi_mn=GenerateMtr(grid_neigh_x(gn),grid_neigh_y(gm),Para);
        Func_val=0;
        for bb=1:BS
            seg_M=(bb-1)*M+1:bb*M;
            AS_bb=A_multi_mn(:,bb)'*Sigma_del_dd(:,:,bb);
            varphi_bb=abs(AS_bb*A_multi_mn(:,bb));
            q_bb=abs(AS_bb*mean_z(seg_M)).^2;
            Func_val=Func_val+(-log(1+gamma_ref_dd*varphi_bb)+q_bb/(varphi_bb+1/gamma_ref_dd));
        end
        Func_set(gm,gn)=Func_val;
    end
end
%
[Func_est_rowM,Ind_ml]=max(Func_set,[],2);
[ML_value,Ind_max_row]=max(Func_est_rowM);
grid_est=[grid_neigh_x(Ind_ml(1)),grid_neigh_y(Ind_max_row(1))];
end
