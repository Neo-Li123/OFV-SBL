% This code aims to evaluate the performance of OFVBI on localizing
% off-grid extended target
% Code date: 25/02/22.
clear;
clc;
curfold = pwd;
addpath([curfold,'\Func']);
% Model the extend-target in a bistatic prari of multi-static radar system
c=3e8; % The speed of light
SNRLen=5;  % The length of SNR vector
Num_MC=2e2;  % Monte Carlo trials number
SNRSet=linspace(0,20,SNRLen); % SNR range
% % % %============================================== The setting of transmitter
M=4;
Loc_tran=zeros(2,M);
% Loc_tran=[0,600,900,1800;600 0 1800 700];
Rad_tran=6000;
for m=1:M
    theta_tran=135/180*pi+2*pi/M*m;
    Loc_tran(1,m)=Rad_tran*cos(theta_tran);
    Loc_tran(2,m)=Rad_tran*sin(theta_tran);
end
% Loc_tran=[4000 3900 0 0;0 0 4000 3900];
fc = 5e9; % Carrier frequency
PRF =1/(3.5e-5); % PRF
Ratio=0.05;
tp=1/PRF*Ratio;
FreInterval=20e6;
% B =(M-1)*FreInterval; 
% fs =(M-1)*FreInterval; % Sampling frequency must be no less than bandwidth
% fs=40e6;
B=5e6;% bandwidth of baseband signal
fs=4*B;
Num_sam =floor(fs/PRF); % The number of Samples in each pulse
t=0:1/fs:(Num_sam-1)/fs;
Rect=rectpuls(t,Ratio*2*1/PRF).';

% St=randn(Num_sam,M)+1j*randn(Num_sam,M);
% St=St.*repmat(Rect,1,M);
% % Linear Frequency Modulated Signal
Fm_send=(0:M-1)*FreInterval;
K_st=B/tp; 
phi0=0;
St=zeros(Num_sam,M);
for mm=1:M
    St(:,mm)=Rect.*exp(1j*(pi*K_st*t.^2+phi0+2*pi*Fm_send(mm)*t)).';
    % St(:,mm)=Rect.*exp(1j*(pi*K_st*t)).';
end
% St=repmat(St,M);
% % Fm=fc+(0:M-1)*FreInterval;
% RangeReso=c/2/B;
% %================================================ The setting of  receiver
N=4; % The number of antenna array in receiver
Loc_rec=zeros(2,N);
Rad_rec=3000;
for n=1:N
%     theta_rec=180*pi/180+2*pi/N*n;
    theta_rec=0*pi/180+2*pi/N*n;
    Loc_rec(1,n)=Rad_rec*cos(theta_rec);
    Loc_rec(2,n)=Rad_rec*sin(theta_rec);
end
% Loc_rec=[4000,0;0,4000];
Num_lgrid_x=5;
Num_lgrid_y=5;
Grid_len=20;
Num_samTp=floor(tp*fs);
% % %====================================================The setting of Targets
block_num=2;
block_size=M*N;
RangeLoc_x=0+linspace(-(Num_lgrid_x-1)/2*Grid_len,(Num_lgrid_x-1)/2*Grid_len,Num_lgrid_x)';
RangeLoc_y=0+linspace(-(Num_lgrid_y-1)/2*Grid_len,(Num_lgrid_y-1)/2*Grid_len,Num_lgrid_y)';
% RangeLoc_x=(18+16)*Grid_len+linspace(-(Num_lgrid_x-1)/2*Grid_len,(Num_lgrid_x-1)/2*Grid_len,Num_lgrid_x)';
% RangeLoc_y=(29-16)*Grid_len+linspace(-(Num_lgrid_y-1)/2*Grid_len,(Num_lgrid_y-1)/2*Grid_len,Num_lgrid_y)';
[Delay_min,Delay_max,Indice_min,Indice_max]=DelayScope(Loc_tran,Loc_rec,RangeLoc_x,RangeLoc_y);
Num_samMax=floor(Delay_max*fs);
% Ind_null=20;
Num_samMin=ceil(Delay_min*fs);
IdeMtr=eye(Num_sam);
SamWindow=IdeMtr(Num_samMin+1:(Num_samMax+Num_samTp),:);
Num_samwin=Num_samMax+Num_samTp-Num_samMin;
Ind_samwin=Num_samMin+1:(Num_samMax+Num_samTp);
Ratio_CS=1;
Num_CSsamwin=floor(Num_samwin*Ratio_CS);
Ind_CS=randperm(Num_samwin,Num_CSsamwin);
Ind_CSsamwin=sort(Ind_samwin(Ind_CS));
% %==============================================Define the dictionary matrix
Q=floor(FreInterval*tp);
P=Num_lgrid_x*Num_lgrid_y;
DelayMtr=zeros(M*N,P);
DicMtr= zeros(N*M*Num_CSsamwin,P*M*N);
DM_der_x=zeros(N*M*Num_CSsamwin,P*M*N);
DM_der_y=zeros(N*M*Num_CSsamwin,P*M*N);
tau_derMtr=zeros(2,M*N,P);
Der_delay=zeros(Num_CSsamwin,M*N,P);
RadarPara=struct;
RadarPara.LocTran=Loc_tran;
RadarPara.LocRec=Loc_rec;
RadarPara.K_st=K_st;
RadarPara.FreqInter=FreInterval;
RadarPara.NumTran=M;
RadarPara.NumRec=N;
RadarPara.seq=t;
RadarPara.SamIndex=Ind_CSsamwin;
RadarPara.St=St;
RadarPara.Num_Grid_x=Num_lgrid_x;
RadarPara.Num_Grid_y=Num_lgrid_y;
RadarPara.fs=fs;
Grid=struct;
Grid.x=kron(ones(Num_lgrid_y,1),RangeLoc_x);
Grid.y=kron(RangeLoc_y,ones(Num_lgrid_x,1));
for ii = 1:P
    % ii=14;
    Ind_loc_y=floor((ii-1)/Num_lgrid_x)+1;
    Ind_loc_x=ii-(Ind_loc_y-1)*Num_lgrid_x;
    Loc_cur=[RangeLoc_x(Ind_loc_x),RangeLoc_y(Ind_loc_y)]';
    Dist_tran=repmat(Loc_cur,1,M)-Loc_tran;
    Dist_rec=repmat(Loc_cur,1,N)-Loc_rec;
    Dist_abs_tran=sqrt(sum(abs(Dist_tran).^2));
    Dist_abs_rec=sqrt(sum(abs(Dist_rec).^2));
    Delay_cur=reshape(Dist_abs_tran'+Dist_abs_rec,[],1)/c;
    DelayMtr(:,ii)=Delay_cur;
    DelayMtr_cur=zeros(Num_CSsamwin*M*N,M*N);
    DM_x_cur=zeros(Num_CSsamwin*M*N,M*N);
    DM_y_cur=zeros(Num_CSsamwin*M*N,M*N);
    for mm=1:M*N
        mm_tran=mm-floor((mm-1)/M)*M;
        mm_rec=1+ceil((mm-mm_tran-1)/M);
        FullData=circshift(St(:,mm_tran),floor(Delay_cur(mm)*fs),1);
        DelayMtr_cur((mm-1)*Num_CSsamwin+1:mm*Num_CSsamwin,mm)=FullData(Ind_CSsamwin);
        Fulldata_1d_tau=circshift(St(:,mm_tran)*1j.*(-2*pi*K_st*t'...
        -2*pi*(mm_tran-1)*FreInterval),floor(Delay_cur(mm)*fs),1);
        % Fulldata_1d_tau=circshift(St(:,mm_tran)*1j*(-pi*K_st),floor(Delay_cur(mm)*fs),1);        
        tau_der=(Dist_tran(:,mm_tran)/Dist_abs_tran(mm_tran)+Dist_rec(:,mm_rec)/Dist_abs_rec(mm_rec))/c;
        Der_delay(:,mm,ii)=tau_der(1)*1j*(-2*pi*K_st*t(Ind_CSsamwin)'-2*pi*(mm_tran-1)*FreInterval);
        tau_derMtr(:,mm,ii)=tau_der;
        DM_x_cur((mm-1)*Num_CSsamwin+1:mm*Num_CSsamwin,mm)=Fulldata_1d_tau(Ind_CSsamwin)*tau_der(1); 
        DM_y_cur((mm-1)*Num_CSsamwin+1:mm*Num_CSsamwin,mm)=Fulldata_1d_tau(Ind_CSsamwin)*tau_der(2); 
    end
    DicMtr(:,(ii-1)*M*N+1:ii*M*N) = DelayMtr_cur;
    DM_der_x(:,(ii-1)*M*N+1:ii*M*N) = DM_x_cur;
    DM_der_y(:,(ii-1)*M*N+1:ii*M*N) = DM_y_cur;
end
% DicMtr=DicMtr*diag(sqrt(sum(abs(DicMtr).^2)).^-1);
DM_der=[DM_der_x,DM_der_y];
%=================================================Generate the echo=======
K=1;
Spa_target_x=[4]; % 
Spa_target_y=[ 4 ]; % % Finally
% Spa_target_x=[ 6 7 5 6 7 8 6 7 ]; % 
% Spa_target_y=[ 5 5 6 6 6 6 7 7 ]; % % Finally
% Spa_target_x=[2 4]; % 
% Spa_target_y=[ 2 4]; % % Finally
Spa_target=sort((Spa_target_y-1)*Num_lgrid_x+Spa_target_x);
Loc_target=[RangeLoc_x(Spa_target_x)';RangeLoc_y(Spa_target_y)'];
Ind_unselect=setdiff(1:P,Spa_target);

PRINT=1 ;
bits=2;
for ii_snr=1:SNRLen
    tic
    SNR=SNRSet(ii_snr);
    SNR=20;
    fprintf('=======================Running %d ================\n',ii_snr);   
    for jj_mc=1:Num_MC
    % parfor jj_mc=1:Num_MC
             Loc_target=[RangeLoc_x(Spa_target_x)';RangeLoc_y(Spa_target_y)'];
             % Loc_bias_rand=[3.9455 3.9455;1.5735 1.5735];
             % Loc_bias=3*[ 1;1];
             % Loc_target=Loc_target+Loc_bias;
             Loc_bias_rand=8*(-0.5+rand(2,K)).*ones(2,K);
            Loc_target=Loc_target+Loc_bias_rand;
            x=zeros(M*N*P,1);
            prior_mean=0;prior_var=1e0;
            % prior_mean=1+1*1j;prior_var=1e0;
            % Randvec=prior_mean+sqrt(prior_var)*(randn(block_size,1)+1j*(randn(block_size,1)));
            % Randvec=ones(block_size,1);
            % Randvec=Randvec/norm(Randvec);
            % Amp=repmat(Randvec,1,K);
            Amp=prior_var/sqrt(2)*(randn(block_size,K)+1j*(randn(block_size,K)));
            Amp_norm=sqrt(sum(abs(Amp).^2,1));
            Ind_element=ComputeBlockInd(Spa_target,block_size);
            x(Ind_element)=reshape(Amp,block_size*K,1);
            b_x=zeros(M*N*P,1);
            b_x(Ind_element)=repmat(5*ones(block_size,1),K,1);
            b_y=zeros(M*M*P,1);
            

%             x_norm=x/norm(x);
            WF=zeros(Num_CSsamwin,K,M*N);
            Echo_pure= zeros(Num_CSsamwin,M*N,K);
            DelayMtr_tar=zeros(M*N,K);
            DelSampMtr=zeros(M*N,K);
            for nn=1:M*N
                for bb=1:K
                Loc_target_temp=Loc_target(:,bb);
%                 Loc_target_temp=1e-3*[1 ;1];  
                Dist_tran=Loc_target_temp-Loc_tran;
                Dist_rec=Loc_target_temp-Loc_rec;
                Dist_tran_tar=sqrt(sum(abs(Dist_tran).^2));
                Dist_rec_tar=sqrt(sum(abs(Dist_rec).^2));
                Dist_total=reshape(Dist_tran_tar'+Dist_rec_tar,[],1);
                Delay_target=Dist_total/c;
                DelayMtr_tar(:,bb)=Delay_target;
                DelSampMtr(:,bb)=floor(Delay_target*fs);
                Dist_diff_x=reshape(Dist_tran(1,:)'+Dist_rec(1,:),[],1);
                Dist_diff_y=reshape(Dist_tran(2,:)'+Dist_rec(2,:),[],1);
                
                nn_x=nn-floor((nn-1)/M)*M;
                nn_y=1+ceil((nn-nn_x-1)/M);
                Fulldata=circshift(St(:,nn_x),DelSampMtr(nn,bb));
                WF(:,bb,nn)= Fulldata(Ind_CSsamwin);
                Echo_pure(:,nn,bb)=Amp(nn,bb)*WF(:,bb,nn);
%                 
                    
                end
            end
            % x(Ind_element)=CalRCSUnres(x(Ind_element),DelSampMtr);
            % x_tar=x(Ind_element);
            WF=sum(WF,2);
            WF1=reshape(WF,Num_CSsamwin,M*N);
            Echo=sum(Echo_pure,3);
            Echo=reshape(Echo,Num_CSsamwin*M*N,1);
            %====================================================Mix Noise=============
%                         rng('default');
            sigma = sqrt(norm(Echo,'fro')^2/length(Echo(:))/(10^(SNR/10)));
            var_n=sigma^2;
            Noise = sigma/sqrt(2)*(randn(Num_CSsamwin*M*N,1)+1j*randn(Num_CSsamwin*M*N,1));
            Y_unq = Noise+Echo;
            Y_unq_r=real(Y_unq);
            Y_unq_i=imag(Y_unq);
            lambda=var_n;
            %====================================================Quantizer=============
             h=0*(randn(M*N*Num_CSsamwin,1)+1j*randn(M*N*Num_CSsamwin,1));
             Num_bit=6;
             % h =RandomThres(Y_unq,Num_bit,'uniform'); % 阈值设定对于幅度恢复很重要
            Z=sign(Y_unq_r-real(h))+1j*sign(Y_unq_i-imag(h));
            QuanOut_2=Quan_fewbits(Y_unq,2);Quan_bound_2=QuanOut_2.bound;        
            QuanOut_3=Quan_fewbits(Y_unq,3);Quan_bound_3=QuanOut_3.bound;
            QuanOut_4=Quan_fewbits(Y_unq,4);Quan_bound_4=QuanOut_4.bound;
            y_quan_2=QuanOut_2.data;Quan_len_2=QuanOut_2.len;
            y_quan_3=QuanOut_3.data;Quan_len_3=QuanOut_3.len;
            y_quan_4=QuanOut_4.data;Quan_len_4=QuanOut_4.len;    
            %% 1bBSBL
           %   iters_in=1e1;
           %  Result_1bB = My1bBSBL_V1(Z,DicMtr,h,lambda,block_size,iters_in,PRINT);
           %  x_1bB=Result_1bB.x; 
           %  % B=Result_1bB.B;
           %  dmx=Result_1bB.report;
           %  x_1bB_a=sum(abs(reshape(x_1bB,block_size,P).^2),1);
           %  [~,Ind_temp]=sort(x_1bB_a,'descend');
           %  Spa_1bB=sort(Ind_temp(1:K));
           %  succ_1bB(ii_snr,jj_mc) =length(intersect(Spa_target,Spa_1bB))==K;
           %   Spa_1bB_y=floor((Spa_1bB-1)/Num_lgrid_x)+1;
           % Spa_1bB_x=Spa_1bB-(Spa_1bB_y-1)*Num_lgrid_x;
           % Loc_1bB=[RangeLoc_x(Spa_1bB_x)';RangeLoc_y(Spa_1bB_y)'];
           %  Mse1_1bB(ii_snr,jj_mc)=mean(sum(abs(Loc_target-Loc_1bB).^2,1)); 
           %  Nmse_1bB(ii_snr, jj_mc)=norm(x_1bB-x)^2;             
            %==================================================OFVBI====
        
        tic
        bits=8;
        [x_ofv,ofv_report]=MyOFVBI_V2(Y_unq,DicMtr,h,Grid_len,bits,RadarPara,Grid,K,Loc_target,...
            'block_size',block_size,'print',PRINT,'max_iters',2e2,'max_iters_in',1e1,'var_n',lambda,'tol',1e-4);
        time_ofv_1(ii_snr,jj_mc)=toc;
       Loc_ofv=ofv_report.Loc;
        Mse_ofv(ii_snr,jj_mc)=mean(sqrt(sum(abs(Loc_target-Loc_ofv).^2,1)));
        Nmse_ofv(ii_snr, jj_mc)=norm(x_ofv-Amp,'fro')^2;
        Grid_est=ofv_report.Grid;
        % Lmse=sum(abs(repmat(Loc_target',size(Grid_est,1),1)-Grid_est).^2,2);
        % [~,ind]=sort(Lmse);
        % Loc_mindist=Grid_est(ind(1),:);
        % Mse_min=sqrt(Lmse(ind(1)));
        % Lmse_opt=sum(abs(repmat(Loc_mindist,size(Grid_est,1),1)-Grid_est).^2,2);
        
        if Mse_ofv(ii_snr,jj_mc)>4
            keyboard;
        end
    end
    toc
end
Amse_ofv=mean(Mse_ofv,2);
Anmse_ofv=mean(Nmse_ofv,2);
  Succ_ofv=mean(suc_ofv,2);
  Succ_ofv_1=mean(suc_ofv_1,2);
Succ_mm_2=mean(suc_mm_2,2);
Succ_mm_3=mean(suc_mm_3,2);
Succ_mm_4=mean(suc_mm_4,2);
Succ_MM=mean(suc_MM,2);

% Plot
figure;
% plot(SNRSet,(Amse_ofv),'rx-','LineWidth',2,'MarkerSize',8);
plot(SNRSet,(Anmse_ofv),'rx-','LineWidth',2,'MarkerSize',8);
xlabel('SNR (dB)');
% ylabel('MSE (m)');
ylabel('MSE of amplitude ');
grid on

% Save Data
c_time =clock;
filename = ['CRB-AllMethod-Offgrid-VarKnown-K=8-SNR-[-20,10]-', date,'_',num2str(c_time(4)),'_',num2str(c_time(5)),'.mat'];
save(filename);