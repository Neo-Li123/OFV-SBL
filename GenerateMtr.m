% generate dictionary matrix applied in a multi-task sparse model for
% distributed MIMO radars
% Code date: 2025/02/25
% V1:2025/03/03:delete partial derivative matrix
function [A]=GenerateMtr(grid_x,grid_y,Para)
% Input:
%  grid_x: grid set of x axe
%  grid_y: grid set of y axe
%  Para: parameter of radar systems
% Output:
%  A: dictionary matrix
%  D_x: first partial derivative of A w.r.t x axe
%  D_y: first partial derivative of A w.r.t y axe
c=3e8;
P=length(grid_x);
Loc_tran=Para.LocTran;
Loc_rec=Para.LocRec;
K_st=Para.K_st;
FreInterval=Para.FreqInter;
M=Para.NumTran;
N=Para.NumRec;
t=Para.seq;
St=Para.St;
fs=Para.fs;
% P_x=Para.Num_Grid_x;
% P_x=length(grid_x);
% P_y=length(grid_y);
% P=P_x*P_y;
Num_task=M*N;
Ind_CSsamwin=Para.SamIndex;
N_w=length(Ind_CSsamwin);
A=zeros(N_w,P,Num_task);
% D_x=zeros(N_w,P,Num_task);
% D_y=zeros(N_w,P,Num_task);

for ii = 1:P
    Loc_cur=[grid_x(ii),grid_y(ii)]';
    Dist_tran=repmat(Loc_cur,1,M)-Loc_tran;
    Dist_rec=repmat(Loc_cur,1,N)-Loc_rec;
    Dist_abs_tran=sqrt(sum(abs(Dist_tran).^2));
    Dist_abs_rec=sqrt(sum(abs(Dist_rec).^2));
    Delay_cur=reshape(Dist_abs_tran'+Dist_abs_rec,[],1)/c;
    for mm=1:Num_task
        mm_tran=mm-floor((mm-1)/M)*M;
        % mm_rec=1+ceil((mm-mm_tran-1)/M);
        mm_rec=1+floor((mm-mm_tran)/M);
        FullData=circshift(St(:,mm_tran),floor(Delay_cur(mm)*fs),1);
        A(:,ii,mm)=FullData(Ind_CSsamwin);
        % Fulldata_1d_tau=circshift(St(:,mm_tran)*1j.*(-2*pi*K_st*t'...
        % -2*pi*(mm_tran-1)*FreInterval),floor(Delay_cur(mm)*fs),1);     
        % tau_der=(Dist_tran(:,mm_tran)/Dist_abs_tran(mm_tran)+Dist_rec(:,mm_rec)/Dist_abs_rec(mm_rec))/c;
        % D_x(:,ii,mm)=Fulldata_1d_tau(Ind_CSsamwin)*tau_der(1); 
        % D_y(:,ii,mm)=Fulldata_1d_tau(Ind_CSsamwin)*tau_der(2); 
    end
end
end