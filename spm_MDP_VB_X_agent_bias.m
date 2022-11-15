function [MDP] = spm_MDP_VB_X_agent_bias(MDP)
% active inference and learning using variational message passing
% FORMAT [MDP] = spm_MDP_VB_X(MDP,OPTIONS)

% global ichunk

% set up and preliminaries
%==========================================================================

% defaults
%--------------------------------------------------------------------------

try, stepsyn   = MDP.stepsyn;   catch, stepsyn   = 8;    end % update time constant
try, steps   = MDP.steps;   catch, steps   = 16*ones(1,3); end 
try, stepm   = MDP.stepm;   catch, stepm   = 16*ones(1,2); end
try, erp   = MDP.erp;   catch, erp   = 2;    end % update reset


T = MDP.T;
D = MDP.D;
% L = MDP.L;
A = MDP.A;
Z = MDP.Z;
% Nc = length(D{1});
% Nt = length(D{2});
Nw = size(A{1}, 1);

W = cell(1, T);
W0 = cell(1, T);
% initialise model-specific variables
%==========================================================================
Ni    = 16;                                % number of VB iterations

% initialize x, X, nx
% Xc = zeros(Nc, T);
% Xc(:, 1) = D{1};
% xc = D{1};
% % xc = zeros(Nc, 1);
% nxc = zeros(Ni, Nc, T);
% 
% Xt = zeros(Nt, T);
% Xt(:, 1) = D{2};
% xt = D{2};
% % xt = zeros(Nt, 1);
% nxt = zeros(Ni, Nt, T);

Xs = cell(1, 3); xs = Xs; nxs = Xs;
for ks = 1:3
    Ns = size(A{ks+1}, 2);
    Xs{ks} = zeros(Ns, T);
    Xs{ks}(:, 1) = D{ks};
    xs{ks} = D{ks};
%     xs{ks} = zeros(Ns, 1);
    nxs{ks} = zeros(Ni, Ns, T);
end

% morphology factor
Xm = cell(1, 2); xm = Xm; nxm = Xm;
for km = 1:2
    Nm = size(A{km+1}, 3);
    Xm{km} = zeros(Nm, T);
    Xm{km}(:, 1) = D{km+3};
    xm{km} = D{km+3};
%     xs{ks} = zeros(Ns, 1);
    nxm{km} = zeros(Ni, Nm, T);
end

Nsyn = size(D{end}, 1);
Xsyn = cell(1, T); xsyn = Xsyn; nxsyn = Xsyn;
for tau = 1:T
    
    Xsyn{tau} = zeros(Nsyn, 1);
%     Xsyn{tau}(:, 1) = D{end}(:, tau);
    xsyn{tau} = D{end}(:, tau);
    nxsyn{tau} = zeros(Ni, Nsyn);
end

% xc(:) = 1/length(xc);
% xt(:) = 1/length(xt);
for ks = 1:3
    xs{ks}(:) = 1/length(xs{ks});
end
for km = 1:2
    xm{km}(:) = 1/length(xm{km});
end
for tau = 1:4
    xsyn{tau}(:) = 1/length(xsyn{tau});
end
F = zeros(Ni, T);


% belief updating over successive time points
%==========================================================================
for t = 1:T
    
    % calculate word prediction
    %======================================================================
    if t==1
        W0{1} = A{1};
    else
        W0{t} = zeros(Nw, 1);
%         % do we need a higher-level factor in order to update prediction?
%         % can just use the updated posterior
% %         syn = Z{t}*Xt(:, t);
%         syn = Xsyn{t};
%         W0{t} = W0{t}+A{1}*syn(1);
%         for ksyn = 2:4
%             for km = 1:2
%                 mm = Xm{ksyn-1}(km, t)*squeeze(spm_log(A{ksyn}(:, :, km)));
%                 W0{t} = W0{t} + syn(ksyn)*mm*Xs{ksyn-1}(:, t);
%             end
%         end
% %             W0{t} = W0{t}+syn(ksyn)*A{ksyn}*Xs{ksyn-1}(:, t);
    end

    
%     W{t} = MDP.mdp(t).X{1}(:, 1);
% get the lemma input directly
    wo = zeros(size(W0{t}));
    wo(MDP.o(t)) = 1;
%     wo = spm_softmax(wo);
    W{t} = wo;
    
               
    for i = 1:Ni
        
        
        % Variational updates (skip to t = T in HMM mode)
        %==================================================================
            
            % processing time and reset
            %--------------------------------------------------------------
            tstart = tic;        
            % Variational updates (hidden states) 
            %==============================================================
            
            % syntax
            for tau = 1:t
                v0 = spm_log(xsyn{tau});
                BU = zeros(length(v0), 1);
                BU(1) = W{tau}'*spm_log(A{1});
                for ksyn = 2:3 % subject, object
                    for km = 1:2 % sum across all possible morphologies
                        AA = xm{ksyn-1}(km)*squeeze(spm_log(A{ksyn}(:, :, km)));
                        BU(ksyn) = BU(ksyn) + W{tau}'*(AA*xs{ksyn-1});
                    end
%                     BU(ksyn) = W{tau}'*(spm_log(A{ksyn})*xs{ksyn-1});
                end
                % verb agrees with the agent/subject
                for km = 1:2 % sum across all possible morphologies
                    AA = xm{1}(km)*squeeze(spm_log(A{4}(:, :, km)));
                    BU(4) = BU(4) + W{tau}'*(AA*xs{3});
                end
%                 TD = spm_log(Z{tau})*xt;
                dFdx = v0 - BU - spm_log(D{6}(:, tau));
                dFdx = dFdx - mean(dFdx);
                sxsyn{tau} = spm_softmax(v0 - dFdx/stepsyn);
%                 F(i, t) = F(i, t) + sxsyn{tau}'*(spm_log(sxsyn{tau}) - spm_log(Z{tau})*sxt);
            end
            
            % semantic
            for ks = 1:3
                v0 = spm_log(xs{ks});
                BU = zeros(length(v0), 1);
%                 TD = BU;
                ww = zeros(Nw, 1);
                if t>1
                    for tau = 2:t
                        ww = ww + W{tau}*xsyn{tau}(ks+1);
                    end
                end
                if ks<3
                    for km = 1:2 % sum across all possible morphologies
                        AA = xm{ks}(km)*squeeze(spm_log(A{ks+1}(:, :, km)));
                        BU = BU + AA'*ww;
                    end
                else % action agrees with agent count
                    for km = 1:2 % sum across all possible morphologies
                        AA = xm{1}(km)*squeeze(spm_log(A{ks+1}(:, :, km)));
                        BU = BU + AA'*ww;
                    end
                end
%                 BU = spm_log(A{ks+1}')*ww;
%                 for kst = 1:2
%                     ll = xt(kst)*squeeze(spm_log(L{ks}(:, :, kst)));
%                     TD = TD + ll*xc;
%                 end
                dFdx = v0 - BU - spm_log(D{ks});
                dFdx = dFdx - mean(dFdx);
                sxs{ks} = spm_softmax(v0 - dFdx/steps(ks));
%                 F(i, t) = F(i, t) + sxs{ks}'*spm_log(sxs{ks});
%                 for kst = 1:2
%                     ll = sxt(kst)*squeeze(spm_log(L{ks}(:, :, kst)));
%                     F(i, t) = F(i, t) - sxs{ks}'*(ll*sxc);
%                 end
            end
            
             % morphology
            for km = 1:2
                v0 = spm_log(xm{km});
                BU = zeros(length(v0), 1);
%                 TD = BU;
                ww = zeros(Nw, 1);
                if t>1                
                    for tau = 2:t
                        ww = ww + W{tau}*xsyn{tau}(km+1);
                        if km == 1
                            ww = ww + W{tau}*xsyn{tau}(4);
                        end
                    end   
                end
                for ks = 1:length(xs{km}) % agent and patient
                    AA = xs{km}(ks)*squeeze(spm_log(A{km+1}(:, ks, :)));
                    BU = BU + AA'*ww;
                end
                if km == 1 % action agrees with agent
                    for ks = 1:length(xs{3}) 
                        AA = xs{3}(ks)*squeeze(spm_log(A{4}(:, ks, :)));
                        BU = BU + AA'*ww;
                    end
                end
                dFdx = v0 - BU - spm_log(D{km+3});
                dFdx = dFdx - mean(dFdx);
                sxm{km} = spm_softmax(v0 - dFdx/stepm(km));
            end
            
            
            for ks = 1:3
                xs{ks} = sxs{ks};
                Xs{ks}(:, t) = sxs{ks};
                nxs{ks}(i, :, t) = sxs{ks};
                if t<T
                    Xs{ks}(:, t+1) = Xs{ks}(:, t);
                end
            end
            
            for km = 1:2
                xm{km} = sxm{km};
                Xm{km}(:, t) = sxm{km};
                nxm{km}(i, :, t) = sxm{km};
                if t<T
                    Xm{km}(:, t+1) = Xm{km}(:, t);
                end
            end
            
            for tau = 1:t
                xsyn{tau} = sxsyn{tau};
                Xsyn{tau}(:) = sxsyn{tau};
                nxsyn{tau}(i, :) = sxsyn{tau};
%                 if t<T % here needs a transition matrix
%                     Xsyn{tau} = sxsyn{tau};
%                 end
            end
            % Free energy
            %--------------------------------------------------------------

%             for tau = 1:t
%                 F(i, t) = F(i, t) - xsyn{tau}(1)*W{tau}'*spm_log(A{1});
%                 for ksyn = 2:5
%                     F(i, t) = F(i, t) - xsyn{tau}(ksyn)*W{tau}'*(spm_log(A{ksyn})*xs{ksyn-1});
%                 end
%             end
           
    end
    
%     xc(:) = 1/length(xc);
%     xt(:) = 1/length(xt);
    for ks = 1:3
        xs{ks}(:) = 1/length(xs{ks});
    end
    for km = 1:2
        xm{km}(:) = 1/length(xm{km});
    end
    for tau = 1:4
        xsyn{tau}(:) = 1/length(xsyn{tau});
    end
   
            
            
            
    % processing (i.e., reaction) time
    %--------------------------------------------------------------
    rt(t)      = toc(tstart);


end % end of loop over time


    
    % assemble results and place in NDP structure
    %----------------------------------------------------------------------

MDP.Xs = Xs;
MDP.nxs = nxs;
MDP.Xm = Xm;
MDP.nxm = nxm;
MDP.Xsyn = Xsyn;
MDP.nxsyn = nxsyn;
MDP.W = W;
MDP.W0 = W0;
MDP.F = F;

MDP.rt = rt;        % simulated reaction time (seconds)
    



% auxillary functions
%==========================================================================

function A  = spm_log(A)
% log of numeric array plus a small constant
%--------------------------------------------------------------------------
A  = log(A + 1e-16);

function A  = spm_norm(A, mode)
% normalisation of a probability transition matrix (columns)
%--------------------------------------------------------------------------
A           = bsxfun(@rdivide,A,sum(A,1));
if nargin<2 || mode==1
    A(isnan(A)) = 1/size(A,1);
else
    A(isnan(A)) = 0;
end

function A  = spm_wnorm(A)
% summation of a probability transition matrix (columns)
%--------------------------------------------------------------------------
A   = A + 1e-16;
A   = bsxfun(@minus,1./sum(A,1),1./A)/2;

function sub = spm_ind2sub(siz,ndx)
% subscripts from linear index
%--------------------------------------------------------------------------
n = numel(siz);
k = [1 cumprod(siz(1:end-1))];
for i = n:-1:1,
    vi       = rem(ndx - 1,k(i)) + 1;
    vj       = (ndx - vi)/k(i) + 1;
    sub(i,1) = vj;
    ndx      = vi;
end

return



function [M,MDP] = spm_MDP_get_M(MDP,T,Ng)
% FORMAT [M,MDP] = spm_MDP_get_M(MDP,T,Ng)
% returns an update matrix for multiple models
% MDP(m) - structure array of m MPDs
% T      - number of trials or updates
% Ng(m)  - number of output modalities for m-th MDP
%
% M      - update matrix for multiple models
% MDP(m) - structure array of m MPDs
%
% In some applications, the outcomes are generated by a particular model
% (to maximise free energy, based upon the posterior predictive density).
% The generating model is specified in the matrix MDP(m).n, with a row for
% each outcome modality, such that each row lists the index of the model
% responsible for generating outcomes.
%__________________________________________________________________________

% check for VOX and ensure the agent generates outcomes when speaking
%--------------------------------------------------------------------------
if numel(MDP) == 1
    if isfield(MDP,'MDP')
        if isfield(MDP.MDP,'VOX')
            MDP.n = [MDP.MDP.VOX] == 1;
        end
    end
end
    
for m = 1:size(MDP,1)
    
    % check size of outcome generating agent, as specified by MDP(m).n
    %----------------------------------------------------------------------
    if ~isfield(MDP(m),'n')
        MDP(m).n = zeros(Ng(m),T);
    end
    if size(MDP(m).n,1) < Ng(m)
        MDP(m).n = repmat(MDP(m).n(1,:),Ng(m),1);
    end
    if size(MDP(m).n,1) < T
        MDP(m).n = repmat(MDP(m).n(:,1),1,T);
    end
    
    % mode of generating model (most frequent over outcome modalities)
    %----------------------------------------------------------------------
    n(m,:) = mode(MDP(m).n.*(MDP(m).n > 0),1);
    
end

% reorder list of model indices for each update
%--------------------------------------------------------------------------
n     = mode(n,1);
for t = 1:T
    if n(t) > 0
        M(t,:) = circshift((1:size(MDP,1)),[0 (1 - n(t))]);
    else
        M(t,:) = 1;
    end
end


return

function MDP = spm_MDP_update(MDP,OUT)
% FORMAT MDP = spm_MDP_update(MDP,OUT)
% moves Dirichlet parameters from OUT to MDP
% MDP - structure array (new)
% OUT - structure array (old)
%__________________________________________________________________________

% check for concentration parameters at this level
%--------------------------------------------------------------------------
try,  MDP.a = OUT.a; end
try,  MDP.b = OUT.b; end
try,  MDP.c = OUT.c; end
try,  MDP.d = OUT.d; end
try,  MDP.e = OUT.e; end

% check for concentration parameters at nested levels
%--------------------------------------------------------------------------
try,  MDP.MDP(1).a = OUT.mdp(end).a; end
try,  MDP.MDP(1).b = OUT.mdp(end).b; end
try,  MDP.MDP(1).c = OUT.mdp(end).c; end
try,  MDP.MDP(1).d = OUT.mdp(end).d; end
try,  MDP.MDP(1).e = OUT.mdp(end).e; end

return




