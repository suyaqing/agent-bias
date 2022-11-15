function MDP = DEM_MDP_agent_bias_var1(sen)
% single-level model from semantic to lemma level. The semantic level
% contains semantic, morphological, and syntactical factors. The generative
% model and its inversion are defined in a formal way
%__________________________________________________________________________
%
%
%--------------------------------------------------------------------------
rng('default')
% close all
clear
% sen = 1;
% global ichunk
% ichunk = 0;
prediction = 1;
d = load('Knowledge_AB.mat');
dict = d.dict;
% dict = dict(1:20);
% clear d;
sen = [1 20 4 10];
alist = {'boy', 'girl', 'policeman', 'dog', 'car', 'fish', 'book', 'tree'};
rlist = {'eat', 'chase', 'drive', 'hit', 'throw'};
plist = alist;
morpha = {'singluar', 'plural'};
morphp = {'singluar', 'plural'};
syn = {'main', 'subject', 'object', 'verb'};
wlist = {dict.Word};

% level two: association--possible combinations and their probability
%==========================================================================

D{1} = [1 1 1 1 .5 .5 .5 .5]'; % prior preferences for agent roles
D{2} = [.5 .5 .5 .5 1 1 1 1]'; % prior preferences for patient roles
% D{1} = [1 1 1 1 1 1 1 1]'; % prior preferences for agent roles
% D{2} = [1 1 1 1 1 1 1 1]'; % prior preferences for patient roles
D{3} = [1 1 1 1 1]'; % prior preferences for actions
D{4} = [0.5 0.5]'; % prior preference for morphology, singular or plural agent
D{5} = [0.5 0.5]'; % prior preference for morphology, singular or plural patient
% D{6} = D{4}; % prior preference for morphology, singular or plural verb
% D{6} = [1 0 0 0 0]'; % prior preference for the starting component
 
% probabilistic mapping from hidden states to outcomes: A
%--------------------------------------------------------------------------
Nf    = numel(D);
for f = 1:Nf
    Ns(f) = numel(D{f});
    D{f} = spm_norm_exp(D{f});
end

na = length(alist);
np = length(plist);
nr = length(rlist);
nma = length(morpha);
nmp = length(morphp);

    
Nw = length(dict);
A = cell(1, 4);
A{1} = zeros(Nw, 1); A{1}(1) = 1;
A{2} = zeros(Nw, na, nma);
A{3} = zeros(Nw, np, nmp);
A{4} = zeros(Nw, nr, nma);
m1 = {dict(:).Meaning1};

for ns = 1:na
    sem = alist{ns};
    idx_a = find(strcmp(m1, sem));  
    A{2}(idx_a, ns, 1) = 1; % singlular form
    A{2}(idx_a+13, ns, 2) = 1; % plural form
end

for ns = 1:np
    sem = plist{ns};
    idx_p = find(strcmp(m1, sem));  
    A{3}(idx_p, ns, 1) = 1; % singlular form
    A{3}(idx_p+13, ns, 2) = 1; % plural form
end

for ns = 1:nr
    sem = rlist{ns};
    idx_r = find(strcmp(m1, sem));
    A{4}(idx_r, ns, 2) = 1;
    A{4}(idx_r+13, ns, 1) = 1;
end


for ksyn = 1:4
    A{ksyn} = spm_norm_exp(A{ksyn}, 2);
end
                  
Z = cell(1, 4);
Z{1}(:, 1) = [1 0 0 0]'; Z{1}(:, 2) = [1 0 0 0]';
Z{2}(:, 1) = [0 1 0 0]'; Z{2}(:, 2) = [0 0 1 0]';
Z{3}(:, 1) = [0 0 1 0]'; Z{3}(:, 2) = [0 1 0 0]';
Z{4}(:, 1) = [0 0 0 1]'; Z{4}(:, 2) = [0 0 0 1]';


bs = [1 2]'; % bias for subject first: object first order
D{6} = zeros(4, 4); % # of possible states x # of epochs
for tau = 1:4
    Z{tau} = spm_norm_exp(Z{tau});
    D{6}(:, tau) = spm_norm_exp(Z{tau}*bs);
end
%--------------------------------------------------------------------
 
mdp.T = 4;                      % number of moves
mdp.stepsyn = 4;
mdp.steps = [4 4 4]; %[a p r]mdp.A = A; 
mdp.stepm = [4 4];
mdp.A = A;
mdp.D = D;                      % prior over initial states
mdp.Z = Z;
% the input can be defined as indexes of outcome list
mdp.o = sen;
% mdp.s = [1 2 2 2 1 1];

mdp.label.name{1}   = alist;
mdp.label.name{2}   = plist;
mdp.label.name{3}   = rlist;
mdp.label.name{4}   = morpha;
mdp.label.name{5}   = morphp;
% mdp.label.name{6}   = morphp;
mdp.label.name{6} = syn;
mdp.label.factor   = {'Agent', 'Patient', 'Relation', 'Agent Count', 'Patient Count', 'Syntax'};
mdp.label.outcome{1} = wlist(1:end);
% mdp         = spm_MDP_check(mdp);
%%
% illustrate a single trial
%==========================================================================
% prediction = 1;
% OPTIONS.pred = prediction;
MDP  = spm_MDP_VB_X_agent_bias(mdp);
% if nargin
%     return;
% end
%%
% spm_figure('GetWin','Figure 1'); clf
% spm_MDP_VB_ERP(MDP,[]);
% 
% spm_figure('GetWin','Figure 1'); clf
% spm_MDP_VB_LFP(MDP,[],1); 
sentence = [dict(sen(1)).Word ' ' dict(sen(2)).Word ' ' dict(sen(3)).Word ' ' dict(sen(4)).Word '.'];
spm_MDP_VB_ERP_single_AB_formal(MDP, sentence);

% spm_figure('GetWin','Figure 2'); clf
% spm_MDP_VB_trial(MDP);
% 
% spm_figure('GetWin','Figure 3'); clf
% spm_MDP_VB_trial_AB(MDP);
% spm_MDP_VB_ERP_ALL_hybrid(MDP)

% figure;
% spm_MDP_VB_ERP_YS(MDP.mdp(4).mdp, 2)

% spm_figure('GetWin','Figure 2'); clf
% spm_MDP_VB_LFP(MDP.mdp(4).mdp.mdp,[], 1); 
% 
