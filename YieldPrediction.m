%%
clc;
close;
clear all;

%% Data Batch Collection
TifFilePath='G:\CropYield\01_Data\05_TifGrid\';
TifFileNames=dir(sprintf('%s/*.tif',TifFilePath));
for TifFileNum=1:length(TifFileNames)
    TifFileTemp=geotiffread([TifFilePath,TifFileNames(TifFileNum).name,]);
    eval([TifFileNames(TifFileNum).name(4:end-4),'Matrix=TifFileTemp;']);
end

%% NoData Values Batch Removal
for TifFileNum=1:length(TifFileNames)
    eval([TifFileNames(TifFileNum).name(4:end-4),'Matrix(all(',...
        TifFileNames(TifFileNum).name(4:end-4),'Matrix==-99,2),:)=[];']);
    eval([TifFileNames(TifFileNum).name(4:end-4),'Matrix(:,all(',...
        TifFileNames(TifFileNum).name(4:end-4),'Matrix==-99,1))=[];']);    
end

StandardColumnNum=size(MaizeYieldMatrix,2);
for TifFileNum=1:length(TifFileNames)
    eval(['TifFileTemp=',TifFileNames(TifFileNum).name(4:end-4),'Matrix;']);
    while size(TifFileTemp,2)>StandardColumnNum
        TifFileTemp(:,end)=[];
    end
    eval([TifFileNames(TifFileNum).name(4:end-4),'Matrix=TifFileTemp;']);
end

%% Matrix Batch Transformation
StandardPixNum=length(MaizeYieldMatrix(:));
for TifFileNum=1:length(TifFileNames)
    eval([TifFileNames(TifFileNum).name(4:end-4),'=reshape(',...
        TifFileNames(TifFileNum).name(4:end-4),'Matrix,StandardPixNum,1);']);
end

%% Format Batch Conversion
for TifFileNum=1:length(TifFileNames)
    eval(['TifFileTemp=',TifFileNames(TifFileNum).name(4:end-4),';']);
    if ~(strcmp(class(TifFileTemp),'single'))
        TifFileTemp=single(TifFileTemp);
    end
    eval([TifFileNames(TifFileNum).name(4:end-4),'=TifFileTemp;']);
end

%% Invalid Values Removal
InputOutput=[];
for TifFileNum=1:length(TifFileNames)
    if ~(strcmp(TifFileNames(TifFileNum).name(4:end-4),'MaizeArea') | ...
            strcmp(TifFileNames(TifFileNum).name(4:end-4),'MaizeYield'))
        eval(['InputOutput=[InputOutput,',TifFileNames(TifFileNum).name(4:end-4),'];']);
    end
end
InputOutput=[InputOutput,MaizeArea,MaizeYield];

InputOutput(all(InputOutput==-99,2),:)=[];
AreaPercent=0.40;
for ColumnNum=1:size(InputOutput,2)
    if ColumnNum~=size(InputOutput,2)-1
        UselessRow=InputOutput(:,ColumnNum)==-99 | InputOutput(:,ColumnNum)==0;
        InputOutput(UselessRow,:)=[];
    else
        UselessRow=InputOutput(:,ColumnNum)==-99 | InputOutput(:,ColumnNum)<7461.4841*AreaPercent;
        InputOutput(UselessRow,:)=[];
    end
end

%% Input and Output Division
Input=InputOutput(:,(1:end-2));
Output=InputOutput(:,end);

%% Number of Leaves and Trees Optimization
for RFOptimizationNum=1:20
    
RFLeaf=[5,10,20,50,100,200,500];
col='rgbcmyk';
figure('Name','RF Leaves and Trees');
for i=1:length(RFLeaf)
    RFModel=TreeBagger(2000,Input,Output,'Method','R','OOBPrediction','On','MinLeafSize',RFLeaf(i));
    plot(oobError(RFModel),col(i));
    hold on
end
xlabel('Number of Grown Trees');
ylabel('Mean Squared Error') ;
LeafTreelgd=legend({'5' '10' '20' '50' '100' '200' '500'},'Location','NorthEast');
title(LeafTreelgd,'Number of Leaves');
hold off;

disp(RFOptimizationNum);
end

%% Notification
% Set breakpoints here.

%% Cycle Preparation
RFScheduleBar=waitbar(0,'Random Forest is Solving...');
RFRMSEMatrix=[];
RFrAllMatrix=[];
RFRunNumSet=50000;
for RFCycleRun=1:RFRunNumSet

%% Training Set and Test Set Division
RandomNumber=(randperm(length(Output),floor(length(Output)*0.2)))';
TrainYield=Output;
TestYield=zeros(length(RandomNumber),1);
TrainVARI=Input;
TestVARI=zeros(length(RandomNumber),size(TrainVARI,2));
for i=1:length(RandomNumber)
    m=RandomNumber(i,1);
    TestYield(i,1)=TrainYield(m,1);
    TestVARI(i,:)=TrainVARI(m,:);
    TrainYield(m,1)=0;
    TrainVARI(m,:)=0;
end
TrainYield(all(TrainYield==0,2),:)=[];
TrainVARI(all(TrainVARI==0,2),:)=[];

%% RF
nTree=150;
nLeaf=5;
RFModel=TreeBagger(nTree,TrainVARI,TrainYield,...
    'Method','regression','OOBPredictorImportance','on', 'MinLeafSize',nLeaf);
[RFPredictYield,RFPredictConfidenceInterval]=predict(RFModel,TestVARI);
% PredictBC107=cellfun(@str2num,PredictBC107(1:end));

%% Accuracy of RF
RFRMSE=sqrt(sum(sum((RFPredictYield-TestYield).^2))/size(TestYield,1));
RFrMatrix=corrcoef(RFPredictYield,TestYield);
RFr=RFrMatrix(1,2);
RFRMSEMatrix=[RFRMSEMatrix,RFRMSE];
RFrAllMatrix=[RFrAllMatrix,RFr];
if RFRMSE<1000
    disp(RFRMSE);
    break;
end
disp(RFCycleRun);
str=['Random Forest is Solving...',num2str(100*RFCycleRun/RFRunNumSet),'%'];
waitbar(RFCycleRun/RFRunNumSet,RFScheduleBar,str);
end
close(RFScheduleBar);

%% Variable Importance Contrast
VariableImportanceX={};
XNum=1;
for TifFileNum=1:length(TifFileNames)
    if ~(strcmp(TifFileNames(TifFileNum).name(4:end-4),'MaizeArea') | ...
            strcmp(TifFileNames(TifFileNum).name(4:end-4),'MaizeYield'))
        eval(['VariableImportanceX{1,XNum}=''',TifFileNames(TifFileNum).name(4:end-4),''';']);
        XNum=XNum+1;
    end
end
figure('Name','Variable Importance Contrast');
VariableImportanceX=categorical(VariableImportanceX);
bar(VariableImportanceX,RFModel.OOBPermutedPredictorDeltaError)
xtickangle(45);
set(gca, 'XDir','normal')
xlabel('Factor');
ylabel('Importance');

%% RF Model Storage
RFModelSavePath='G:\CropYield\02_CodeAndMap\00_SavedModel\';
save(sprintf('%sRF0410.mat',RFModelSavePath),'AreaPercent','InputOutput','nLeaf','nTree',...
    'RandomNumber','RFModel','RFPredictConfidenceInterval','RFPredictYield','RFr','RFRMSE',...
    'TestVARI','TestYield','TrainVARI','TrainYield');

%% Notification
% Set breakpoints here.

%% ANN Cycle Preparation
ANNRMSE=9999;
ANNRunNum=0;
ANNRMSEMatrix=[];
ANNrAllMatrix=[];
while ANNRMSE>1000

%% ANN
x=TrainVARI';
t=TrainYield';
trainFcn = 'trainlm';
hiddenLayerSize = [10 10 10];
ANNnet = fitnet(hiddenLayerSize,trainFcn);
ANNnet.input.processFcns = {'removeconstantrows','mapminmax'};
ANNnet.output.processFcns = {'removeconstantrows','mapminmax'};
ANNnet.divideFcn = 'dividerand';
ANNnet.divideMode = 'sample';
ANNnet.divideParam.trainRatio = 0.6;
ANNnet.divideParam.valRatio = 0.4;
ANNnet.divideParam.testRatio = 0.0;
ANNnet.performFcn = 'mse';
ANNnet.trainParam.epochs=5000;
ANNnet.trainParam.goal=0.01;
% For a list of all plot functions type: help nnplot
ANNnet.plotFcns = {'plotperform','plottrainstate','ploterrhist','plotregression','plotfit'};
[ANNnet,tr] = train(ANNnet,x,t);
y = ANNnet(x);
e = gsubtract(t,y);
performance = perform(ANNnet,t,y);
% Recalculate Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(ANNnet,trainTargets,y);
valPerformance = perform(ANNnet,valTargets,y);
testPerformance = perform(ANNnet,testTargets,y);
% view(net)
% Plots
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotregression(t,y)
%figure, plotfit(net,x,t)
% Deployment
% See the help for each generation function for more information.
if (false)
    % Generate MATLAB function for neural network for application
    % deployment in MATLAB scripts or with MATLAB Compiler and Builder
    % tools, or simply to examine the calculations your trained neural
    % network performs.
    genFunction(ANNnet,'myNeuralNetworkFunction');
    y = myNeuralNetworkFunction(x);
end
if (false)
    % Generate a matrix-only MATLAB function for neural network code
    % generation with MATLAB Coder tools.
    genFunction(ANNnet,'myNeuralNetworkFunction','MatrixOnly','yes');
    y = myNeuralNetworkFunction(x);
end
if (false)
    % Generate a Simulink diagram for simulation or deployment with.
    % Simulink Coder tools.
    gensim(ANNnet);
end

%% Accuracy of ANN
ANNPredictYield=sim(ANNnet,TestVARI')';
ANNRMSE=sqrt(sum(sum((ANNPredictYield-TestYield).^2))/size(TestYield,1));
ANNrMatrix=corrcoef(ANNPredictYield,TestYield);
ANNr=ANNrMatrix(1,2);
ANNRunNum=ANNRunNum+1;
ANNRMSEMatrix=[ANNRMSEMatrix,ANNRMSE];
ANNrAllMatrix=[ANNrAllMatrix,ANNr];
disp(ANNRunNum);
end
disp(ANNRMSE);

%% ANN Model Storage
ANNModelSavePath='G:\CropYield\02_CodeAndMap\00_SavedModel\';
save(sprintf('%sRF0417ANN0399.mat',ANNModelSavePath),'AreaPercent','InputOutput','nLeaf','nTree',...
    'RandomNumber','RFModel','RFPredictConfidenceInterval','RFPredictYield','RFr','RFRMSE',...
    'TestVARI','TestYield','TrainVARI','TrainYield','ANNnet','ANNPredictYield','ANNr','ANNRMSE',...
    'hiddenLayerSize');