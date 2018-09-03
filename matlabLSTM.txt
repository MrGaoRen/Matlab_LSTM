sample_size = size(rnn_training_label,1);
shuffle_idx = randperm(sample_size);
data = rnn_training_cell(shuffle_idx);
label = rnn_training_label(shuffle_idx);
split_factor = 0.7;
training_data = data(1:round(split_factor*sample_size));
training_label = label(1:round(split_factor*sample_size));
testing_data = data(round(split_factor*sample_size)+1:end);
testing_label = label(round(split_factor*sample_size)+1:end);

inputSize = 20;
numHiddenUnits = 250;
numClasses = 2;
bias = zeros(4*numHiddenUnits,1);
bias(numHiddenUnits +1 : 2*numHiddenUnits,1) = 1;
layers = [ ...
    sequenceInputLayer(inputSize)
    fullyConnectedLayer(numHiddenUnits)

    reluLayer
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
      
options = trainingOptions('sgdm', ...
            'MaxEpochs',100,...
            'L2Regularization',0.0002);
rnn = trainNetwork(training_data,categorical(training_label) ,layers,options);


%%
%test it on testing set
[updatedNet,testing_res ] = predictAndUpdateState(rnn, testing_data);
%%
[~,testing_prediction] = max(testing_res,[],2);
testing_prediction = testing_prediction';
label_name = [1 2];
testing_prediction = label_name(testing_prediction)';
error = testing_prediction~=testing_label;
disp('error on testing set')
disp(sum(error)/length(error))



confusion_matrix(testing_label',testing_prediction')

%%
function [] = confusion_matrix(T,Y)
T = double(T);
Y = double(Y);
M = size(unique(T),2);
N = size(T,2);
targets = zeros(M,N);
outputs = zeros(M,N);
targetsIdx = sub2ind(size(targets), T, 1:N);
outputsIdx = sub2ind(size(outputs), Y, 1:N);
targets(targetsIdx) = 1;
outputs(outputsIdx) = 1;
% Plot the confusion matrix
plotconfusion(targets,outputs)
end

function [balanced_data,balanced_label] =  balance_data(data, label,dataset_size)
    %4370/94725 is inj, we need to balance it,shrink it to 1:1

    balanced_data = cell(dataset_size,1);
    balanced_label = zeros(dataset_size,1);
    class_1_data = {};
    class_2_data = {};
    sample_size = size(data,1);
    cnt_1 = 1;
    cnt_2 = 1;
    for i = 1:sample_size
        if label(i)==1
            class_1_data(cnt_1) = data(i);
            cnt_1 = cnt_1 + 1;
        else
            class_2_data(cnt_2) = data(i);
            cnt_2 = cnt_2 + 1;            
        end
    end
%     balanced_data(1:cnt_1-1) = class_1_data;
    
    rand_idx = randperm(cnt_2-1);
    class_2_data = class_2_data(rand_idx);
    balanced_data = [class_1_data(1:cnt_1-1) class_2_data(1:dataset_size-cnt_1+1)]';
    balanced_label(1:cnt_1-1) = 1;
    balanced_label(cnt_1:end) = 2;
end

